import time
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input, Activation, BatchNormalization, LSTM, concatenate, Permute, Add, Lambda, dot
import tensorflow as tf
from GraphAttentionLayer import GraphAttention
from scipy.spatial import distance_matrix
import mne
import keras_nlp
from sklearn.metrics.pairwise import rbf_kernel, chi2_kernel

from data_utils import array_to_mesh, get_raw_coordinates


class MultiviewGAT:
    def __init__(self, window_size,conv1_filters,conv2_filters,conv3_filters,
                 conv1_kernel_shape,conv2_kernel_shape,conv3_kernel_shape,
                 padding1,padding2,padding3,conv1_activation,conv2_activation,
                 conv3_activation,dense_nodes,dense_activation,
                 lstm1_cells,lstm2_cells,dense3_nodes,dense3_activation,depth, gamma,
                 gat_heads, encoder_heads):
        
        self.number_classes = 4
        self.number_channels = 248
        self.mesh_rows = 20
        self.mesh_columns = 21        
        
        self.window_size = window_size
        
        self.conv1_filters = conv1_filters
        self.conv2_filters = conv2_filters
        self.conv3_filters = conv3_filters
        
        self.conv1_kernel_shape = conv1_kernel_shape
        self.conv2_kernel_shape = conv2_kernel_shape
        self.conv3_kernel_shape = conv3_kernel_shape
        
        self.padding1 = padding1
        self.padding2 = padding2
        self.padding3 = padding3
        
        self.conv1_activation = conv1_activation
        self.conv2_activation = conv2_activation
        self.conv3_activation = conv3_activation
        
        self.dense_nodes = dense_nodes
        self.dense_activation = dense_activation
        
        self.lstm1_cells = lstm1_cells
        self.lstm2_cells = lstm2_cells
        
        self.dense3_nodes = dense3_nodes
        self.dense3_activation = dense3_activation
        
        self.depth_k = 6
        self.depth_v = 4
        self.num_heads = 2  
        self.relative = False
        
        self.depth = depth
        self.edge_index = self.get_edge_index(gamma)
        self.gamma = gamma
        self.gat_heads = gat_heads
        self.encoder_heads = encoder_heads
        self.model = self.get_model()

    def get_edge_index(self, gamma):
      coords = get_raw_coordinates()

      #RBF 

      kernel = rbf_kernel(coords, gamma = gamma)



      np.set_printoptions(linewidth=500)
      return kernel
       
    def graphAttention(self, x, heads):

      adjacency_matrix = self.edge_index
      node_indices = tf.range(0,248)
      node_indices = tf.reshape(node_indices, [1,248])
      gat_layer = GraphAttention(units=1, attn_heads=heads)

      graph_out = gat_layer((x,node_indices,adjacency_matrix))
      
      return graph_out


    def shape_list(self,x):
      """Return list of dims, statically where possible."""
      static = x.get_shape().as_list()
      shape = tf.shape(x)
      ret = []
      for i, static_dim in enumerate(static):
        dim = static_dim or shape[i]
        ret.append(dim)
      return ret
    
    
    def split_heads_2d(self, inputs, Nh):
      """Split channels into multiple heads."""
      B, H, W, d = self.shape_list(inputs)
      ret_shape = [B, H, W, Nh, d // Nh]
      print(f"{ret_shape} ret_shape shape")
      split = tf.reshape(inputs, ret_shape)
      return tf.transpose(split, [0, 3, 1, 2, 4])
    
    
    def combine_heads_2d(self,inputs):
      """Combine heads (inverse of split heads 2d)."""
      transposed = tf.transpose(inputs, [0, 2, 3, 1, 4])
      Nh, channels = self.shape_list(transposed)[-2:]
      ret_shape = self.shape_list(transposed)[:-2] + [Nh * channels]
      return tf.reshape(transposed, ret_shape)
    
    
    def rel_to_abs(self,x):
      """Converts tensor from relative to aboslute indexing."""
      # [B, Nh, L, 2L1]
      B, Nh, L, _ = self.shape_list(x)
      # Pad to shift from relative to absolute indexing.
      col_pad = tf.zeros((B, Nh, L, 1))
      x = tf.concat([x, col_pad], axis=3)
      flat_x = tf.reshape(x, [B, Nh, L * 2 * L])
      flat_pad = tf.zeros((B, Nh, L-1))
      flat_x_padded = tf.concat([flat_x, flat_pad], axis=2)
      # Reshape and slice out the padded elements.
      final_x = tf.reshape(flat_x_padded, [B, Nh, L+1, 2*L-1])
      final_x = final_x[:, :, :L, L-1:]
      return final_x
    
    
    def relative_logits_1d(self,q, rel_k, H, W, Nh, transpose_mask):
      """Compute relative logits along one dimenion."""
      rel_logits = tf.einsum('bhxyd,md->bhxym', q, rel_k)
      # Collapse height and heads
      rel_logits = tf.reshape(rel_logits, [-1, Nh * H, W, 2 * W-1])
      rel_logits = self.rel_to_abs(rel_logits)
      # Shape it and tile height times
      rel_logits = tf.reshape(rel_logits, [-1, Nh, H, W, W])
      rel_logits = tf.expand_dims(rel_logits, axis=3)
      rel_logits = tf.tile(rel_logits, [1, 1, 1, H, 1, 1])
      # Reshape for adding to the logits.
      rel_logits = tf.transpose(rel_logits, transpose_mask)
      rel_logits = tf.reshape(rel_logits, [-1, Nh, H*W, H*W])
      return rel_logits
    
    
    def relative_logits(self,q, H, W, Nh, dkh):
      """Compute relative logits."""
      # Relative logits in width dimension first.
      rel_embeddings_w = tf.get_variable('r_width', shape=(2*W - 1, dkh),initializer=tf.random_normal_initializer(dkh**-0.5)) # [B, Nh, HW, HW]
      rel_logits_w = self.relative_logits_1d(q, rel_embeddings_w, H, W, Nh, [0, 1, 2, 4, 3, 5])
      # Relative logits in height dimension next.
      # For ease, we 1) transpose height and width,
      # 2) repeat the above steps and
      # 3) transpose to eventually put the logits
      # in their right positions.
      rel_embeddings_h = tf.get_variable('r_height', shape=(2 * H - 1, dkh),initializer=tf.random_normal_initializer(dkh**-0.5))
      # [B, Nh, HW, HW]
      rel_logits_h = self.relative_logits_1d(tf.transpose(q, [0, 1, 3, 2, 4]),rel_embeddings_h, W, H, Nh, [0, 1, 4, 2, 5, 3])
      return rel_logits_h, rel_logits_w
    
    
    def self_attention_2d(self,inputs, dk, dv, Nh, relative=True):
      """2d relative selfattention."""
      _, H, W, _ = self.shape_list(inputs) #H = 20, W = 21 (meshes)
      dkh = dk // Nh #(6 / 2) = 3
      dvh = dv // Nh #(4 / 2) = 2
      flatten_hw = lambda x, d: tf.reshape(x, [-1, Nh, H*W, d])
      # Compute q, k, v
      kqv = Conv2D(2 * dk + dv, 1)(inputs) #Conv with dk + dk + dv filters (16)
      k, q, v = tf.split(kqv, [dk, dk, dv], axis=3)
      q *= dkh ** -0.5 # scaled dotproduct
      # After splitting, shape is [B, Nh, H, W, dkh or dvh]
      q = self.split_heads_2d(q, Nh)
      k = self.split_heads_2d(k, Nh)
      v = self.split_heads_2d(v, Nh)
      # [B, Nh, HW, HW]
      logits = tf.matmul(flatten_hw(q, dkh), flatten_hw(k, dkh),transpose_b=True)
      if relative:
        rel_logits_h, rel_logits_w = self.relative_logits(q, H, W, Nh,dkh)
        logits += rel_logits_h
        logits += rel_logits_w
      weights = tf.nn.softmax(logits)
      attn_out = tf.matmul(weights, flatten_hw(v, dvh))
      attn_out = tf.reshape(attn_out, [-1, Nh, H, W, dvh])
      attn_out = self.combine_heads_2d(attn_out)
      # Project heads
      attn_out = Conv2D(dv, 1)(attn_out)
      return attn_out
    
    
    def tfaugmented_conv2d(self,X, Fout, k, dk, dv, Nh, relative):
        if Fout - dv < 0:
            filters = 1
        else:
            filters = Fout - dv
        time.sleep(1)
        conv_out = Conv2D(filters=filters,kernel_size=k, padding='same')(X)
        print(f"{conv_out.shape} (conv_out-shape)")
        attn_out = self.self_attention_2d(X, dk, dv, Nh, relative=relative)
        print(f"{attn_out.shape} (attn_out-shape)")
        return tf.concat([conv_out, attn_out], axis=3)
        
    def compute_output_shape(self, input_shape):
        return input_shape

    def encoder_block(self, inputs, heads):
        encoderLayer = keras_nlp.layers.TransformerEncoder(
           248,
           heads,
        )
        
        output = encoderLayer(inputs)
        return output

    def attention_block(self,hidden_states):
        hidden_size = int(hidden_states.shape[2])
        score_first_part = Dense(hidden_size, use_bias=False, name='attention_score_vec')(hidden_states)
        h_t = Lambda(lambda x: x[:, -1, :], output_shape=(hidden_size,), name='last_hidden_state')(hidden_states)
        score = dot([score_first_part, h_t], [2, 1], name='attention_score')
        attention_weights = Activation('softmax', name='attention_weight')(score)
        context_vector = dot([hidden_states, attention_weights], [1, 1], name='context_vector')
        pre_activation = concatenate([context_vector, h_t], name='attention_output')
        attention_vector = Dense(128, use_bias=False, activation='tanh', name='attention_vector')(pre_activation)
        return attention_vector

    def get_model(self):
        inputs_lstm = []
        
        outputs_cnn = []
        # inputs_enc = tf.zeros([None, 0])
        encoder = []
        added = None

        for i in range(self.window_size):
            input= Input(shape=(self.number_channels,self.depth), name = "input"+str(i+1))

            inputs_lstm.append(input)
            
            gat = self.graphAttention(input, self.gat_heads) #Bx248x3

            flat = Flatten()(gat)
            dense = Dense(self.dense_nodes, activation=self.dense_activation)(flat)
            outputs_cnn.append(dense)
            # dense = Dense(64, activation=self.dense_activation)(dense)
            # dense = tf.expand_dims(dense, axis=1)

            # if(i == 0):
            #     inputs_enc = dense
            # else:
            #     inputs_enc = tf.concat([inputs_enc, dense], axis=1)
                # inputs_enc = tf.squeeze(inputs_enc)
            # print(inputs_enc.shape)

            # # print(dense.shape)
            # a = tf.expand_dims(dense, axis=1)
            # # print(a.shape) 
            # res = self.encoder_block(a) # B x SEQ_len x Input_len
            # print(res.shape) 
            # outputs_cnn.append(gat) # L x B x SEQ_len x Input_len (10x64x16x744)

            # permut = Permute((2,1), input_shape=(self.number_channels,1))(input)
            # dense2 = Dense(self.dense_nodes, activation=self.dense_activation, input_shape=(1, self.number_channels))(input)
            # print(dense2.shape)
            encoder.append(input)
            
        # #LSTM
        # merge = concatenate(encoder,axis=1)
        # print(merge.shape)
        # lstm1 = LSTM(self.lstm1_cells, return_sequences=True)(merge)
        # print(lstm1.shape)

        # attention_output = self.attention_block(lstm1) 
        # print(attention_output.shape)
        # attention_output = tf.expand_dims(attention_output,-1)

        # print(attention_output.shape)
        # lstm2 = LSTM(self.lstm2_cells, return_sequences=False)(attention_output)

        # print(lstm2.shape)
        # dense3 = Dense(self.dense3_nodes, activation=self.dense3_activation)(lstm2)

        # second stream
        merge = concatenate(encoder,axis=-1) #(B, 100, 248)
        print(merge.shape)
        # permut = Permute((2,1), input_shape=(100, self.number_channels*self.depth))(merge) #(B, 248, 100)
        # print(permut.shape)
        encoder1 = self.encoder_block(merge, self.encoder_heads)  #(B, 100, 248)
        
        print(encoder1.shape)
        dense7 = Dense(3, activation=self.dense3_activation)(encoder1) #(B, 744, 100)
        print(dense7.shape)
        flatten8 = Flatten()(dense7)
        print(flatten8.shape)



        # attention_output = self.attention_block(lstm1) 
        # attention_output = tf.expand_dims(attention_output,-1)
        
        # lstm2 = LSTM(self.lstm2_cells, return_sequences=False)(attention_output)
        
        # dense3 = Dense(self.dense3_nodes, activation=self.dense3_activation)(lstm2)
        # output = Dense(4, activation="softmax")(encoder1)

        
        added = Add()([i for i in outputs_cnn])
        
        # concat = concatenate(inputs_enc,axis=1)
        # print(concat.shape)
        # aa = tf.expand_dims(concat,-1)
        # print(aa.shape)
        # final = self.encoder_block(inputs_enc, self.encoder_heads, self.gat_heads)
        # print(final.shape)
        # final = tf.squeeze(final, 1)
        # final = Flatten()(final)
        # print(concat.shape)
        # print(final.shape)
        # print(added.shape) # B x SEQ_len x Input_len (64x16x744)
        # concat = Flatten()(added)
        # print(concat.shape) # B x Input_len (64x11904)

        print(added.shape)
        final = concatenate([flatten8,added], axis=-1)
        # print(final.shape)
        output = Dense(4, activation="softmax")(final)
        # print(output.shape) 
        model = Model(inputs=inputs_lstm, outputs=output)
        return model
