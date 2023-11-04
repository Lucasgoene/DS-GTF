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
                 lstm1_cells,lstm2_cells,dense3_nodes,dense3_activation,depth, gamma):
        
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
        self.model = self.get_model()

    def get_edge_index(self, gamma):
      coords = get_raw_coordinates()

      #RBF 

      kernel = rbf_kernel(coords, gamma = gamma)



      np.set_printoptions(linewidth=500)
      return kernel
       
    def graphAttention(self, x):

      adjacency_matrix = self.edge_index
      node_indices = tf.range(0,248)
      node_indices = tf.reshape(node_indices, [1,248])
      gat_layer = GraphAttention(units=1, attn_heads=3)

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
      print(f"{dkh} - {dvh} -- dkhdvh")
      flatten_hw = lambda x, d: tf.reshape(x, [-1, Nh, H*W, d])
      # Compute q, k, v
      kqv = Conv2D(2 * dk + dv, 1)(inputs) #Conv with dk + dk + dv filters (16)
      print(f"{kqv.shape} kqv shape") 
      k, q, v = tf.split(kqv, [dk, dk, dv], axis=3)
      q *= dkh ** -0.5 # scaled dotproduct
      print(f"{q.shape} q* shape")
      # After splitting, shape is [B, Nh, H, W, dkh or dvh]
      q = self.split_heads_2d(q, Nh)
      print(f"{q.shape} q shape (3)")
      k = self.split_heads_2d(k, Nh)
      print(f"{q.shape} q shape (3)")
      v = self.split_heads_2d(v, Nh)
      print(f"{q.shape} q shape (None, 2, 20, 21, 3)")
      # [B, Nh, HW, HW]
      print(f"{flatten_hw(q, dkh).shape} (None, 2, 420, 3)" )
      logits = tf.matmul(flatten_hw(q, dkh), flatten_hw(k, dkh),transpose_b=True)
      if relative:
        rel_logits_h, rel_logits_w = self.relative_logits(q, H, W, Nh,dkh)
        logits += rel_logits_h
        logits += rel_logits_w
      weights = tf.nn.softmax(logits)
      print(f"{weights.shape} weight shape")
      print(f"{flatten_hw(v, dvh).shape} v dvh shape")
      attn_out = tf.matmul(weights, flatten_hw(v, dvh))
      print(f"{attn_out.shape} attn_out shape")
      attn_out = tf.reshape(attn_out, [-1, Nh, H, W, dvh])
      print(f"{attn_out.shape} attn_out reshape")
      attn_out = self.combine_heads_2d(attn_out)
      print(f"{attn_out.shape} attn_out combined reshape")
      # Project heads
      attn_out = Conv2D(dv, 1)(attn_out)
      print("==================")
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

    def encoder_block(self, inputs):
        
        encoder = Encoder(2, 744, 8, 1488, 16, training=True)

        # encoderLayer = keras_nlp.layers.TransformerEncoder(
        #    744,
        #    8,
        # )
        # output = encoderLayer(inputs)
        return encoder([inputs, None])

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
        encoder = []
        added = None

        for i in range(self.window_size):
            input= Input(shape=(self.number_channels,self.depth), name = "input"+str(i+1))

            inputs_lstm.append(input)
            
            gat = self.graphAttention(input) #Bx248x3


            flat = Flatten()(gat)
            dense = Dense(self.dense_nodes, activation=self.dense_activation)(flat)
            # outputs_cnn.append(dense)
            # print(dense.shape)
            a = tf.expand_dims(dense, axis=1)
            # print(a.shape) 
            res = self.encoder_block(a) # B x SEQ_len x Input_len
            # print(res.shape) 
            outputs_cnn.append(res) # L x B x SEQ_len x Input_len (10x64x16x744)

            permut = Permute((2,1), input_shape=(self.number_channels,1))(input)
            dense2 = Dense(self.dense_nodes, activation=self.dense_activation, input_shape=(1,self.number_channels))(permut)
            encoder.append(dense2)
        
        # merge = concatenate(encoder,axis=1)
        # print(merge.shape)
        # encoder1 = self.attention_block(merge)
        
        # attention_output = self.attention_block(lstm1) 
        # attention_output = tf.expand_dims(attention_output,-1)
        
        # lstm2 = LSTM(self.lstm2_cells, return_sequences=False)(attention_output)
        
        # dense3 = Dense(self.dense3_nodes, activation=self.dense3_activation)(lstm2)
        # output = Dense(4, activation="softmax")(encoder1)

        
        added = Add()([i for i in outputs_cnn])
        # print(added.shape) # B x SEQ_len x Input_len (64x16x744)
        concat = Flatten()(added)
        # print(concat.shape) # B x Input_len (64x11904)

        
        # final = concatenate([encoder1,added], axis=-1)
        output = Dense(4, activation="softmax")(concat)
        # print(output.shape) 
        model = Model(inputs=inputs_lstm, outputs=output)
        return model


class Encoder(tf.keras.layers.Layer):
    def __init__(self, n_layers, d_model, num_heads, middle_units,
                max_seq_len, epsilon=1e-6, dropout_rate=0.1, training=False, **kwargs):
        super(Encoder, self).__init__(**kwargs)

        self.n_layers = n_layers
        self.d_model = d_model
        self.pos_embedding = PositionalEncoding(sequence_len=max_seq_len, embedding_dim=d_model)

        self.encode_layer = [EncoderLayer(d_model=d_model, num_heads=num_heads, 
                                          middle_units=middle_units, 
                                          epsilon=epsilon, dropout_rate=dropout_rate,
                                          training = training)
                            for _ in range(n_layers)]
        
    def call(self, inputs, **kwargs):
        emb, mask = inputs
        emb = self.pos_embedding(emb)
        
        for i in range(self.n_layers):
            emb = self.encode_layer[i](emb, mask)

        return emb





# 编码层
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, middle_units, \
                 epsilon=1e-6, dropout_rate=0.1, training=False, **kwargs):
        super(EncoderLayer, self).__init__(**kwargs)
        
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, middle_units)
        
        self.layernorm1 = LayerNormalization()
        self.layernorm2 = LayerNormalization()
        
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
        
        self.training = training
        
    def call(self, inputs, mask, **kwargs):

        # 多头注意力网络
        att_output = self.mha([inputs, inputs, inputs, mask])
        att_output = self.dropout1(att_output, training=self.training)
        out1 = self.layernorm1(inputs + att_output)  # (batch_size, input_seq_len, d_model)
        
        # 前向网络
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=self.training)
        out2 = self.layernorm2(out1 + ffn_output)   # (batch_size, input_seq_len, d_model)
        
        return out2




# 层标准化
class LayerNormalization(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-6, **kwargs):
        self.eps = epsilon
        super(LayerNormalization, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.gamma = self.add_weight(name='gamma', shape=input_shape[-1:],
                                     initializer=tf.ones_initializer(), trainable=True)
        self.beta = self.add_weight(name='beta', shape=input_shape[-1:],
                                    initializer=tf.zeros_initializer(), trainable=True)
        super(LayerNormalization, self).build(input_shape)
        
    def call(self, x):
        mean = tf.keras.backend.mean(x, axis=-1, keepdims=True)
        std = tf.keras.backend.std(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta
    
    def compute_output_shape(self, input_shape):
        return input_shape


# 前向网络
def point_wise_feed_forward_network(d_model, middle_units):
    
    return tf.keras.Sequential([
        tf.keras.layers.Dense(middle_units, activation='relu'),
        tf.keras.layers.Dense(d_model, activation='relu')])



# dot attention
def scaled_dot_product_attention(q, k, v, mask):
    
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    dim_k = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dim_k)
    
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(attention_weights, v)

    return output



# 构造 multi head attention 层
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.d_model = d_model

        # d_model 必须可以正确分为各个头
        assert d_model % num_heads == 0
        
        # 分头后的维度
        self.depth = d_model // num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)
        
        self.dot_attention = scaled_dot_product_attention

    def split_heads(self, x, batch_size):
        # 分头, 将头个数的维度 放到 seq_len 前面
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs, **kwargs):
        q, k, v, mask = inputs
        batch_size = tf.shape(q)[0]

        # 分头前的前向网络，获取q、k、v语义
        q = self.wq(q) # (batch_size, seq_len, d_model)
        k = self.wk(k)
        v = self.wv(v)

        # 分头
        q = self.split_heads(q, batch_size) # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size) # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size) # (batch_size, num_heads, seq_len_v, depth)
        
        # 通过缩放点积注意力层
        scaled_attention = self.dot_attention(q, k, v, mask) # (batch_size, num_heads, seq_len_q, depth)
        
        # “多头维度” 后移
        scaled_attention = tf.transpose(scaled_attention, [0, 2, 1, 3]) # (batch_size, seq_len_q, num_heads, depth)

        # 合并 “多头维度”
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))

        # 全连接层
        output = self.dense(concat_attention)
        
        return output
    


# mask功能
def padding_mask(seq):
    # 获取为 0的padding项
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # 扩充维度用于attention矩阵
    return seq[:, np.newaxis, np.newaxis, :] # (batch_size, 1, 1, seq_len)




# 位置编码
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, sequence_len=None, embedding_dim=None, **kwargs):
        self.sequence_len = sequence_len
        self.embedding_dim = embedding_dim
        super(PositionalEncoding, self).__init__(**kwargs)

    def call(self, inputs):
        if self.embedding_dim == None:
            self.embedding_dim = int(inputs.shape[-1])
        
        position_embedding = np.array([
            [pos / np.power(10000, 2. * i / self.embedding_dim) for i in range(self.embedding_dim)]
            for pos in range(self.sequence_len)])

        position_embedding[:, 0::2] = np.sin(position_embedding[:, 0::2])  # dim 2i
        position_embedding[:, 1::2] = np.cos(position_embedding[:, 1::2])  # dim 2i+1
        
        position_embedding = tf.cast(position_embedding, dtype=tf.float32)
        
        return position_embedding + inputs
