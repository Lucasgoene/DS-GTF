import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Input, concatenate, Add
import tensorflow as tf
from GraphAttentionLayer import GraphAttention
import keras_nlp
from sklearn.metrics.pairwise import rbf_kernel

from data_utils import get_raw_coordinates


class MultiviewGAT:
    def __init__(self, window_size,dense_activation,
                 dense3_activation,depth, gamma,gat_heads,
                 encoder_heads, threshold, mode, k):
        
        self.number_classes = 4
        self.number_channels = 248        
        self.window_size = window_size
        self.depth = depth
        self.threshold = threshold

        self.dense_nodes = 248*gat_heads
        self.dense_activation = dense_activation        
        self.dense3_nodes = 248*gat_heads
        self.dense3_activation = dense3_activation
        
        self.depth = depth
        self.threshold = threshold

        self.gamma = gamma
        self.gat_heads = gat_heads
        self.encoder_heads = encoder_heads
        self.edge_index = self.get_edge_index(gamma, threshold, mode, k)

        self.model = self.get_model()

    def get_edge_index(self, gamma, threshold = 0.997, mode = "none", k=3):
        coords = get_raw_coordinates()

        # kernel = np.zeros((248,248))
        kernel = rbf_kernel(coords, gamma=gamma)

        if(mode == "topk"):
          for channel in kernel:
              # indices = np.argpartition(kernel[i],-k)[-k:]
              indices = np.argpartition(channel,-k)[-k:]
              mask = np.ones(248, dtype=bool)
              mask[indices] = False
              channel[mask] = 0.0
              channel[~mask] = 1.0
        if(mode == "treshold"):
          kernel = [edge_weight if edge_weight > threshold else 0 for edge_weight in kernel]
        if(mode == "none"):
          kernel = np.ones((248,248))

        print(np.count_nonzero(kernel))
        return kernel
       
    def graphAttention(self, x, heads, name):
      adjacency_matrix = self.edge_index
      node_indices = tf.range(0,248)
      node_indices = tf.reshape(node_indices, [1,248])
      gat_layer = GraphAttention(units=1, attn_heads=heads, name=name, trainable=True)

      graph_out = gat_layer((x,node_indices,adjacency_matrix))
      
      return graph_out

    def encoder_block(self, inputs, heads):
        encoderLayer = keras_nlp.layers.TransformerEncoder(
           248,
           heads,
           name="encoder_block",
           trainable=True
        )
        
        output = encoderLayer(inputs)
        return output

    def get_model(self):
        inputs_lstm = []
        
        outputs_cnn = []
        encoder = []
        added = None

        for i in range(self.window_size):
            input= Input(shape=(self.number_channels,self.depth), name = "input"+str(i+1))

            inputs_lstm.append(input)
            
            gat = self.graphAttention(input, self.gat_heads, name="gat_{}".format(i+1)) #Bx248x3
            flat = Flatten()(gat)
            dense = Dense(self.dense_nodes, activation=self.dense_activation, name="gat_dense_{}".format(i+1), trainable=True)(flat)

            outputs_cnn.append(dense)
            encoder.append(input)
            
        # Temporal stream
        merge = concatenate(encoder,axis=-1)
        encoder1 = self.encoder_block(merge, self.encoder_heads)
        dense7 = Dense(3, activation=self.dense3_activation, name="encoder_dense", trainable=True)(encoder1) 
        flatten8 = Flatten()(dense7)

        # Add Spatial stream
        added = Add()([i for i in outputs_cnn])

        # Combine streams
        final = concatenate([flatten8,added], axis=-1)


        # Generate outputs
        output = Dense(4, activation="softmax")(final)
        model = Model(inputs=inputs_lstm, outputs=output)
        return model
