import time
from tensorflow.keras import backend as K
from tensorflow.keras import activations, constraints, initializers, regularizers
from tensorflow.keras.layers import Layer, Dropout, LeakyReLU
from tensorflow.keras.backend import transpose
import tensorflow as tf

class GraphAttention(Layer):
    """
    Graph Attention (GAT) layer. The base implementation is taken from
    https://github.com/danielegrattarola/keras-gat,
    with some modifications added for ease of use.

    Based on the original paper: Graph Attention Networks. P. Velickovic et al. ICLR 2018 https://arxiv.org/abs/1710.10903

    Notes:
      - The inputs are tensors with a batch dimension of 1:
        Keras requires this batch dimension, and for full-batch methods
        we only have a single "batch".

      - There are three inputs required, the node features, the output
        indices (the nodes that are to be selected in the final layer)
        and the graph adjacency matrix

      - This does not add self loops to the adjacency matrix, you should preprocess
        the adjacency matrix to add self-loops

      - The output indices are used when ``final_layer=True`` and the returned outputs
        are the final-layer features for the nodes indexed by output indices.

      - If ``final_layer=False`` all the node features are output in the same ordering as
        given by the adjacency matrix.

    Args:
        F_out (int): dimensionality of output feature vectors
        attn_heads (int or list of int): number of attention heads
        attn_heads_reduction (str): reduction applied to output features of each attention head, 'concat' or 'average'.
            'Average' should be applied in the final prediction layer of the model (Eq. 6 of the paper).
        in_dropout_rate (float): dropout rate applied to features
        attn_dropout_rate (float): dropout rate applied to attention coefficients
        activation (str): nonlinear activation applied to layer's output to obtain output features (eq. 4 of the GAT paper)
        final_layer (bool): If False the layer returns output for all nodes,
                            if True it returns the subset specified by the indices passed to it.
        use_bias (bool): toggles an optional bias
        saliency_map_support (bool): If calculating saliency maps using the tools in
            stellargraph.utils.saliency_maps this should be True. Otherwise this should be False (default).
        kernel_initializer (str or func): The initialiser to use for the head weights;
            defaults to 'glorot_uniform'.
        kernel_regularizer (str or func): The regulariser to use for the head weights;
            defaults to None.
        kernel_constraint (str or func): The constraint to use for the head weights;
            defaults to None.
        bias_initializer (str or func): The initialiser to use for the head bias;
            defaults to 'zeros'.
        bias_regularizer (str or func): The regulariser to use for the head bias;
            defaults to None.
        bias_constraint (str or func): The constraint to use for the head bias;
            defaults to None.
        attn_kernel_initializer (str or func): The initialiser to use for the attention weights;
            defaults to 'glorot_uniform'.
        attn_kernel_regularizer (str or func): The regulariser to use for the attention weights;
            defaults to None.
        attn_kernel_constraint (str or func): The constraint to use for the attention weights;
            defaults to None.
    """

    def __init__(
        self,
        units,
        attn_heads=1,
        attn_heads_reduction="concat",  # {'concat', 'average'}
        in_dropout_rate=0.0,
        attn_dropout_rate=0.0,
        activation="relu",
        use_bias=True,
        final_layer=False,
        saliency_map_support=False,
        **kwargs,
    ):

        if attn_heads_reduction not in {"concat", "average"}:
            raise ValueError(
                "{}: Possible heads reduction methods: concat, average; received {}".format(
                    type(self).__name__, attn_heads_reduction
                )
            )

        self.units = units  # Number of output features (F' in the paper)
        self.attn_heads = attn_heads  # Number of attention heads (K in the paper)
        self.attn_heads_reduction = attn_heads_reduction  # Eq. 5 and 6 in the paper
        self.in_dropout_rate = in_dropout_rate  # dropout rate for node features
        self.attn_dropout_rate = attn_dropout_rate  # dropout rate for attention coefs
        self.activation = activations.get(activation)  # Eq. 4 in the paper
        self.use_bias = use_bias
        self.final_layer = final_layer

        self.saliency_map_support = saliency_map_support
        # Populated by build()
        self.kernels = []  # Layer kernels for attention heads
        self.biases = []  # Layer biases for attention heads
        self.attn_kernels = []  # Attention kernels for attention heads

        if attn_heads_reduction == "concat":
            # Output will have shape (..., K * F')
            self.output_dim = self.units * self.attn_heads
        else:
            # Output will have shape (..., F')
            self.output_dim = self.units

        self._get_regularisers_from_keywords(kwargs)

        super().__init__(**kwargs)

    def _get_regularisers_from_keywords(self, kwargs):
        initializer = initializers.GlorotUniform(seed=0)

        self.kernel_initializer = initializer

        self.kernel_regularizer = regularizers.get(
            kwargs.pop("kernel_regularizer", None)
        )
        self.kernel_constraint = constraints.get(kwargs.pop("kernel_constraint", None))

        self.bias_initializer = initializer

        self.bias_regularizer = regularizers.get(kwargs.pop("bias_regularizer", None))
        self.bias_constraint = constraints.get(kwargs.pop("bias_constraint", None))

        self.attn_kernel_initializer = initializer
        
        self.attn_kernel_regularizer = regularizers.get(
            kwargs.pop("attn_kernel_regularizer", None)
        )
        self.attn_kernel_constraint = constraints.get(
            kwargs.pop("attn_kernel_constraint", None)
        )

    def get_config(self):
        """
        Gets class configuration for Keras serialization

        """
        config = {
            "units": self.units,
            "attn_heads": self.attn_heads,
            "attn_heads_reduction": self.attn_heads_reduction,
            "in_dropout_rate": self.in_dropout_rate,
            "attn_dropout_rate": self.attn_dropout_rate,
            "activation": activations.serialize(self.activation),
            "use_bias": self.use_bias,
            "final_layer": self.final_layer,
            "saliency_map_support": self.saliency_map_support,
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
            "kernel_constraint": constraints.serialize(self.kernel_constraint),
            "bias_initializer": initializers.serialize(self.bias_initializer),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer),
            "bias_constraint": constraints.serialize(self.bias_constraint),
            "attn_kernel_initializer": initializers.serialize(
                self.attn_kernel_initializer
            ),
            "attn_kernel_regularizer": regularizers.serialize(
                self.attn_kernel_regularizer
            ),
            "attn_kernel_constraint": constraints.serialize(
                self.attn_kernel_constraint
            ),
        }
        base_config = super().get_config()
        return {**base_config, **config}

    def compute_output_shape(self, input_shapes):
        """
        Computes the output shape of the layer.
        Assumes the following inputs:

        Args:
            input_shapes (tuple of ints)
                Shape tuples can include None for free dimensions, instead of an integer.

        Returns:
            An input shape tuple.
        """
        feature_shape, out_shape, *As_shapes = input_shapes

        batch_dim = feature_shape[0]
        if self.final_layer:
            out_dim = out_shape[1]
        else:
            out_dim = feature_shape[1]

        return batch_dim, out_dim, self.output_dim

    def build(self, input_shapes ):
        """
        Builds the layer

        Args:
            input_shapes (list of int): shapes of the layer's inputs (node features and adjacency matrix)

        """
        feat_shape = input_shapes[0]
        input_dim = int(feat_shape[-1])
        batch_dim = feat_shape[0]

        # Variables to support integrated gradients
        self.delta = self.add_weight(
            name="ig_delta", shape=(), trainable=False, initializer=initializers.ones()
        )
        self.non_exist_edge = self.add_weight(
            name="ig_non_exist_edge",
            shape=(),
            trainable=False,
            initializer=initializers.zeros(),
        )

        # Initialize weights for each attention head
        for head in range(self.attn_heads):
            # Layer kernel
            kernel = self.add_weight(
                shape=(input_dim, self.units),
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
                name="kernel_{}".format(head),
            )
            self.kernels.append(kernel)

            # # Layer bias
            if self.use_bias:
                bias = self.add_weight(
                    shape=(self.units,),
                    initializer=self.bias_initializer,
                    regularizer=self.bias_regularizer,
                    constraint=self.bias_constraint,
                    name="bias_{}".format(head),
                )
                self.biases.append(bias)

            # Attention kernels
            attn_kernel_self = self.add_weight(
                shape=(self.units, 1),
                initializer=self.attn_kernel_initializer,
                regularizer=self.attn_kernel_regularizer,
                constraint=self.attn_kernel_constraint,
                name="attn_kernel_self_{}".format(head),
            )
            attn_kernel_neighs = self.add_weight(
                shape=(self.units, 1),
                initializer=self.attn_kernel_initializer,
                regularizer=self.attn_kernel_regularizer,
                constraint=self.attn_kernel_constraint,
                name="attn_kernel_neigh_{}".format(head),
            )
            self.attn_kernels.append([attn_kernel_self, attn_kernel_neighs])
        self.built = True

    def map_call(self, inputs, A, N, out_indices):
        # X = tf.reshape(inputs, [1, inputs.shape[0], inputs.shape[1]])
        X = inputs
        X_dim, n_nodes, _ = K.int_shape(X)
        # if X_dim != 1:
        #     raise ValueError(
        #         "Currently full-batch methods only support a batch dimension of one"
        #     )

        # else:
        #     # Remove singleton batch dimension
        #     X = K.squeeze(X, 0)
        #     out_indices = K.squeeze(out_indices, 0)

        batch_outputs = []
        for head in range(self.attn_heads):
            kernel = self.kernels[head]  # W in the paper (1 x F x F')
            #kernel = tf.reshape(kernel, [X_dim, kernel.shape[0], kernel.shape[1]])
            attention_kernel = self.attn_kernels[head]  # Attention kernel a in the paper (1 x 2F' x 1)

            features = tf.matmul(X, kernel)  # (N x F')
            
            # Compute feature combinations
            # Note: [[a_1], [a_2]]^T [[Wh_i], [Wh_2]] = [a_1]^T [Wh_i] + [a_2]^T [Wh_j]
            self_attention_kernel = attention_kernel[0]
            attn_for_self = tf.matmul(
                features, tf.transpose(self_attention_kernel)
            )  # (N x 1), [a_1]^T [Wh_i]
            attn_for_neighs = tf.matmul(
                features, tf.transpose(attention_kernel[1])
            )  # (N x 1), [a_2]^T [Wh_j]

            # Attention head a(Wh_i, Wh_j) = a^T [[Wh_i], [Wh_j]]
            attn_for_neighs_transpose = tf.transpose(attn_for_neighs, perm=[0,2,1])
            dense = attn_for_self + attn_for_neighs_transpose # (N x N) via broadcasting

            # Add nonlinearity
            dense = LeakyReLU(alpha=0.2)(dense)

            # Mask values before activation (Vaswani et al., 2017)
            # YT: this only works for 'binary' A, not for 'weighted' A!
            # YT: if A does not have self-loops, the node itself will be masked, so A should have self-loops


            # YT: this is ensured by setting the diagonal elements of A tensor to 1 above
            if not self.saliency_map_support:
                mask = -10e9 * (1.0 - A)
                self.A = A
                dense += mask
                dense = K.softmax(dense)  # (N x N), Eq. 3 of the paper
            else:
                # dense = dense - tf.reduce_max(dense)
                # GAT with support for saliency calculations
                W = (self.delta * A) * K.exp(
                    dense - K.max(dense, axis=1, keepdims=True)
                ) * (1 - self.non_exist_edge) + self.non_exist_edge * (
                    A
                    + self.delta * (K.ones(shape=[N, N], dtype="float") - A)
                    + K.eye(N)
                ) * K.exp(
                    dense - K.max(dense, axis=1, keepdims=True)
                )
                dense = W / K.sum(W, axis=1, keepdims=True)

            # Apply dropout to features and attention coefficients
            dropout_feat = Dropout(self.in_dropout_rate)(features)  # (N x F')
            dropout_attn = Dropout(self.attn_dropout_rate)(dense)  # (N x N)

            # Linear combination with neighbors' features [YT: see Eq. 4]
            node_features = K.batch_dot(dropout_attn, dropout_feat)  # (N x F')

            if self.use_bias:
                node_features = K.bias_add(node_features, self.biases[head])

            # Add output of attention head to final output
            batch_outputs.append(node_features)

        # Aggregate the heads' output according to the reduction method
        if self.attn_heads_reduction == "concat":
            batch_output = tf.concat(batch_outputs, axis=2)  # (N x KF')
        else:
            batch_output = K.mean(K.stack(batch_outputs), axis=0)  # N x F')

        # Nonlinear activation function
        batch_output = self.activation(batch_output)

        # On the final layer we gather the nodes referenced by the indices
        if self.final_layer:
            batch_output = K.gather(batch_output, out_indices)

        return batch_output

    def call(self, inputs):
        """
        Creates the layer as a Keras graph.

        Note that the inputs are tensors with a batch dimension of 1:
        Keras requires this batch dimension, and for full-batch methods
        we only have a single "batch".

        There are three inputs required, the node features, the output
        indices (the nodes that are to be selected in the final layer)
        and the graph adjacency matrix

        Notes:
            This does not add self loops to the adjacency matrix.
            The output indices are only used when ``final_layer=True``

        Args:
            inputs (list): list of inputs with 3 items:
            node features (size 1 x N x F),
            output indices (size 1 x M),
            graph adjacency matrix (size N x N),
            where N is the number of nodes in the graph,
                  F is the dimensionality of node features
                  M is the number of output nodes
        """

        BATCHED_X = inputs[0]  # Node features (B x N x F)
        out_indices = inputs[1]  # output indices (1 x K)
        A = inputs[2]  # Adjacency matrix (N x N)
        N = K.int_shape(A)[-1] #

        batch_dim, _, _ = K.int_shape(BATCHED_X)
        result = self.map_call(BATCHED_X, A, N, out_indices)
        #result = tf.map_fn(lambda x: self.map_call(x, A, N, out_indices), BATCHED_X)

        return result