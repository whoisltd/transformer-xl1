import tensorflow as tf
from tensorflow.python.keras.layers import Dropout
import numpy as np

class RelativeMultiHeadAttention(tf.keras.layers.Layer):
    """
    Transformer XL Relative Multi-Head Attention Layer
    """

    def __init__(self, d_model = 512, num_heads = 6, dropout_rate=0.1, **kwargs):
        super(RelativeMultiHeadAttention, self).__init__(**kwargs)
        self.d_model = d_model #number of features per head
        self.num_heads = num_heads #number of heads
        self.dropout_rate = dropout_rate
        self.P = 2 ** 12 # number of relative positions

        self.q_dense = tf.keras.layers.Dense(self.d_model, use_bias=False, name='q')
        self.k_dense = tf.keras.layers.Dense(self.d_model, use_bias=False, name='k')
        self.v_dense = tf.keras.layers.Dense(self.d_model, use_bias=False, name='v')
        self.r_dense = tf.keras.layers.Dense(self.d_model, use_bias=False, name='r')
        self.o_dense = tf.keras.layers.Dense(self.d_model, use_bias=False, name='o')
        self.dropout = Dropout(self.dropout_rate)

    def call(self, inputs, pos_emb, r_w_bias, r_r_bias, mems, training, **kwargs):
        q, k, v, r = inputs
        batch_size = tf.shape(q)[0]
        # these transform the query, key, value vectors for multi-head attention
        q = self.q_dense(q)
        k = self.k_dense(k)
        v = self.v_dense(v)
        r = self.r_dense(r)
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        # r = self.split_heads(r, batch_size)
        scaled_attention = self.scaled_dot_product_attention(q, k, v)
        scaled_attention = tf.transpose(scaled_attention, [0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
        output = self.o_dense(concat_attention)
        return output

    def split_heads(self, inputs, batch_size):
        """
        Args:
            inputs: (batch_size, len, d_model)
            batch_size:
        Returns:
            (batch_size, num_heads, len, d_model/num_heads)
        """
        inputs = tf.reshape(inputs, (batch_size, -1, self.num_heads, self.d_model//self.num_heads))
        return tf.transpose(inputs, [0, 2, 1, 3])
    
    def scaled_dot_product_attention(self, q, k, v, mask):
        """
        Args:
            q: query tensor, [batch_size, seq_len, d_model]
            k: key tensor, [batch_size, seq_len, d_model]
            v: value tensor, [batch_size, seq_len, d_model]
        Returns:
            outputs: tensor, [batch_size, seq_len, d_model]
        """
        d_k = self.d_model // self.num_heads #dimension of each head
        scores = tf.matmul(q, k, transpose_b=True) / tf.sqrt(d_k)
        if mask:
            scores += (mask * -1e30)
        
        weights = tf.nn.softmax(scores, axis=-1)
        outputs = tf.matmul(weights, v)
        return outputs

    