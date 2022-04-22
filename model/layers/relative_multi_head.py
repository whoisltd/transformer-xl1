import tensorflow as tf
from tensorflow.python.keras.layers import Embedding, Dropout
import numpy as np

class RelativeMultiHeadAttention(tf.keras.layers.Layer):
    """
    Transformer XL Relative Multi-Head Attention Layer
    """

    def __init__(self, hidden_size, num_heads, dropout_rate=0.1, **kwargs):
        super(RelativeMultiHeadAttention, self).__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.q_dense = tf.keras.layers.Dense(self.hidden_size, use_bias=False, name='q')
        self.k_dense = tf.keras.layers.Dense(self.hidden_size, use_bias=False, name='k')
        self.v_dense = tf.keras.layers.Dense(self.hidden_size, use_bias=False, name='v')
        self.o_dense = tf.keras.layers.Dense(self.hidden_size, use_bias=False, name='o')
        self.dropout = Dropout(self.dropout_rate)

    def call(self, inputs, **kwargs):
        q, k, v, r = inputs
        batch_size = tf.shape(q)[0]
        q = self.q_dense(q)
        k = self.k_dense(k)
        v = self.v_dense(v)
        r = self.o_dense(r)
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        r = self.split_heads(r, batch_size)
        scaled_attention = self.scaled_dot_product_attention(q, k, v, r)
        scaled_attention = tf.transpose(scaled_attention, [0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.hidden_size))
        output = self.o_dense(concat_attention)
        return output

    def split_heads(self, inputs, batch_size):
        inputs = tf.reshape(inputs, (batch_size, -1, self.num_heads, self.hidden_size//self.num_heads))
        return tf.transpose(inputs, [0, 2, 1, 3])
    
    def scaled_dot_product_attention(self, q, k, v, r):
        """
        Multi-Head Attention
        """
        outputs = tf.matmul(q, k, transpose_b=True)
        outputs /= np.sqrt(self.hidden_size//self.num_heads)
        outputs += r
        outputs = tf.nn.softmax(outputs)
        outputs = tf.matmul(outputs, v)
        return outputs
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'hidden_size': self.hidden_size,
            'num_heads': self.num_heads,
            'dropout_rate': self.dropout_rate
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    