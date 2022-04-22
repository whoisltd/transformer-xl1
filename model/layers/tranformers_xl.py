import tensorflow as tf
from tensorflow.python.keras.layers import Embedding, Dropout
import numpy as np
from positional_embedding import PositionEmbedding
from relative_multi_head import RelativeMultiHeadAttention

class TransformerXL(tf.keras.Model):
    """
    Transformer XL model
    """

    def __init__(self, max_seq_length, hidden_size, num_attention_heads, intermediate_size, dropout_rate=0.1, **kwargs):
        super(TransformerXL, self).__init__(**kwargs)
        self.max_seq_length = max_seq_length
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.dropout_rate = dropout_rate
        self.position_embedding = PositionEmbedding(max_seq_length, hidden_size, dropout_rate)
        self.relative_multi_head_attention = RelativeMultiHeadAttention(hidden_size, num_attention_heads, dropout_rate)
        self.intermediate_dense = tf.keras.layers.Dense(intermediate_size, activation='relu')
        self.output_dense = tf.keras.layers.Dense(hidden_size)

    def call(self, inputs, **kwargs):
        seq_length = tf.shape(inputs)[1]
        position_embedding = self.position_embedding(inputs)
        relative_multi_head_attention = self.relative_multi_head_attention(position_embedding, position_embedding, position_embedding)
        intermediate_output = self.intermediate_dense(relative_multi_head_attention)
        layer_output = self.output_dense(intermediate_output)
        return layer_output

    def get_config(self):
        config = super(TransformerXL, self).get_config().copy()
        config.update({
            'max_seq_length': self.max_seq_length,
            'hidden_size': self.hidden_size,
            'num_attention_heads': self.num_attention_heads,
            'intermediate_size': self.intermediate_size,
            'dropout_rate': self.dropout_rate
        })
        return config


