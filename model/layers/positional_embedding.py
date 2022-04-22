import tensorflow as tf
from tensorflow.python.keras.layers import Embedding, Dropout
import numpy as np

class PositionEmbedding(tf.keras.layers.Layer):
    """
    Transformer XL Position Embedding Layer
    """

    def __init__(self, max_seq_length, hidden_size, dropout_rate=0.1, **kwargs):
        super(PositionEmbedding, self).__init__(**kwargs)
        self.max_seq_length = max_seq_length
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.position_embedding = Embedding(
            self.max_seq_length,
            self.hidden_size,
            embeddings_initializer=tf.keras.initializers.Constant(self.get_position_encoding_matrix()),
            trainable=False
        )
        self.dropout = Dropout(self.dropout_rate)

    def call(self, inputs, **kwargs):
        seq_length = tf.shape(inputs)[1]
        position_embedding = self.position_embedding(tf.range(seq_length))
        position_embedding = tf.reshape(position_embedding, [seq_length, self.hidden_size])
        position_embedding = self.dropout(position_embedding)
        return position_embedding

    def get_position_encoding_matrix(self):
        """
        Generate the position encoding as a numpy array
        """
        position_encoding = np.array([
            [pos / np.power(10000, 2 * (j // 2) / self.hidden_size) for j in range(self.hidden_size)]
            if pos != 0 else np.zeros(self.hidden_size) for pos in range(self.max_seq_length)
        ])
        position_encoding[1:, 0::2] = np.sin(position_encoding[1:, 0::2]) # dim 2i
        position_encoding[1:, 1::2] = np.cos(position_encoding[1:, 1::2]) # dim 2i+1
        return position_encoding