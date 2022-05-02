from turtle import position
from grpc import Channel
import tensorflow as tf
import numpy as np

class PositionEmbedding(tf.keras.layers.Layer):
    """
    Transformer XL Position Embedding Layer
    """

    def __init__(self, d_model, pos_seq, clamp_len, **kwargs):
        super(PositionEmbedding, self).__init__(**kwargs)
        self.d_model = d_model
        self.pos_seq = pos_seq
        self.clamp_len = clamp_len
    def call(self, pos, **kwargs):
        """
        Args:
            d_model: dimension of embedding
            pos_seq: (batch_size, seq_len, d_model)
            clamp_len: max length of position sequence
        Returns:
            (batch_size, seq_len, d_model)
        """
        inv_freq = 1 / (10000 ** (tf.range(0, self.d_model, 2.0) / self.d_model))
        if self.clamp_len >0:
            self.pos_seq = tf.minimum(self.pos_seq, self.clamp_len)
        positions = tf.tensordot(pos, inv_freq, axes=0)
        pos_emb = tf.concat([tf.sin(pos_emb), tf.cos(pos_emb)], -1)
        pos_emb = tf.cast(pos_emb, tf.float32)
        return pos_emb
        
        
