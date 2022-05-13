import tensorflow as tf
from model.layers.relative_multi_head import *
from model.layers.position_wise_feed_forward_network import *

class Transformer(tf.keras.layers.Layer):
    def __init__(self, d_model, d_ff, num_heads, dropout_rate):
        super(Transformer, self).__init__()

        self.rel_multi_head_attention = RelativeMultiHeadAttention(d_model, num_heads, dropout_rate)

        #feed forward network
        self.pos_ffn = ffn(d_ff = d_ff, d_model= d_model)

        #layer norm
        self.layer_norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        #dropout
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, inputs, inputs_mem, r, training):

        attn_out = self.rel_multi_head_attention(inputs, inputs_mem, r, training)
        attn_out = self.dropout1(attn_out, training=training)
        out1 = self.layer_norm1(inputs + attn_out)

        ffn_out = self.pos_ffn(out1, training = training) 
        ffn_out = self.dropout2(ffn_out, training=training)

        out2 = self.layer_norm2(out1 + ffn_out)

        return out2

# if __name__ == '__main__':
#     from positional_embedding import *
#     a = Transformer(128, 512, 128, 0.1)
#     inputs = tf.reshape(tf.range(8 * 16), shape=(8, 16))
#     pos_embedding = PositionEmbedding(128, 16+32)
#     b = a(inputs = inputs, inputs_mem=None, r=pos_embedding)
#     print(b)