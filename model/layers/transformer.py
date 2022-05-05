import tensorflow as tf
from relative_multi_head import *
from position_wise_feed_forward_network import *

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
        # 8.82292632e-04  5.70151024e-05  3.64560820e-02  2.05970183e-02]