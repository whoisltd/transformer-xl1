import tensorflow as tf
from tensorflow.python.keras.layers import Dropout, Dense
from rel_shift import *
from generate_mask import *
import numpy as np

class RelativeMultiHeadAttention(tf.keras.layers.Layer):
    """
    Transformer XL Relative Multi-Head Attention Layer
    """

    def __init__(self, d_model = 512, num_heads = 6, dropout_rate=0.1):
        super(RelativeMultiHeadAttention, self).__init__()
        self.d_model = d_model #hidden size
        self.num_heads = num_heads #number of heads of multihead
        self.d_k = d_model // num_heads
        self.dropout_rate = dropout_rate

        self.w = Dense(3*d_model) #query, key, value
        self.w_r = Dense(d_model) # position
        self.w_o = Dense(d_model) #output

        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)

        # content_attention_bias [numheads, d_k]
        self.u = self.add_weight(
            name="global_content_bias",
            shape=[num_heads,self.d_k],
            initializer="glorot_normal"
        )
        # positional_attention_bias [numheads, d_K]
        self.v = self.add_weight(
            name="global_position_bias",
            shape=[num_heads,self.d_k],
            initializer="glorot_normal"
        )

    def call(self, inputs, inputs_mem, r, training):

        """
        inputs: shape=(batch_size, q_len, d_model)
        r:(pos_embedding) shape=(1, k_len, d_model) 
        inputs_mem: shape=(batch_size, m_len, d_model)
        """
        #q_len=query_len, k_len=key_len, k_len = q_len + m_len
        #r = position_embedding
        batch_size = tf.shape(inputs)[0]
        q_len = tf.shape(inputs)[1]

        #concat input vs memory
        if inputs_mem is not None:
            print(inputs_mem.shape)
            inputs = tf.concat((inputs_mem, inputs), axis=1)

        inputs = self.dropout1(inputs, training=training)

        k_len = tf.shape(inputs)[1]

        r = r[:, -k_len:]
        r =self.dropout2(r, training=training)
        r = self.w_r(r)

        w = self.w(inputs)
        #split head
        w_heads = tf.reshape(w, (batch_size, k_len, 3 * self.num_heads, self.d_k))
        heads_r = tf.reshape(r, (k_len, self.num_heads, self.d_k))

        heads_q, heads_k, heads_v = tf.split(w_heads, 3, axis=2)
        heads_q = heads_q[:, -q_len:]
        #attention

        scaled_attention = self.scaled_dot_product_attention(heads_q, heads_k, heads_v, heads_r, q_len, k_len)
        concat_attention = tf.reshape(scaled_attention, (batch_size, q_len, self.d_model))

        output = self.w_o(concat_attention)

        return output
    
    def scaled_dot_product_attention(self, heads_q, heads_k, heads_v, r, q_len, k_len):

        ac = tf.einsum('bqnd,bknd->bnqk', heads_q + self.u, heads_k)
        
        bd = tf.einsum('bqnd,knd->bnqk', heads_q + self.v, r)
        bd = relative_shift(bd)

        score = (ac + bd) / (tf.sqrt(self.d_k))
        # scores = tf.matmul(q, k, transpose_b=True) / tf.sqrt(d_k)

        attn_mask = create_mask(q_len, k_len-q_len) # attn_mask: shape=(m_len + q_len, q_len)
        score = score * attn_mask - 1e30 * (1. - attn_mask)
        
        weights = tf.nn.softmax(score, axis=-1)
        # outputs = tf.matmul(weights, v)
        outputs = tf.einsum('bnqk,bknd->bqnd', weights, heads_v)
        
        return outputs  	