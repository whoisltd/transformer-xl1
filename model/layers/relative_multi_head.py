import tensorflow as tf
from tensorflow.python.keras.layers import Dropout, Dense
from rel_shift import *
from generate_mask import *
import numpy as np
INITIALIZER = tf.keras.initializers.RandomNormal(stddev=0.01)

class RelativeMultiHeadAttention(tf.keras.layers.Layer):
    """
    Transformer XL Relative Multi-Head Attention Layer
    """

    def __init__(self, d_model = 512, num_heads = 6, dropout_rate=0.1):
        super(RelativeMultiHeadAttention, self).__init__()
        self.d_model = d_model #number of features per head
        self.num_heads = num_heads #number of heads
        self.d_k = d_model // num_heads
        self.dropout_rate = dropout_rate

        self.w = Dense(3*d_model, use_bias=False, kernel_initializer=INITIALIZER)
        self.w_r = Dense(d_model, use_bias=False, kernel_initializer=INITIALIZER)
        self.w_o = Dense(d_model, use_bias=False, kernel_initializer=INITIALIZER)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)

        self.u = self.add_weight(
            name="global_content_bias",
            shape=[num_heads,self.d_k],
            initializer="glorot_normal"
        )

        self.v = self.add_weight(
            name="global_position_bias",
            shape=[num_heads,self.d_k],
            initializer="glorot_normal"
        )

    def call(self, inputs, inputs_mem, r, training):
        # q, k, v, r = inputs
        print("oke arre okkee")
        print(inputs.shape)
        batch_size = tf.shape(inputs)[0]
        q_len = tf.shape(inputs)[1]

        if inputs_mem is None:
            cat = inputs
        else:
            print(inputs_mem.shape)
            cat = tf.concat((inputs_mem, inputs), axis=1)

        cat = self.dropout1(cat, training=training)
        k_len = tf.shape(cat)[1]
        # r_len = tf.shape(r)[1]
        r = r[:, -k_len:]
        r =self.dropout2(r, training=training)
        w = self.w(cat)
        r = self.w_r(r)
        # w_r = tf.reshape(self.w_r(pos_emb), (k_len, self.num_heads, self.d_model//self.num_heads))
        w_heads = tf.reshape(w, (
        batch_size, k_len, 3 * self.num_heads, self.d_k))
        heads_q, heads_k, heads_v = tf.split(w_heads, 3, axis=2)
        # w_q, w_v, w_k = tf.split(w, 3, axis=2)
        heads_q = heads_q[:, -q_len:]
        # heads_q = self.split_heads(w_q[:,-q_len:])
        # # heads_q = heads_q[:,-q_len:]
        # heads_k = self.split_heads(w_k)
        # heads_v = self.split_heads(w_v)
        
        heads_r = tf.reshape(r, (k_len, self.num_heads, self.d_k))

        #attention

        scaled_attention = self.scaled_dot_product_attention(heads_q, heads_k, heads_v, heads_r, q_len, k_len)
        # scaled_attention = tf.transpose(scaled_attention, [0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, q_len, self.d_model))
        output = self.w_o(concat_attention)

        return output
    
    def scaled_dot_product_attention(self, heads_q, heads_k, heads_v, r, q_len, k_len):
        """
        Args:
            q: query tensor, [batch_size, len, d_model]
            k: key tensor, [batch_size, len, d_model]
            v: value tensor, [batch_size, len, d_model]
            r: relative position tensor, [batch_size, len, d_model]
        Returns:
            outputs: tensor, [batch_size, len, d_model]
        """
        print(heads_q.shape, heads_k.shape, heads_v.shape, r.shape)
        ac = tf.einsum('bqnd,bknd->bnqk', heads_q + self.u, heads_k)
        # ac = tf.matmul(q + self.u, k, transpose_b=True)
        # bd = tf.matmul(q + self.v, r, transpose_b=True)
        bd = tf.einsum('bqnd,knd->bnqk', heads_q + self.v, r) #(batch, batch, q_len,q_len)
        bd = RelativeShift(bd)
        print(ac.shape, bd.shape)
        score = (ac + bd) / (self.d_k ** 0.5)
        # scale = tf.cast(tf.shape(k)[-1], tf.float32)
        # scores = tf.matmul(q, k, transpose_b=True) / tf.sqrt(d_k)
        mask = CreateMask(q_len, k_len-q_len)
        score = score * mask - 1e30 * (1. - mask)
        
        weights = tf.nn.softmax(score, axis=-1)
        # outputs = tf.matmul(weights, v)
        outputs = tf.einsum('bnqk,bknd->bqnd', weights, heads_v)

        
        return outputs  	