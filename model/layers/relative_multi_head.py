import tensorflow as tf
from tensorflow.python.keras.layers import Dropout, Dense
from rel_shift import *
from generate_mask import *
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

        self.w = Dense(3*d_model)
        self.w_r = Dense(d_model)
        self.w_o = Dense(d_model)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)

        self.u = self.add_weight(
            name="global_content_bias",
            shape=[1, num_heads,1, d_model//num_heads],
            initializer="glorot_normal"
        )

        self.v = self.add_weight(
            name="global_position_bias",
            shape=[1, num_heads,1, d_model//num_heads],
            initializer="glorot_normal"
        )

    def call(self, inputs, inputs_mem, r, training):
        # q, k, v, r = inputs
        print(r.shape)
        print(inputs.shape)
        batch_size = tf.shape(inputs)[0]
        q_len = tf.shape(inputs)[1]

        if inputs_mem is None:
            cat = inputs
        else:
            cat = tf.concat((inputs_mem, inputs), 1)


        cat = self.dropout1(cat, training=training)
        k_len = tf.shape(cat)[1]
        # r_len = tf.shape(r)[1]
        r = r[:, -k_len:]
        r =self.dropout2(r, training=training)
        w = self.w(cat)
        r = self.w_r(r)
        # w_r = tf.reshape(self.w_r(pos_emb), (k_len, self.num_heads, self.d_model//self.num_heads))

        # these transform the query, key, value vectors for multi-head attention
        w_q, w_v, w_k = tf.split(w, 3, axis=-1)
        
        heads_q = self.split_heads(w_q)
        heads_q = heads_q[:,-q_len:]
        heads_k = self.split_heads(w_k)
        heads_v = self.split_heads(w_v)
        heads_r = self.split_heads(r)

        #attention

        scaled_attention = self.scaled_dot_product_attention(heads_q, heads_k, heads_v, heads_r, q_len, k_len)
        scaled_attention = tf.transpose(scaled_attention, [0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, q_len, self.d_model))
        output = self.w_o(concat_attention)

        return output

    def split_heads(self, inputs):
        """
        Args:
            inputs: (batch_size, len, d_model)
        Returns:
            (batch_size, num_heads, len, d_model/num_heads)
        """
        batch_size = tf.shape(inputs)[0]
        len = tf.shape(inputs)[1] 
        d_model = tf.shape(inputs)[2] 
        hd_v = d_model // self.num_heads

        inputs = tf.reshape(inputs, (batch_size, len, -1, hd_v))

        return tf.transpose(inputs, [0,2,1,3])
    
    def scaled_dot_product_attention(self, q, k, v, r, q_len, k_len):
        """
        Args:
            q: query tensor, [batch_size, len, d_model]
            k: key tensor, [batch_size, len, d_model]
            v: value tensor, [batch_size, len, d_model]
            r: relative position tensor, [batch_size, len, d_model]
        Returns:
            outputs: tensor, [batch_size, len, d_model]
        """
        d_k = self.d_model // self.num_heads #dimension of each head
        # ac = tf.einsum('bqnd,bknd->bnqk', q + self.u, k)
        ac = tf.matmul(q + self.u, k, transpose_b=True)
        bd = tf.matmul(q + self.v, r, transpose_b=True)
        bd = RelativeShift()(bd)
        print(ac.shape, bd.shape)
        score = (ac + bd) / (d_k ** 0.5)
        scale = tf.cast(tf.shape(k)[-1], tf.float32)
        # scores = tf.matmul(q, k, transpose_b=True) / tf.sqrt(d_k)
        mask = CreateMask()(q_len, k_len-q_len)
        if mask is not None:
            score = score * mask - 1e30 * (1. - mask)
        
        weights = tf.nn.softmax(score, axis=-1)
        outputs = tf.matmul(weights, v)
        return outputs

    