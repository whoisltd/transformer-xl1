import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LayerNormalization
from tensorflow.keras.layers import Embedding
from model.layers.positional_embedding import *
from model.layers.transformer import *

INITIALIZER = tf.keras.initializers.RandomNormal(stddev=0.01)

class TransformerXL(tf.keras.Model):
    """
    Transformer XL model
    """

    def __init__(self, n_vocab, d_embed, d_model, 
                d_ff, q_len, m_len, num_heads,
                n_layer, dropout_rate):
        super(TransformerXL, self).__init__()
        self.d_embed = d_embed
        self.d_model = d_model

        self.q_len = q_len
        self.m_len = m_len
        self.n_layer = n_layer
        #word embedding
        self.embedding = Embedding(n_vocab, d_model)
        # self.embedding = tf.Variable(INITIALIZER((n_vocab, d_embed)), name='embedding')
        # # word embedding size to model size
        # self.projection = tf.Variable(INITIALIZER((d_embed, d_model)), name='projection')
        
        self.dropout1 = Dropout(dropout_rate)

        #positional embedding
        k_len = q_len + m_len
        self.pos_embedding = position_embedding(d_model, k_len)
        # self.logit_bias = tf.Variable(tf.zeros((n_vocab,)), name='logit_bias')

        #transformer
        self.multihead_layers = [Transformer(d_model, d_ff, num_heads, dropout_rate) for _ in range(n_layer)]

    def cache_mems(self, current, previous):
        if self.m_len is None or self.m_len <= 0:
            return None
        if previous is None:
            new_mem = current
        else:
            new_mem = tf.concat([previous, current], axis=1)
        return tf.stop_gradient(new_mem[:, -self.m_len:])
    
    def call(self, inputs, inputs_mem=None, training = False):
        new_mems = []
        x = self.embedding(inputs)

        if inputs_mem is None:
            inputs_mem = [None] * self.n_layer

        for i in range(self.n_layer):
            new_mems.append(self.cache_mems(x, inputs_mem[i]))
            x = self.multihead_layers[i](inputs=x, 
                                        inputs_mem=inputs_mem[i], 
                                        r=self.pos_embedding,
                                        training=training)

        x=self.dropout1(x, training=training)
        print(x.shape)
        # x = tf.keras.layers.GlobalAveragePooling1D()(x)
        # x = tf.keras.layers.Dropout(0.1)(x, training=True)
        # x = tf.keras.layers.Dense(20, activation="relu")(x)
        # x = tf.keras.layers.Dropout(0.1)(x, training=True)
        logits = tf.keras.layers.Dense(2)(x)

        return logits, new_mems

if __name__ == '__main__':
    n_vocab = 1000
    d_embed = 128
    d_model = 128
    d_ff = 512
    q_len = 16
    m_len = 32
    num_heads = 8
    n_layer = 6
    dropout_rate = 0.1
    batch_size = 8
    mem_transformer = TransformerXL(n_vocab=n_vocab,
                                    d_embed=d_embed,
                                    d_model=d_model,
                                    d_ff=d_ff,
                                    q_len=q_len,
                                    m_len=m_len,
                                    num_heads=num_heads,
                                    n_layer=n_layer,
                                    dropout_rate=dropout_rate)
    inputs = tf.reshape(tf.range(batch_size * q_len), shape=(batch_size, q_len))
    # print(inputs)
    output1, mems1 = mem_transformer(inputs, training=False)
    # mem_transformer. = mems1
    output2, mems2 = mem_transformer(inputs, training=False)