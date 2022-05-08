import tensorflow as tf

def position_embedding(d_model, k_len):
    # 
    inv_freq = 1 / (10000 ** (tf.range(0, d_model, 2.0) / d_model))
    pos_seq = tf.range(k_len - 1, -1, -1.0)
    positions = tf.tensordot(pos_seq, inv_freq, axes=0)
    sinusoid_inp = tf.concat((tf.sin(positions), tf.cos(positions)), -1)
    pos_emb = tf.cast(sinusoid_inp, tf.float32)

    return pos_emb[None, :, :]
        
# if __name__ == '__main__':
#     d_model = 128
#     k_len = 32+16
#     pos_emb = PositionEmbedding()
#     pos_emb = pos_emb(d_model, k_len)