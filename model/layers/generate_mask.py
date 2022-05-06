import tensorflow as tf

def CreateMask(q_len, m_len):
        mask = tf.sequence_mask(tf.range(1, q_len + 1), q_len, dtype=tf.float32)
        mask = tf.pad(mask, [[0, 0], [m_len, 0]], constant_values=1)
        return mask