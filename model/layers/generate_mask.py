import tensorflow as tf

class CreateMask():
    """
    Creates a mask to be used in attention layer.
    """
    
    def __init__(self):
        super(CreateMask, self).__init__()
    
    def __call__(self, q_len, m_len):
        mask = tf.sequence_mask(tf.range(1, q_len + 1), q_len, dtype=tf.float32)
        mask = tf.pad(mask, [[0, 0], [m_len, 0]], constant_values=1)
        return mask