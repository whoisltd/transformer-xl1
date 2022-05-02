import tensorflow as tf

class RelativeShift():
    """
    Performs relative shift to form the relative attention score.
    """
    
    def __init__(self):
        super(RelativeShift, self).__init__()

    def call(self, x):
        zero_pad = tf.zeros([x.shape[0], 1], *x.shape[2:])
        x_padded = tf.concat([zero_pad, x], axis=1)
        x_padded = tf.reshape(x_padded, [x.shape[1] + 1, x.shape[0]], *x.shape[2:])
        x = tf.reshape(x_padded[1:], tf.shape(x))
        return x