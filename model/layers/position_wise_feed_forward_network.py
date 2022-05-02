import tensorflow as tf

def ffn(d_ff=2048, d_model=512, activation='relu'):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(d_ff, activation=activation),
        tf.keras.layers.Dense(d_model)
    ])