import tensorflow as tf

def ffn(d_ff, d_model, activation='relu'):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(d_ff, activation=activation),
        tf.keras.layers.Dense(d_model)
    ])