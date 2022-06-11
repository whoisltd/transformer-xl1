import tensorflow as tf
from tensorflow.keras.layers import Sequential, Dense, Dropout

def ffn(d_ff, d_model, activation='relu'):
    return Sequential([
        Dense(d_ff, activation=activation),
        Dropout(0.1),
        Dense(d_model),
        Dropout(0.1)
    ])