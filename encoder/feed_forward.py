import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

class TransformerFeedForwardLayer(tf.keras.layers.Layer):
    def __init__(self, d_model=512, d_ff=2048, dropout_rate=0.1):
        super(TransformerFeedForwardLayer, self).__init__()
        self.dense_1 = layers.Dense(d_ff, activation='relu')
        self.dense_2 = layers.Dense(d_model)
        self.dropout = layers.Dropout(dropout_rate)
        self.layer_norm = layers.LayerNormalization(epsilon=1e-6)

    def call(self, x = None, training=True):
        # Feedforward network
        out_1 = self.dense_1(x)  # (batch_size, seq_len, d_ff)
        out_2 = self.dense_2(out_1)  # (batch_size, seq_len, d_model)
        # Apply dropout (only during training)
        out_2 = self.dropout(out_2, training=training)
        # Add skip connection (residual connection) and apply layer normalization
        out = self.layer_norm(x + out_2)
        return out
