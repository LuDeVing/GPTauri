import tensorflow as tf
import numpy as np


class MultiHeadAttention(tf.keras.layers.Layer):

    def __init__(self, qkv_dim, num_heads, context_length, dropout_rate, qkv_bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)

        if qkv_dim % num_heads != 0:
            raise ArithmeticError("Dimensions do not align, each head needs equal amount of dimensions")

        self.qkv_dim = qkv_dim
        self.num_heads = num_heads
        self.head_dim = qkv_dim // num_heads

        self.query_weights = tf.keras.layers.Dense(qkv_dim, use_bias=qkv_bias)
        self.key_weights = tf.keras.layers.Dense(qkv_dim, use_bias=qkv_bias)
        self.value_weights = tf.keras.layers.Dense(qkv_dim, use_bias=qkv_bias)

        self.dropout = tf.keras.layers.Dropout(dropout_rate)

        self.out_proj = tf.keras.layers.Dense(self.qkv_dim)

        mask = tf.linalg.band_part(tf.ones((context_length, context_length)), 0, -1) - tf.eye(context_length)
        self.mask = tf.cast(mask, tf.bool)

    def call(self, input_data, training=None, **kwargs):
        batch = tf.shape(input_data)[0]
        context_len = tf.shape(input_data)[1]

        query = self.query_weights(input_data)
        key = self.key_weights(input_data)
        value = self.value_weights(input_data)

        query = tf.reshape(query, [batch, context_len, self.num_heads, self.head_dim])
        key = tf.reshape(key, [batch, context_len, self.num_heads, self.head_dim])
        value = tf.reshape(value, [batch, context_len, self.num_heads, self.head_dim])

        query = tf.transpose(query, perm=[0, 2, 1, 3])
        key = tf.transpose(key, perm=[0, 2, 1, 3])
        value = tf.transpose(value, perm=[0, 2, 1, 3])

        attention_scores = tf.matmul(query, tf.transpose(key, [0, 1, 3, 2]))

        mask = self.mask[:context_len, :context_len]
        mask = tf.expand_dims(tf.expand_dims(mask, 0), 0)

        attention_scores = tf.where(mask, -float('inf'), attention_scores)

        attention_weights = tf.nn.softmax(attention_scores / (self.head_dim ** 0.5), axis=-1)
        attention_weights = self.dropout(attention_weights, training=training)

        context_vecs = tf.matmul(attention_weights, value)
        context_vecs = tf.transpose(context_vecs, perm=[0, 2, 1, 3])

        context_vecs = tf.reshape(context_vecs, [batch, context_len, self.qkv_dim])
        context_vecs = self.out_proj(context_vecs)

        return context_vecs
