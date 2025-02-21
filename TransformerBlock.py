import tensorflow as tf

from MultiHeadAttention import MultiHeadAttention


class TransformerBlock(tf.keras.layers.Layer):

    def __init__(self, conf, **kwargs):

        super(TransformerBlock, self).__init__(**kwargs)

        self.norm_layer_1 = tf.keras.layers.LayerNormalization(
            beta_initializer="zeros",
            gamma_initializer="ones"
        )

        self.multi_head_attention = MultiHeadAttention(
            conf['embedding_dimension'],
            conf['num_heads'],
            conf['context_length'],
            conf['drop_out_rate'],
            conf['qkv_bias']
        )

        self.drop_out = tf.keras.layers.Dropout(conf["drop_out_rate"])

        self.norm_layer_2 = tf.keras.layers.LayerNormalization(
            beta_initializer="zeros",
            gamma_initializer="ones"
        )

        self.linear_layer_1 = tf.keras.layers.Dense(conf['embedding_dimension'] * 4)
        self.gelu_activation = tf.keras.activations.gelu
        self.linear_layer_2 = tf.keras.layers.Dense(conf['embedding_dimension'])

    def call(self, input_data, training=None):

        shortcut = input_data

        x = self.norm_layer_1(shortcut, training=training)
        x = self.multi_head_attention(x, training=training)
        x = self.drop_out(x, training=training)

        x = x + shortcut
        shortcut = x

        x = self.norm_layer_2(x, training=training)
        x = self.linear_layer_1(x)
        x = self.gelu_activation(x)
        x = self.linear_layer_2(x)
        x = self.drop_out(x, training=training)

        x = x + shortcut

        return x
