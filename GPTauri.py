import tensorflow as tf
from TransformerBlock import TransformerBlock


class GPTauri(tf.keras.Model):
    CONFIGURATION = {
        "vocabulary_size": 50257,
        "embedding_dimension": 768,
        "num_heads": 12,
        "context_length": 128,  # 1024,
        "drop_out_rate": 0.1,
        "qkv_bias": False,
        "num_layers": 12
    }

    vocabulary_size = 'vocabulary_size'
    embedding_dimension = 'embedding_dimension'
    num_heads = 'num_heads'
    context_length = 'context_length'
    drop_out_rate = 'drop_out_rate'
    qkv_bias = 'qkv_bias'
    num_layers = 'num_layers'

    def __init__(self):
        super(GPTauri, self).__init__()

        self.token_embedding_layer = tf.keras.layers.Embedding(
            input_dim=self.CONFIGURATION[self.vocabulary_size],
            output_dim=self.CONFIGURATION[self.embedding_dimension]
        )

        self.position_embedding_layer = tf.keras.layers.Embedding(
            input_dim=self.CONFIGURATION[self.context_length],
            output_dim=self.CONFIGURATION[self.embedding_dimension]
        )

        self.drop_out_layer = tf.keras.layers.Dropout(self.CONFIGURATION[self.drop_out_rate])

        self.transformer_blocks = [TransformerBlock(self.CONFIGURATION)
                                   for _ in range(self.CONFIGURATION[self.num_layers])]

        self.normalization_layer = tf.keras.layers.LayerNormalization(
            beta_initializer="zeros",
            gamma_initializer="ones"
        )

        self.linear_output_layer = tf.keras.layers.Dense(self.CONFIGURATION[self.vocabulary_size])

    def call(self, input_data, training=None):
        batch_size, sentence_len = input_data.shape

        token_embeddings = self.token_embedding_layer(input_data)
        positional_embeddings = self.position_embedding_layer(tf.range(0, sentence_len))
        x = token_embeddings + positional_embeddings

        x = self.drop_out_layer(x, training=training)

        for idx in range(len(self.transformer_blocks)):
            x = self.transformer_blocks[idx](x, training=training)

        x = self.normalization_layer(x, training=training)
        logits = self.linear_output_layer(x)

        return logits
