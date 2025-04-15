import tensorflow as tf

from CustomCrossEntropy import CustomCrossEntropy
from TransformerBlock import TransformerBlock


class GPTauri(tf.keras.Model):
    CONFIGURATION = {
        "vocabulary_size": 50257,
        "embedding_dimension": 768,
        "num_heads": 12,
        "context_length": 1024,
        "drop_out_rate": 0.1,
        "qkv_bias": True,
        "num_layers": 12
    }

    vocabulary_size = 'vocabulary_size'
    embedding_dimension = 'embedding_dimension'
    num_heads = 'num_heads'
    context_length = 'context_length'
    drop_out_rate = 'drop_out_rate'
    qkv_bias = 'qkv_bias'
    num_layers = 'num_layers'

    WEIGHTS_PATH = 'model_data\\model_weights\\weights.ckpt'

    def __init__(self, configuration=None):

        if configuration is not None:
            self.CONFIGURATION = configuration

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
            gamma_initializer="ones",
            epsilon=1e-6
        )

        self.linear_output_layer = tf.keras.layers.Dense(self.CONFIGURATION[self.vocabulary_size], use_bias=False)

    def call(self, input_data, training=False, **kwargs):
        shape = tf.shape(input_data)
        batch_size, sentence_len = shape[0], shape[1]

        token_embeddings = self.token_embedding_layer(input_data)
        positional_embeddings = self.position_embedding_layer(tf.range(sentence_len))

        x = token_embeddings + positional_embeddings

        x = self.drop_out_layer(x, training=training)

        for idx in range(len(self.transformer_blocks)):
            x = self.transformer_blocks[idx](x, training=training)

        x = self.normalization_layer(x)
        logits = self.linear_output_layer(x)

        return logits

    def train(self, data):

        self.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                      loss=CustomCrossEntropy(),  # tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        self.fit(data.dataset, epochs=150)

        self.save_weights(self.WEIGHTS_PATH)
