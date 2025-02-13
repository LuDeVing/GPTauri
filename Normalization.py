import tensorflow as tf


class Normalization:

    def __init__(self, embedding_dimension):
        self.weights = tf.ones(embedding_dimension)
        self.bias = tf.zeros(embedding_dimension)
        self.eps = 1e-5

    def forward(self, input_data):
        mean = tf.reduce_mean(input_data, axis=-1, keepdims=True)
        var = tf.reduce_mean(input_data, axis=-1, keepdims=True)
        normalized_data = (input_data - mean) / tf.sqrt(var + self.eps)
        return self.weights * normalized_data + self.bias
