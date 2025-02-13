import tensorflow as tf


class CrossEntropyGPT(tf.keras.losses.Loss):
    def __init__(self, model, name="CrossEntropyGPT"):
        super(CrossEntropyGPT, self).__init__(name=name)
        self.model = model

    def call(self, output_batch, logits):
        context_len_output_batches = output_batch[:, -self.model.CONFIGURATION[self.model.context_length]:]

        context_len_output_batches = tf.reshape(context_len_output_batches, [-1])
        context_len_output_batches = tf.cast(context_len_output_batches, tf.int32)

        one_hot_output = tf.one_hot(context_len_output_batches, self.model.CONFIGURATION[self.model.vocabulary_size])

        logits = tf.reshape(logits, [-1, self.model.CONFIGURATION[self.model.vocabulary_size]])

        loss_func = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

        return loss_func(logits, one_hot_output)
