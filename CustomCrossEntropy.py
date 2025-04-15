import tensorflow as tf


class CustomCrossEntropy(tf.keras.losses.Loss):
    def __init__(self, reduction=tf.keras.losses.Reduction.AUTO, name="custom_crossentropy"):
        super().__init__()
        self.name = name
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True,
            reduction=reduction
        )

    def call(self, y_true, y_pred):

        y_true_flat = tf.reshape(y_true, [-1])
        y_pred_flat = tf.reshape(y_pred, [-1, tf.shape(y_pred)[-1]])

        loss = self.loss_fn(y_true_flat, y_pred_flat)

        return loss

    def get_config(self):
        config = super().get_config()  # Get base config
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)