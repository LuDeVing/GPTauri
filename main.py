import tensorflow as tf

from GPTauri import GPTauri
from DataProcess import DataProcess

from ModelCrossEntropyLoss import CrossEntropyGPT
from MultiHeadAttention import MultiHeadAttention
from TransformerBlock import TransformerBlock

a = open("input_text.txt", "r", encoding='utf-8').read()

model = GPTauri()

data = DataProcess(
    input_text=a,
    stride=model.CONFIGURATION[model.context_length],
    context_length=model.CONFIGURATION[model.context_length],
    batch_size=8
)


def calculate_loss(input_batch, output_batch):
    context_len_input_batches = input_batch[:, -model.CONFIGURATION[model.context_length]:]
    context_len_output_batches = output_batch[:, -model.CONFIGURATION[model.context_length]:]

    context_len_output_batches = tf.reshape(context_len_output_batches, [-1])
    one_hot_output = tf.one_hot(context_len_output_batches, model.CONFIGURATION[model.vocabulary_size])

    logits = model(context_len_input_batches)
    logits = tf.reshape(logits, [-1, model.CONFIGURATION[model.vocabulary_size]])

    loss_func = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    return loss_func(logits, one_hot_output)


def generate_text(input_batch, num_of_additions):
    for _ in range(num_of_additions):
        context_len_input_batches = input_batch[:, -model.CONFIGURATION[model.context_length]:]

        output = model(context_len_input_batches)

        logits = output[:, -1, :]

        probabilities = tf.keras.activations.softmax(logits, axis=-1)

        next_word = tf.argmax(probabilities, axis=-1)
        next_word = tf.expand_dims(next_word, axis=-1)

        input_batch = tf.concat([input_batch, next_word], axis=1)

    return input_batch


def train_model():

    inputs, outputs = 1, 1

    for batch in data.dataset:
        inputs, outputs = batch

    model(inputs)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss=CrossEntropyGPT(model),
                  metrics=['accuracy'],
            )

    model.summary()

    model.fit(data.dataset, epochs=1)


train_model()

loss = 0

print(len(data.dataset))
for batch in data.dataset:
    inputs, outputs = batch
    loss += calculate_loss(inputs, outputs)
    print(loss)

print(loss)
