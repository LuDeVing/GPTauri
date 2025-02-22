import os.path

import tensorflow as tf

import MultiHeadAttention
from CustomCrossEntropy import CustomCrossEntropy
from GPTauri import GPTauri
from DataProcess import DataProcess

tf.keras.backend.clear_session()
tf.compat.v1.reset_default_graph()

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


a = open("input_text.txt", "r", encoding='utf-8').read()
model = GPTauri()

batch_size = 4

data = DataProcess(
    input_text=a,
    stride=model.CONFIGURATION[model.context_length],
    context_length=model.CONFIGURATION[model.context_length],
    batch_size=batch_size
)


def generate_text(input_batch, num_of_additions, temperature=1.5, top_k=30):
    input_batch = tf.cast(input_batch, tf.int64)
    for _ in range(num_of_additions):
        context_len_input_batches = input_batch[:, -model.CONFIGURATION[model.context_length]:]

        output = model.call(context_len_input_batches)

        logits = output[:, -1, :]
        logits = logits / temperature

        if top_k is not None:
            values, _ = tf.math.top_k(logits, k=top_k)
            min_values = tf.reduce_min(values)
            logits = tf.where(logits < min_values, -float("inf"), logits)

        probabilities = tf.keras.activations.softmax(logits, axis=-1)

        next_word = tf.random.categorical(probabilities, num_samples=1)
        next_word = tf.cast(next_word, tf.int64)

        input_batch = tf.concat([input_batch, next_word], axis=1)

    return input_batch


def generate(input_text, num_of_additions=25, temperature=1.5, top_k=30):
    encoded_text = data.tokenize(input_text)
    encoded_text = tf.constant([encoded_text])
    out_tokens = generate_text(encoded_text, num_of_additions, temperature, top_k)[0]
    return data.tokenizer.decode(out_tokens)


model.build(input_shape=(batch_size, model.CONFIGURATION[model.context_length]))
model.summary()

model.train(data)

tf.keras.backend.clear_session()
tf.compat.v1.reset_default_graph()

loaded_model = tf.keras.models.load_model(
    "model_data/model_weights",
    custom_objects={
        "MultiHeadAttention": MultiHeadAttention,
        "CustomCrossEntropy": CustomCrossEntropy
    }
)

print(generate("The desultory life of the Riviera lends"))

for batch in data.dataset:
    inputs, outputs = batch
    print(data.tokenizer.decode(generate_text(inputs, 5)[0][-10:]))
    print('---')
    print('---')
