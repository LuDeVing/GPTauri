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


def generate_text(input_batch, num_of_additions, temperature=1.4, top_k=25):
    input_batch = tf.cast(input_batch, tf.int64)
    for _ in range(num_of_additions):
        context_len_input_batches = input_batch[:, -model.CONFIGURATION[model.context_length]:]

        output = model(context_len_input_batches)

        logits = output[:, -1, :]
        logits = logits / temperature

        if top_k is not None:
            values, _ = tf.math.top_k(logits, k=top_k)
            min_values = tf.reduce_min(values, axis=-1, keepdims=True)
            logits = tf.where(logits < min_values, -float("inf"), logits)

        next_word = tf.random.categorical(logits, num_samples=1)
        next_word = tf.cast(next_word, tf.int64)

        input_batch = tf.concat([input_batch, next_word], axis=1)


    return input_batch

def generate_and_print(input_text, num_of_additions, temperature=0.7, top_k=100):

    encoded_text = data.tokenize(input_text)
    input_batch = tf.constant([encoded_text])

    print(input_text, end="")

    input_batch = tf.cast(input_batch, tf.int64)
    for _ in range(num_of_additions):
        context_len_input_batches = input_batch[:, -model.CONFIGURATION[model.context_length]:]


        output = model(context_len_input_batches)

        logits = output[:, -1, :]
        logits = logits / temperature

        if top_k is not None:
            values, _ = tf.math.top_k(logits, k=top_k)
            min_values = tf.reduce_min(values, axis=-1, keepdims=True)
            logits = tf.where(logits < min_values, -float("inf"), logits)

        next_word = tf.random.categorical(logits, num_samples=1)
        next_word = tf.cast(next_word, tf.int64)

        print(data.tokenizer.decode(next_word[0]), end="")

        input_batch = tf.concat([input_batch, next_word], axis=1)

    print()
    print()

    return input_batch

def generate(input_text, num_of_additions=25, temperature=1.4, top_k=25):
    encoded_text = data.tokenize(input_text)
    encoded_text = tf.constant([encoded_text])
    out_tokens = generate_text(encoded_text, num_of_additions, temperature, top_k)[0]
    return data.tokenizer.decode(out_tokens)


model.build(input_shape=(batch_size, model.CONFIGURATION[model.context_length]))
# random_tensor = tf.random.uniform((batch_size, model.CONFIGURATION[model.context_length]),
#                                  minval=0, maxval=50257, dtype=tf.int32)
# model(random_tensor)

model.summary()

# model.train_model(data)
# model.load_weights(model.WEIGHTS_PATH)

model.load_weights("model_data\\gpt2_pretrained_weights\\gpt2_model_weights.ckpt")

while True:
    txt = input()
    if txt == "":
        continue
    if txt == "exit":
        break
    generate_and_print(txt, 100, 1.2, 10)

# for batch in data.dataset:
#     inputs, outputs = batch
#     print(data.tokenizer.decode(generate_text(inputs, 5)[0][-10:]))
#     print('---')
#     print('---')
