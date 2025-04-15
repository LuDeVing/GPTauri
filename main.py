import tensorflow as tf

from GPTauri import GPTauri
from DataProcess import DataProcess
from TextGeneration import TextGeneration
from Tokenizer import Tokenizer

tf.keras.backend.clear_session()
tf.compat.v1.reset_default_graph()

tokenizer = Tokenizer()

def test_training():

    a = open("input_text.txt", "r", encoding='utf-8').read()

    conf = {
        "vocabulary_size": 50257,
        "embedding_dimension": 768,
        "num_heads": 12,
        "context_length": 256,
        "drop_out_rate": 0.1,
        "qkv_bias": True,
        "num_layers": 12
    }

    model = GPTauri(conf)

    batch_size = 1

    data = DataProcess(
        input_text=a,
        tokenizer=tokenizer,
        stride=model.CONFIGURATION[model.context_length],
        context_length=model.CONFIGURATION[model.context_length],
        batch_size=batch_size
    )

    model.train(data)
    model.load_weights(model.WEIGHTS_PATH)

    model.build(input_shape=(batch_size, model.CONFIGURATION[model.context_length]))
    random_tensor = tf.random.uniform((batch_size, model.CONFIGURATION[model.context_length]),
                                     minval=0, maxval=50257, dtype=tf.int32)
    model(random_tensor)

    model.summary()

    TextGeneration.generate_text("Let Me tell you about a world far, far away.", model, 10)


def generate_response():

    model = GPTauri()
    model.load_weights("model_data\\gpt2_pretrained_weights\\gpt2_model_weights.ckpt")

    model.build(input_shape=(1, model.CONFIGURATION[model.context_length]))
    random_tensor = tf.random.uniform((1, model.CONFIGURATION[model.context_length]),
                                      minval=0, maxval=50257, dtype=tf.int32)
    model(random_tensor)

    model.summary()

    while True:
        txt = input()
        if txt == "":
            continue
        if txt == "exit":
            break
        TextGeneration.generate_and_print(txt, model, tokenizer, 100, 1.2, 10)

generate_response()