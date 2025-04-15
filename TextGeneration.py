import tensorflow as tf

class TextGeneration:

    @staticmethod
    def generate_and_print(input_text, model, tokenizer, num_of_additions, temperature=0.7, top_k=100):

        encoded_text = tokenizer.tokenize(input_text)
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

            print(tokenizer.decode(next_word[0]), end="")

            input_batch = tf.concat([input_batch, next_word], axis=1)

        print()

        return input_batch

    @staticmethod
    def generate_text(input_batch, model, num_of_additions, temperature=1.4, top_k=25):
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

    @staticmethod
    def generate(input_text, model, tokenizer, num_of_additions=25, temperature=1.4, top_k=25):
        encoded_text = tokenizer.tokenize(input_text)
        encoded_text = tf.constant([encoded_text])
        out_tokens = TextGeneration.generate_text(encoded_text, model, num_of_additions, temperature, top_k)[0]
        return tokenizer.decode(out_tokens)