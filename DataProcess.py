import numpy as np
import tiktoken
import tensorflow as tf


class DataProcess:

    special_characters = {'<|endoftext|>'}

    def __init__(self, input_text, stride=384, context_length=1024, batch_size=8):
        self.tokenizer = tiktoken.get_encoding('gpt2')
        self.tokens = None

        self.stride = stride
        self.context_length = context_length
        self.batch_size = batch_size

        self.dataset = None

        self.preprocess(input_text)

    def tokenize(self, input_text):
        self.tokens = self.tokenizer.encode(input_text, allowed_special=self.special_characters)

    def create_dataset(self):

        input_data = []
        out_data = []

        for i in range(0, len(self.tokens) - self.context_length, self.stride):
            input_data.append(tf.constant(self.tokens[i: i + self.context_length], dtype=tf.int64))
            out_data.append(tf.constant(self.tokens[i + 1: i + self.context_length + 1], dtype=tf.int64))

        input_data = np.array(input_data)
        out_data = np.array(out_data)

        self.dataset = tf.data.Dataset.from_tensor_slices((input_data, out_data))
        self.dataset.batch(batch_size=self.batch_size)
        self.dataset = self.dataset.shuffle(buffer_size=1000).batch(self.batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    def preprocess(self, input_text):
        self.tokenize(input_text)
        self.create_dataset()

