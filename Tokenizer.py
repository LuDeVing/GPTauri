import tiktoken

class Tokenizer:

    special_characters = {'<|endoftext|>'}

    def __init__(self):
        self.tokenizer = tiktoken.get_encoding('gpt2')

    def tokenize(self, input_text):
        return self.tokenizer.encode(input_text, allowed_special=self.special_characters)

    def decode(self, tokens):
        return self.tokenizer.decode(tokens)
