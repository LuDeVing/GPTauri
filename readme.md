# GPTauri 🐂✨ - A GPT-2 Implementation in TensorFlow

![GPTauri Logo Placeholder](placeholder_logo.gif)

Welcome to GPTauri! This project is a custom implementation of a GPT-2-like transformer architecture built using TensorFlow. It's designed for text generation tasks.

## 📜 Overview

GPTauri leverages the power of transformer models to understand and generate human-like text. It's built with modular components, making it adaptable for various natural language processing experiments.

The core architecture includes:
* **Token and Positional Embeddings**: To represent input text and word order.
* **Transformer Blocks**: Multiple layers of multi-head self-attention and feed-forward networks.
* **Multi-Head Self-Attention**: Allows the model to weigh the importance of different words in the input sequence.
* **Layer Normalization and Dropout**: For stable training and regularization.
* **Linear Output Layer**: To predict the next token in the sequence.

## ⚙️ Architecture Details

The model (`GPTauri` class) consists of several key components:

1.  **Embeddings**:
    * `token_embedding_layer`: Maps input token IDs to dense vectors.
    * `position_embedding_layer`: Adds positional information to token embeddings.
2.  **Transformer Blocks (`TransformerBlock` class )**:
    * Each block contains:
        * `LayerNormalization`.
        * `MultiHeadAttention`.
        * `Dropout`.
        * A feed-forward network (Dense -> GELU -> Dense).
        * Residual connections.
3.  **Multi-Head Attention (`MultiHeadAttention` class)**:
    * Splits queries, keys, and values into multiple heads.
    * Applies scaled dot-product attention with causal masking.
    * Concatenates heads and projects the result.
4.  **Final Layers**:
    * `LayerNormalization` after the transformer blocks.
    * A final `Dense` layer maps the output to vocabulary logits.

## 🛠️ Key Components & Files

* `GPTauri.txt`: Defines the main `GPTauri` model class.
* `TransformerBlock.txt`: Implements the `TransformerBlock` layer.
* `MultiHeadAttention.txt`: Implements the `MultiHeadAttention` layer.
* `Tokenizer.txt`: Handles text tokenization using `tiktoken`'s GPT-2 encoding.
* `TextGeneration.txt`: Provides functions for generating text sequences.
* `DataProcess.txt`: Contains the class for preparing the training dataset.
* `CustomCrossEntropy.txt`: Defines the custom loss function used for training.
* `main.txt`: Includes example scripts for training and running inference.
* `utils.txt`: Contains utility functions (optional: describe if relevant to users).

## 🚀 Getting Started

### Prerequisites

* `tensorflow~=2.10.0`
* `numpy~=1.26.4`
* `tiktoken~=0.9.0`
* `requests~=2.32.3`
* `tqdm~=4.67.1`

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/LuDeVing/GPTauri
    ```
2. After installing, initialize Git LFS: (Optional, do this only when you need pretrained data)
    ```bash
    git lfs install
    ```
   
3. Install required packages:
    ```bash
    pip install -r requirements.txt
    ```

### Configuration

The model configuration is defined within the `GPTauri` class:

```python
CONFIGURATION = {
    "vocabulary_size": 50257, 
    "embedding_dimension": 768,
    "num_heads": 12,           
    "context_length": 1024,    
    "drop_out_rate": 0.1,      
    "qkv_bias": True,          
    "num_layers": 12           
}
```

You can adjust these parameters in GPTauri.txt or modify the class to accept configuration externally.

# Training

Prepare your training data (e.g., a large text file named `input_text.txt`).   

Run the training script (adapt from `main.txt`'s `test_training` function):   

```python
import tensorflow as tf
from GPTauri import GPTauri
from DataProcess import DataProcess
from Tokenizer import Tokenizer

tf.keras.backend.clear_session()
tf.compat.v1.reset_default_graph()

tokenizer = Tokenizer() 
a = open("input_text.txt", "r", encoding='utf-8').read()

# Use default or custom config
model = GPTauri()

# --- Data Processing --- 
data = DataProcess(
    input_text=a,
    tokenizer=tokenizer, # Using the tiktoken tokenizer 
    stride=model.CONFIGURATION[model.context_length] // 2, # Example stride
    context_length=model.CONFIGURATION[model.context_length],
    batch_size=4 # Example batch size
)

# --- Compile and Train ---
model.train(data) # training the model, the model is saved at model WEIGHTS_PATH
```

# Text Generation / Inference

Use the `generate_response` function in `main.txt` as a template, or the `TextGeneration` class directly:  

```python
import tensorflow as tf
from GPTauri import GPTauri
from Tokenizer import Tokenizer
from TextGeneration import TextGeneration

# --- Load Model ---
model = GPTauri()

# Ensure the model architecture matches the saved weights
model.load_weights("model_data\\gpt2_pretrained_weights\\gpt2_model_weights.ckpt")

# Build the model first by calling it with dummy input or using model.build()
model.build(input_shape=(1, model.CONFIGURATION[model.context_length])) 
random_tensor = tf.random.uniform((1, model.CONFIGURATION[model.context_length]),
                                  minval=0, maxval=model.CONFIGURATION['vocabulary_size'], dtype=tf.int32) 
model(random_tensor) # Call with dummy input to build layers

print("✅ Model weights loaded.")
model.summary() 

# --- Generate Text ---
tokenizer = Tokenizer() 
prompt = "The universe is"
print(f"💬 Generating text for prompt: '{prompt}'")

# Using TextGeneration class method 
TextGeneration.generate_and_print(
    input_text=prompt,
    model=model,
    tokenizer=tokenizer,
    num_of_additions=100, # How many new tokens to generate 
    temperature=1.0,      # Controls randomness (higher = more random) 
    top_k=50              # Considers only top_k most likely tokens 
)
```

# 🎬 Demonstrations

## Example 1: Generating Creative Text

### prompt:

On a quiet morning in a small village nestled between the mountains, a group of travelers arrived. They were strangers to the village, and their presence stirred up a mixture of curiosity and unease among the locals. As they walked into the town square, their leader, a tall man with a dark cloak, looked around with a sense of urgency. The village elder, sensing something unusual, approached them and asked, 'What brings you here?' 

### Generated text:

The group was taken in by their leader's curiosity. The elder's voice echoed through the village.

It's hard to describe what it sounded like. A tall, thin man with a dark cloak. His hands were held tightly around his waist. His hands were bound, with no other means of movement, and his face was pale, but his gaze was fixed on the young man. His gaze was so focused on the traveler that it looked as if he had no idea what he was...

## Example 2: GIF of text generation:

![ezgif-2609cda9f41033](https://github.com/user-attachments/assets/a6eb3277-aa04-4c9d-9995-c6dcd39d872c)

# 📄 License

MIT license, use according to your will.

---

Thanks for checking my project out!
