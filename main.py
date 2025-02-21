import tensorflow as tf

from GPTauri import GPTauri
from DataProcess import DataProcess

a = open("input_text.txt", "r", encoding='utf-8').read()
a = a.replace('\n', '')
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

    logits = model.call(context_len_input_batches)
    logits = tf.reshape(logits, [-1, model.CONFIGURATION[model.vocabulary_size]])

    loss_func = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    return loss_func(logits, one_hot_output)


def generate_text(input_batch, num_of_additions):
    for _ in range(num_of_additions):
        context_len_input_batches = input_batch[:, -model.CONFIGURATION[model.context_length]:]

        output = model.call(context_len_input_batches)

        logits = output[:, -1, :]

        probabilities = tf.keras.activations.softmax(output, axis=-1)

        next_word = tf.argmax(probabilities, axis=-1)
        next_word = tf.expand_dims(next_word, axis=-1)

        print([data.tokenizer.decode(word) for word in next_word[0]])

        probabilities = tf.keras.activations.softmax(logits, axis=-1)

        next_word = tf.argmax(probabilities, axis=-1)
        next_word = tf.expand_dims(next_word, axis=-1)

        input_batch = tf.concat([input_batch, next_word], axis=1)

    return input_batch


lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate=0.005,
    decay_steps=40,
    end_learning_rate=0.001
)
optimizer = tf.keras.optimizers.Adam(lr_schedule)

def calc_loss_batch(input_batch, target_batch):
    with tf.GradientTape() as tape:
        predictions = model.call(input_batch, training=True)
        loss = tf.losses.sparse_categorical_crossentropy(target_batch, predictions)
        loss = tf.reduce_mean(loss)
    return loss, tape


def apply_gradients(tape, loss):
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def train_model_simple(num_epochs):
    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    # Main training loop
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        for input_batch, target_batch in data.dataset:
            # Calculate loss and gradients
            loss, tape = calc_loss_batch(input_batch, target_batch)
            apply_gradients(tape, loss)

            tokens_seen += tf.size(input_batch).numpy()  # Count number of tokens
            global_step += 1

            # Optional evaluation step
            if global_step % 5 == 0:
                print(f"Ep {epoch+1} (Step {global_step:06d}): ")
                print(loss)
                print(data.tokenizer.decode(generate_text(input_batch, 5)[0][-10:]))

    return train_losses, val_losses, track_tokens_seen



def train_model():
    inputs, outputs = 1, 1

    for batch in data.dataset:
        inputs, outputs = batch

    model.call(inputs)

    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.summary()

    # model.fit(data.dataset, epochs=10)
    train_model_simple(10)


train_model()

for batch in data.dataset:
    inputs, outputs = batch
    print(data.tokenizer.decode(generate_text(inputs, 5)[0][-5:]))
    print('---')
    print('---')
