# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 11:21:31 2024

@author: jaege
"""

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.activations import gelu
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
import tensorflow_datasets as tfds

# Load IMDB dataset
(train_data, test_data), info = tfds.load(
    'imdb_reviews/subwords8k',
    split=(tfds.Split.TRAIN, tfds.Split.TEST),
    with_info=True,
    as_supervised=True
)

encoder = info.features['text'].encoder
vocab_size = encoder.vocab_size

# Prepare the data
BUFFER_SIZE = 10000
BATCH_SIZE = 32  # Decreased batch size

train_data = train_data.shuffle(BUFFER_SIZE)
train_data = train_data.padded_batch(BATCH_SIZE, padded_shapes=([None], []))
test_data = test_data.padded_batch(BATCH_SIZE, padded_shapes=([None], []))

# Transformer Model
def transformer_model(vocab_size, d_model, n_heads, n_encoder_layers, n_dense_layers, dropout_rate, max_sequence_length):
    inputs = Input(shape=(None,))
    embedding_layer = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)

    transformer_block = embedding_layer
    for _ in range(n_encoder_layers):
        transformer_block = transformer_encoder(transformer_block, d_model, n_heads, dropout_rate)

    transformer_block = tf.keras.layers.GlobalAveragePooling1D()(transformer_block)
    for _ in range(n_dense_layers):
        transformer_block = Dense(d_model, activation=gelu)(transformer_block)
        transformer_block = Dropout(dropout_rate)(transformer_block)

    outputs = Dense(2, activation='softmax')(transformer_block)
    model = Model(inputs=inputs, outputs=outputs)
    return model

def transformer_encoder(inputs, d_model, n_heads, dropout_rate):
    # Multi-head self-attention
    attention = tf.keras.layers.MultiHeadAttention(
        key_dim=d_model // n_heads,
        num_heads=n_heads,
        dropout=dropout_rate
    )(inputs, inputs)
    attention = Dropout(dropout_rate)(attention)
    res = tf.keras.layers.Add()([inputs, attention])

    # Feedforward Neural Network
    ffnn = tf.keras.Sequential([
        Dense(d_model, activation=gelu),
        Dropout(dropout_rate),
        Dense(d_model)
    ])
    ffnn_output = ffnn(res)
    res = tf.keras.layers.Add()([res, ffnn_output])

    return res

# Hyperparameters
D_MODEL = 64  # From 512, Decreased model size
N_HEADS = 2    # From 4, Decreased number of heads
N_ENCODER_LAYERS = 2
N_DENSE_LAYERS = 2
DROPOUT_RATE = 0.1
MAX_SEQUENCE_LENGTH = 200

# Build and compile the model
model = transformer_model(vocab_size, D_MODEL, N_HEADS, N_ENCODER_LAYERS, N_DENSE_LAYERS, DROPOUT_RATE, MAX_SEQUENCE_LENGTH)

model.compile(
    optimizer=Adam(learning_rate=3e-5),
    loss=SparseCategoricalCrossentropy(),
    metrics=[SparseCategoricalAccuracy()]
)

# Train the model
history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=5,
    callbacks=[EarlyStopping(patience=2, restore_best_weights=True)]
)

# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(test_data)

print(f'Test Loss: {test_loss:.4f}')
print(f'Test Accuracy: {test_accuracy:.4f}')

# Make predictions on a few examples
sample_texts = [
    "This movie is fantastic!",
    "I didn't like the plot of this film."
]

# Convert sample texts to sequences
sample_sequences = [encoder.encode(text) for text in sample_texts]

# Pad sequences
sample_sequences_padded = tf.keras.preprocessing.sequence.pad_sequences(sample_sequences, padding='post')

# Make predictions
predictions = model.predict(sample_sequences_padded)

# Display predictions
for i, text in enumerate(sample_texts):
    sentiment = "Positive" if np.argmax(predictions[i]) == 1 else "Negative"
    print(f"Text: {text}")
    print(f"Predicted Sentiment: {sentiment}")
    print()