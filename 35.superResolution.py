# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 19:38:54 2024

@author: jaege
"""

from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.optimizers import Adam
from keras.datasets import cifar10
import numpy as np
from skimage.transform import resize

# Load CIFAR-10 dataset
(x_train, _), (x_test, _) = cifar10.load_data()

# Normalize pixel values to be between 0 and 1
y_train = x_train.astype('float32') / 255.0 # original is for label(32x32)
y_test = x_test.astype('float32') / 255.0 # original is for label(32x32)

# Down sample the images for training data set(16x16)
x_train = np.array([resize(img, (16, 16, 3), mode='constant', anti_aliasing=True) for img in y_train])

# Down sampl the images for validation
x_test = np.array([resize(img, (16, 16, 3), mode='constant', anti_aliasing=True) for img in y_test])
# Model for super-resolution
input_img = Input(shape=(None,None, 3))
#input_img = Input(shape=(16,16,3))

# Encoder
x = Conv2D(3, (3, 3), activation='relu', padding='same')(input_img)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
encoded = Conv2D(6, (3, 3), activation='relu', padding='same')(x)

# Decoder
x = UpSampling2D((2, 2))(encoded)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x) 

autoencoder = Model(input_img, decoded)


# Compile the model
autoencoder.compile(optimizer=Adam(learning_rate=0.0002),
                      loss='mean_squared_error')
autoencoder.summary()

# Training
autoencoder.fit(x_train, y_train, epochs=20, batch_size=128,
                validation_data=(x_test, y_test))
score = autoencoder.evaluate(x_test, y_test)
print("Test loss:", score)

import matplotlib.pyplot as plt

# Function to plot original and reconstructed images
def plot_images(original, reconstructed, n=5):
    plt.figure(figsize=(10, 4))

    for i in range(n):
        # Display original images
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(original[i])
        plt.title("Original")
#        plt.axis("off")
    plt.show()
    
    plt.figure(figsize=(10, 4))
    for i in range(n):
        # Display reconstructed images
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(reconstructed[i])
        plt.title("Reconstructed")
#        plt.axis("off")

    plt.show()
    
# Generate reconstructed images
reconstructed_images = autoencoder.predict(x_test[:5])
# Plot original and reconstructed images
plot_images(x_test[:5], reconstructed_images)

# 2nd Up resolution
x_test = reconstructed_images
reconstructed_images = autoencoder.predict(x_test[:5])
plot_images(x_test[:5], reconstructed_images)