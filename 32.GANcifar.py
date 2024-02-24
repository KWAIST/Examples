# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 13:41:24 2024

@author: jaege
"""

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import cifar10
from tensorflow.keras.models import Model

# Generator 모델 정의
def build_generator(latent_dim, image_dim):
    model = models.Sequential()
#    model.add(layers.Dense(512, input_dim=latent_dim, activation='relu')) #origin
    model.add(layers.Dense(8*8*256, use_bias=False, input_shape=(100,))) #origin M
#    model.add(layers.Dense(8*8*256, input_dim=latent_dim)) #origin M

    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    model.add(layers.Reshape((8,8,256)))
    
    model.add(layers.Conv2DTranspose(128, kernel_size=(5, 5), strides=(1, 1), 
                                     padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2DTranspose(64, kernel_size=(5, 5), strides=(2, 2), 
                                     padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(3, kernel_size=(5, 5), strides=(2, 2), 
                                     padding='same', activation='tanh'))

    return model

# Discriminator 모델 정의
def build_discriminator(image_dim):
    model = models.Sequential()
    model.add(layers.Conv2D(64,kernel_size=(5,5), strides=(2, 2),
                            padding='same',input_shape=[32,32,3]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    model.add(layers.Conv2D(128, kernel_size=(5, 5), strides=(2, 2),
                            padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='sigmoid'))
    return model

# GAN 모델 정의
def build_gan(generator, discriminator):
    discriminator.trainable = False
    model = models.Sequential()
    model.add(generator)
    model.add(discriminator)
    return model

# 생성된 이미지 확인
def plot_generated_images(generator, examples=10, dim=(1, 10), figsize=(10, 1)):
    noise = np.random.normal(0, 1, size=(examples, latent_dim))
    generated_images = generator.predict(noise)
    generated_images = (generated_images + 1) / 2  # [0, 1] 범위로 변환

    generated_images = generated_images.reshape(examples, 32, 32,3)

    plt.figure(figsize=figsize)
    plt.title("Generated Image")
    plt.axis('off')
    for i in range(10):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generated_images[i], interpolation='nearest')#, cmap='gray_r')
        plt.axis('off')
    plt.show()

# Load CIFAR-10 dataset
(X_train, _), (X_test, _) = cifar10.load_data()

#Nomalizing
x_train = (X_train - 127.5) / 127.5
x_test = (X_test - 127.5) / 127.5

# 모델 및 최적화기 초기화
latent_dim = 100
image_dim = 32 * 32 * 3
lr = 0.0002

generator = build_generator(latent_dim, image_dim)
generator.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr, 
                                    beta_1=0.5), loss='binary_crossentropy')
#generator.summary()
discriminator = build_discriminator(image_dim)
discriminator.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr, 
                                    beta_1=0.5), loss='binary_crossentropy')
#discriminator.summary()
gan = build_gan(generator, discriminator)
gan.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr, 
                                    beta_1=0.5), loss='binary_crossentropy')
gan.summary()

# 학습
batch_size = 128
epochs = 40000
num_examples = 10
d_loss = []
g_loss = []

for epoch in range(epochs):
    noise = np.random.normal(0, 1, size=(batch_size, latent_dim))
    generated_images = generator.predict(noise)

    real_images = x_train[np.random.randint(0, x_train.shape[0], batch_size)]
  
    real_images = real_images.reshape(batch_size, 32,32,3)

    labels_real = np.ones((batch_size, 1))
    labels_fake = np.zeros((batch_size, 1))

    discriminator.trainable = True
    d_loss_real = discriminator.train_on_batch(real_images, labels_real)
    d_loss_fake = discriminator.train_on_batch(generated_images, labels_fake)

    d_loss.append(0.5 * np.add(d_loss_real,d_loss_fake))

    noise = np.random.normal(0, 1, size=(batch_size, latent_dim))
    labels_gan = np.ones((batch_size, 1))
    
    discriminator.trainable = False 
    g_loss.append(gan.train_on_batch(noise, labels_gan))

    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, D Loss: {d_loss[epoch]}, G Loss: {g_loss[epoch]}")
        plot_generated_images(generator)
print(f"Epoch {epoch}, D Loss: {d_loss[epoch]}, G Loss: {g_loss[epoch]}")

#plot loss trends
plt.plot(d_loss, label="d_loss")
plt.plot(g_loss, label="g_loss")
plt.legend()

#원본이미지 출력
dim=(1,10)
figsize = (10,1)
plt.figure(figsize=figsize)
plt.title("Input Image")
plt.axis('off')

for i in range(10):
    plt.subplot(dim[0], dim[1], i+1)
    plt.imshow(X_train[i],interpolation='nearest')#, cmap='gray_r')
    plt.axis('off')

plot_generated_images(generator)