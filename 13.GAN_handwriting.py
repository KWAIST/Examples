# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 23:06:17 2024

@author: jaege
"""

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Reshape, Flatten
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt


import tensorflow as tf
from tensorflow.keras import layers, models

# 생성자 모델 정의
def build_generator(latent_dim, image_dim):
    model = models.Sequential()
    model.add(layers.Dense(512, input_dim=latent_dim, activation='relu'))
    model.add(layers.Dense(image_dim, activation='tanh'))
    return model

# 판별자 모델 정의
def build_discriminator(image_dim):
    model = models.Sequential()
    model.add(layers.Dense(1024,input_dim = image_dim, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model

# GAN 모델 정의
def build_gan(generator, discriminator):
    discriminator.trainable = False  # discriminator의 가중치 업데이트를 막음
    gan_input = layers.Input(shape=(latent_dim,))
    x = generator(gan_input)
    gan_output = discriminator(x)
    gan = Model(gan_input, gan_output)
    return gan

# 생성된 이미지 시각화
def plot_generated_images(generator):
    examples = 10
    dim=(1,10)
    figsize = (10,1)
    
    noise = np.random.normal(0, 1, size=(examples, latent_dim))
    generated_images = generator.predict(noise)
    generated_images = generated_images.reshape(examples, 28,28)  # [0, 1]로 복원

    plt.figure(figsize=figsize)
    plt.title("Generated Image")
    plt.axis('off')
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generated_images[i], interpolation='nearest',cmap='gray_r')
        plt.axis('off')
    plt.show()

# MNIST 데이터 로드 : 손글씨 이미지
from tensorflow.keras.datasets import mnist
(X_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
X_train = X_train / 127.5 - 1.0  # 이미지를 [-1, 1]로 정규화

# 생성자, 판별자, GAN 모델 생성
latent_dim = 100
image_dim = 28 * 28
generator = build_generator(latent_dim,image_dim)
discriminator = build_discriminator(image_dim)

# 판별자 컴파일
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002, beta_1=0.5))
discriminator.trainable = False

gan = build_gan(generator, discriminator)
gan.compile(optimizer=Adam(learning_rate=0.0002, beta_1=0.5), loss='binary_crossentropy')
gan.summary()

# 학습 파라미터 설정
epochs = 700
batch_size = 128

# 학습 루프
for epoch in range(epochs):
    # 진짜 이미지 샘플링
    real_images = X_train[np.random.randint(0,X_train.shape[0],batch_size)]
    real_images = real_images.reshape(batch_size,image_dim)

    # 랜덤한 잠재 공간 포인트 생성
    noise = np.random.normal(0, 1, size =(batch_size,latent_dim))

    # 가짜 이미지 생성
    generated_images = generator.predict(noise)

    # 진짜 이미지와 가짜 이미지를 합쳐서 판별자를 훈련
    labels_real = np.ones((batch_size, 1))
    labels_fake = np.zeros((batch_size, 1))
    
    discriminator.trainable = True
    d_loss_real = discriminator.train_on_batch(real_images, labels_real)
    d_loss_fake = discriminator.train_on_batch(generated_images, labels_fake)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    # 생성자를 훈련
    discriminator.trainable = False
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    labels_gan = np.ones((batch_size, 1))
    g_loss = gan.train_on_batch(noise, labels_gan)

    # 1000 에폭마다 결과 출력
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, D Loss: {d_loss}, G Loss: {g_loss}")      
        plot_generated_images(generator)

#원본이미지 출력
dim=(1,10)
figsize = (10,1)
plt.figure(figsize=figsize)
plt.title("Input Image")
plt.axis('off')

for i in range(10):
    plt.subplot(dim[0], dim[1], i+1)
    plt.imshow(X_train[i],interpolation='nearest', cmap='gray_r')
    plt.axis('off')
plt.show()

#최종 결과 출력
plot_generated_images(generator)