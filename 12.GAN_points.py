# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 21:35:00 2024

@author: jaege
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization, Reshape, Flatten, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import initializers

# 생성자 모델 정의
def build_generator(latent_dim):
    model = Sequential()
    model.add(Dense(256, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2)) 
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(2, activation='tanh'))
    return model

# 판별자 모델 정의
def build_discriminator(input_dim):
    model = Sequential()
    model.add(Dense(256, input_dim=input_dim))
    model.add(LeakyReLU(alpha=0.2)) 
    model.add(Dense(1, activation='sigmoid'))
    model.add(BatchNormalization(momentum=0.8))
    return model

# GAN 모델 정의
def build_gan(generator, discriminator):
    discriminator.trainable = False
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    return model

# GAN 모델 및 생성자, 판별자 컴파일
latent_dim = 10
generator = build_generator(latent_dim)
discriminator = build_discriminator(2)
gan = build_gan(generator, discriminator)
discriminator.compile(optimizer=Adam(0.0022, 0.5), loss='binary_crossentropy', metrics=['accuracy']) 
gan.compile(optimizer=Adam(0.0022, 0.5), loss='binary_crossentropy')

# 가짜 샘플 생성 함수
def generate_fake_samples(generator, latent_dim, n):
    latent_points = np.random.randn(latent_dim * n).reshape(n, latent_dim)
    return generator.predict(latent_points)

# 실제 데이터 생성 함수
def generate_real_samples(n):
    x1 = np.random.randn(n) + 5
    x2 = np.random.randn(n) + 5
    return np.vstack((x1, x2)).T

# GAN 학습 함수
def train_gan(generator, discriminator, gan, latent_dim, n_epochs=1000, n_batch=128):
    for epoch in range(n_epochs):
        # 실제 데이터
        X_real = generate_real_samples(n_batch)
        y_real = np.ones((n_batch, 1))
        # 판별자 훈련
        d_loss_real = discriminator.train_on_batch(X_real, y_real)
        # 가짜 데이터
        X_fake = generate_fake_samples(generator, latent_dim, n_batch)
        y_fake = np.zeros((n_batch, 1))
        # 판별자 훈련
        d_loss_fake = discriminator.train_on_batch(X_fake, y_fake)
        # 생성자 훈련
        X_gan = np.random.randn(latent_dim * n_batch).reshape(n_batch, latent_dim)
        y_gan = np.ones((n_batch, 1))
        g_loss = gan.train_on_batch(X_gan, y_gan)
        # 결과 출력
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, D Loss Real: {d_loss_real}, D Loss Fake: {d_loss_fake}, G Loss: {g_loss}")

# GAN 학습
train_gan(generator, discriminator, gan, latent_dim)

# 생성된 데이터 및 실제 데이터 시각화
generated_samples = generate_fake_samples(generator, latent_dim, 1000)
real_samples = generate_real_samples(1000)

plt.scatter(real_samples[:, 0], real_samples[:, 1], color='blue', alpha=0.5, label='Real Data')
plt.scatter(generated_samples[:, 0], generated_samples[:, 1], color='red', alpha=0.5, label='Generated Data')
plt.legend()
plt.title('Generated vs Real Data Distribution')
plt.show()

#GAN으로 만든 새로운 데이터 시각화
plt.scatter(generated_samples[:, 0], generated_samples[:, 1], color='red', alpha=0.5, label='Generated Data')
plt.legend()
plt.title('Zoom In Generated Data Distribution')
plt.show()
print(generated_samples)