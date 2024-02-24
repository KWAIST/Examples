# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 10:47:29 2024

@author: jaege
"""

import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input, Dense
from keras.models import Model

# 데이터 생성
np.random.seed(42)

# 정상 데이터 생성
normal_data = np.random.randn(100, 2) * 2

# 이상치 데이터 생성
outlier_data = np.random.uniform(low=-10, high=10, size=(5, 2))

# 합치기
data = np.vstack([normal_data, outlier_data])

# 오토인코더 모델 정의
input_dim = data.shape[1]

input_layer = Input(shape=(input_dim,))
encoded = Dense(8, activation='relu')(input_layer)
decoded = Dense(input_dim, activation='linear')(encoded)

autoencoder = Model(input_layer, decoded)

autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# 정상 데이터로 훈련
autoencoder.fit(normal_data, normal_data, epochs=100, batch_size=32, shuffle=True, validation_split=0.2, verbose=0)

# 이상 데이터 예측
predictions = autoencoder.predict(data)

# 각 데이터 포인트의 재구성 오차 계산
reconstruction_errors = np.mean(np.square(data - predictions), axis=1)

# 재구성 오차를 시각화
plt.scatter(range(len(reconstruction_errors)), reconstruction_errors, c='b', marker='o', label='Reconstruction Error')
plt.axhline(y=np.mean(reconstruction_errors), color='r', linestyle='--', label='Threshold')
plt.title('Autoencoder for Anomaly Detection')
plt.xlabel('Data Point Index')
plt.ylabel('Reconstruction Error')
plt.legend()
plt.show()