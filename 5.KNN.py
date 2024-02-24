# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 20:38:58 2024

@author: jaege
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# 두 종류의 물체 데이터 생성
np.random.seed(42)
# 물체 A: (무게, 길이) - 무게는 30부터 70까지, 길이는 10부터 40까지
A_weight = np.random.uniform(30, 70, 100)
A_length = np.random.uniform(10, 40, 100)
A_data = np.column_stack((A_weight, A_length))
A_labels = np.zeros(100)  # 물체 A는 라벨 0

# 물체 B: (무게, 길이) - 무게는 60부터 100까지, 길이는 20부터 50까지
B_weight = np.random.uniform(60, 100, 100)
B_length = np.random.uniform(20, 50, 100)
B_data = np.column_stack((B_weight, B_length))
B_labels = np.ones(100)  # 물체 B는 라벨 1

# 전체 데이터셋 생성
X = np.concatenate((A_data, B_data), axis=0)
y = np.concatenate((A_labels, B_labels))

# KNN 모델 생성 및 훈련
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X, y)

# 분류 결과를 시각화
plt.figure(figsize=(10, 6))

# 물체 A의 데이터 시각화
plt.scatter(A_weight, A_length, label='Object A', color='blue', alpha=0.7)

# 물체 B의 데이터 시각화
plt.scatter(B_weight, B_length, label='Object B', color='orange', alpha=0.7)

# 새로운 물체의 데이터 생성
new_object = np.array([[65, 35]])  # 무게 65 길이 35인 물체

# 새로운 물체를 모델에 입력하여 예측
prediction = knn_model.predict(new_object)

# 예측 결과 시각화
plt.scatter(new_object[:, 0], new_object[:, 1], label=f'New Object (Predicted: {int(prediction)})', color='red', marker='x', s=100)

plt.xlabel('Weight')
plt.ylabel('Length')
plt.title('KNN Classification Example')
plt.legend()
plt.show()