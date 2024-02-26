# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 20:58:49 2024

@author: jaege
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# 데이터 생성
data, _ = make_blobs(n_samples=300, centers=4, random_state=42)

# K-평균 모델 초기화
kmeans = KMeans(n_clusters=4, random_state=42)

# 모델 학습
kmeans.fit(data)

# 클러스터 중심과 할당된 클러스터 시각화
plt.scatter(data[:, 0], data[:, 1], c=kmeans.labels_, cmap='viridis', s=50, alpha=0.8)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='X', s=200)
plt.title('K-Means Clustering')

# 군집 예측
_data = np.array( [[-2.5, 5], [5, 7],[-7,-2],[-10,10]])

predictions = kmeans.predict(_data)
plt.scatter(_data[:,0], _data[:,1], c=predictions, cmap='viridis', s=100, alpha=0.8, marker='^')
plt.show()
print(predictions)