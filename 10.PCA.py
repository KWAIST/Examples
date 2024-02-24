# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 21:31:42 2024

@author: jaege
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

# Iris 데이터셋 로드
iris = load_iris()
X = iris.data
y = iris.target

print(iris.feature_names)
print(iris.target_names)

# PCA 모델 초기화
pca = PCA(n_components=2)

# 주성분 분석 수행
X_pca = pca.fit_transform(X)

# 주성분 분석 결과 시각화
plt.figure(figsize=(8, 6))
for i, c in zip(np.unique(y), ['red', 'green', 'blue']):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], c=c, label=f'Class {i}', alpha=0.7)

plt.title('PCA of Iris Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()