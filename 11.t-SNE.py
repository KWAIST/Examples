# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 21:33:04 2024

@author: jaege
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.datasets import load_iris

# Iris 데이터셋 로드
iris = load_iris()
X = iris.data
y = iris.target

# t-SNE 모델 초기화
tsne = TSNE(n_components=2, random_state=42)

# t-SNE 수행
X_tsne = tsne.fit_transform(X)

# t-SNE 결과 시각화
plt.figure(figsize=(8, 6))
for i, c in zip(np.unique(y), ['red', 'green', 'blue']):
    plt.scatter(X_tsne[y == i, 0], X_tsne[y == i, 1], c=c, label=f'Class {i}', alpha=0.7)

plt.title('t-SNE of Iris Dataset')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.legend()
plt.show()