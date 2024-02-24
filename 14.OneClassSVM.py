# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 15:52:18 2024

@author: jaege
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

# 데이터 생성
np.random.seed(42)

# 정상 데이터 생성
normal_data = np.random.randn(100, 2) * 2

# 이상치 데이터 생성
outlier_data = np.random.uniform(low=-10, high=10, size=(5, 2))

# 합치기
data = np.vstack([normal_data, outlier_data])

# 정상 데이터에 레이블 1, 이상치에 레이블 -1 할당
labels = np.ones(len(normal_data))
labels = np.hstack([labels, -np.ones(len(outlier_data))])

# SVM 모델 생성
clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)

# 모델 훈련
clf.fit(data)

# 예측
pred_labels = clf.predict(data)

# 시각화
xx, yy = np.meshgrid(np.linspace(-15, 15, 500), np.linspace(-15, 15, 500))
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.title("One-Class SVM for Anomaly Detection")
plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.PuBu, alpha=0.7)
plt.scatter(data[:, 0], data[:, 1], c="white", s=20, edgecolors="k")
plt.scatter(data[pred_labels == -1, 0], data[pred_labels == -1, 1], c="red", s=20, edgecolors="k")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()