# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 20:39:53 2024

@author: jaege
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 무작위 데이터 생성을 위한 시드 설정
np.random.seed(42)

# 두 가지 클래스의 데이터 생성
# Class 0: (무게, 길이) - 무게는 30부터 70까지, 길이는 10부터 40까지
class_0_weight = np.random.uniform(30, 70, 100)
class_0_length = np.random.uniform(10, 40, 100)

# Class 1: (무게, 길이) - 무게는 60부터 100까지, 길이는 20부터 50까지
class_1_weight = np.random.uniform(60, 100, 100)
class_1_length = np.random.uniform(20, 50, 100)

# 전체 데이터셋 생성
X = np.concatenate([np.column_stack((class_0_weight, class_0_length)),
                    np.column_stack((class_1_weight, class_1_length))])

# 클래스 라벨 생성
y = np.concatenate([np.zeros(100), np.ones(100)])

# 훈련 데이터와 테스트 데이터로 나누기
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SVM 모델 생성 및 훈련
svm_model = svm.SVC(kernel='linear', C=0.005)
svm_model.fit(X_train, y_train)

# 테스트 데이터로 예측
y_pred = svm_model.predict(X_test)

# 정확도 출력
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# 서포트 벡터와 결정 경계 시각화
plt.scatter(class_0_weight, class_0_length, label='Class 0', c='blue', marker='o')
plt.scatter(class_1_weight, class_1_length, label='Class 1', c='orange', marker='x')

plt.scatter(X_test[0,0], X_test[0,1], c = 'red', marker='^') # X_Test의 값 출력, 붉은색
print(X_test[0],"-->", y_pred[0])

# 서포트 벡터 표시
plt.scatter(svm_model.support_vectors_[:, 0], svm_model.support_vectors_[:, 1], s=100, facecolors='none', edgecolors='k', marker='s', label='Support Vectors')

# 결정 경계 시각화
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# 결정 경계 생성
xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50), np.linspace(ylim[0], ylim[1], 50))
Z = svm_model.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 결정 경계 및 마진 시각화
plt.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])

plt.xlabel('Weight')
plt.ylabel('Length')
plt.title('SVM Classification Example')
plt.legend()
plt.show()