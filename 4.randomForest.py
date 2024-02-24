# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 20:33:34 2024

@author: jaege
"""

# 필요한 라이브러리 임포트
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# 숫자 데이터셋 로드
digits = load_digits()
X = digits.data
y = digits.target

# 훈련 데이터 이미지 예시
import matplotlib.pyplot as plt

plt.gray()
plt.matshow(digits.images[0])
plt.matshow(digits.images[1])
plt.matshow(digits.images[2])
plt.matshow(digits.images[3])
plt.show()

# 훈련 세트와 테스트 세트로 나누기
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 랜덤포레스트 모델 생성 및 훈련
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 테스트 세트로 예측
y_pred = model.predict(X_test)

# 정확도 및 혼동 행렬 출력
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print('Confusion Matrix:')
print(conf_matrix)