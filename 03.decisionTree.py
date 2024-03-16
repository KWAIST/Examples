# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 20:31:58 2024

@author: jaege_son
"""

# 필요한 라이브러리 임포트
from sklearn.tree import DecisionTreeClassifier, export_text
import pandas as pd
from sklearn.tree import export_graphviz
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import graphviz

# 날씨 데이터 생성
data = {
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain', 'Sunny', 'Overcast', 'Overcast', 'Rain'],
    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
    'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Strong'],
    'Play Golf': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
}

df = pd.DataFrame(data)

# 특성과 레이블 분리
X = df.drop('Play Golf', axis=1)
y = df['Play Golf']

# 범주형 데이터를 숫자로 변환 (원-핫 인코딩)
X_encoded = pd.get_dummies(X)

# 결정 트리 모델 생성 및 훈련
model = DecisionTreeClassifier()
model.fit(X_encoded, y)

# 결정 트리 시각화
plt.figure(figsize=(12, 8))
plot_tree(model, filled=True, feature_names=list(X_encoded.columns), class_names=['No', 'Yes'], rounded=True)
plt.show()
