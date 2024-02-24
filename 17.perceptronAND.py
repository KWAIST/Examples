# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 10:52:53 2024

@author: jaege
"""

import numpy as np

# 퍼셉트론 클래스 정의
class Perceptron:
    def __init__(self, input_size):
        # 가중치와 편향 초기화
        #self.weights = np.random.rand(input_size)
        #self.bias = np.random.rand()
        self.weights = [0.7, 0.12]
        self.bias = 0.57
       
    def predict(self, inputs):
        # 퍼셉트론의 예측값 계산 (Step 함수 사용)
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        return 1 if weighted_sum >= 0 else 0

    def train(self, training_data, epochs=10, learning_rate=0.11):
        for epoch in range(epochs):
            for inputs, target in training_data:
                # 예측값 계산
                prediction = self.predict(inputs)
                # 가중치 업데이트
                self.weights += learning_rate * (target - prediction) * inputs
                # 편향 업데이트
                self.bias += learning_rate * (target - prediction)
                
                print("inputs",inputs,"weights", self.weights,"bias", self.bias,"Output", prediction)

# AND 문제에 대한 훈련 데이터
training_data = [
    (np.array([1, 1]), 1),
    (np.array([1, 0]), 0),
    (np.array([0, 1]), 0),
    (np.array([0, 0]), 0)
]

# 단층 퍼셉트론 생성 및 훈련
input_size = len(training_data[0][0])
perceptron = Perceptron(input_size)
perceptron.train(training_data)

# 테스트
test_data = [
    np.array([0, 0]),
    np.array([0, 1]),
    np.array([1, 0]),
    np.array([1, 1])
]

for test_input in test_data:
    prediction = perceptron.predict(test_input)
    print(f"Input: {test_input}, Prediction: {prediction}")

#임의값으로 예측
perceptron.predict([1,1])