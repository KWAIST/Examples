# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 11:03:06 2024

@author: jaege
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# 예제 텍스트 데이터 생성
texts = ["This is a positive sentence.",
         "I love natural language processing.",
         "Negative sentiment is not good.",
         "RNNs are powerful for sequence modeling.",
         "Today's dinner is very good.",
         "My yesterday golf scores were bad.",
         "The political news are not positive.",
         "He feels lonely.",
         "She is very sensitive.",
         "All of my friends are going to get jobs."
        ]

labels = [1, 1, 0, 1, 1, 0, 0, 0, 1, 1]  # 1: Positive, 0: Negative

# 텍스트 데이터를 토큰화하고 시퀀스로 변환
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

# 시퀀스를 패딩하여 길이를 맞춤
padded_sequences = pad_sequences(sequences)

# 데이터를 학습 및 테스트 세트로 분할
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, 
                                                    test_size=0.2, random_state=42)

# y_train을 넘파이 배열로 변환
y_train = np.array(y_train)

# RNN 모델 정의
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index) + 1, 
                              output_dim=16, 
                              input_length=padded_sequences.shape[1]),
    tf.keras.layers.SimpleRNN(units=8),
    tf.keras.layers.Dense(units=1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 모델 훈련
model.fit(X_train, y_train, epochs=20, batch_size=2)

# 모델 평가
y_test = np.array(y_test)
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}')

# 텍스트 예측
texts_to_predict = ["The dinner was very good.", "I don't like this at all."]
sequences_to_predict = tokenizer.texts_to_sequences(texts_to_predict)
padded_sequences_to_predict = pad_sequences(sequences_to_predict, maxlen=padded_sequences.shape[1])

# 모델 예측
predictions = model.predict(padded_sequences_to_predict)

# 결과 출력
for text, prediction in zip(texts_to_predict, predictions):
    sentiment = "Positive" if prediction > 0.5 else "Negative"
    print(f'Text: {text} | Sentiment: {sentiment} | Probability: {prediction[0]:.4f}')