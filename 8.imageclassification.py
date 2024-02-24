# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 20:43:22 2024

@author: jaege
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

# 데이터셋 다운로드
train_generator = tf.keras.utils.get_file('cats_and_dogs/train', 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip',
       untar=True, cache_dir='/Users/jaege/TestPGM/')
validation_generator = tf.keras.utils.get_file('cats_and_dogs/validation', 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip',
       untar=True, cache_dir='/Users/jaege/TestPGM/')
# 이미지 데이터 증강을 위한 ImageDataGenerator 설정
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
validation_datagen = ImageDataGenerator(rescale=1./255)

# 데이터 로딩 및 전처리
train_dataset = train_datagen.flow_from_directory(
    train_generator, target_size=(197, 150), batch_size=200, class_mode='binary')

validation_dataset = validation_datagen.flow_from_directory(
    validation_generator, target_size=(197, 150), batch_size=150, class_mode='binary')

print(train_dataset.class_indices)
print(validation_dataset.class_indices)

# CNN 모델 구성
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(197, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# 모델 컴파일
opt = keras.optimizers.Adam(learning_rate=0.0001)
#opt = tf.keras.optimizers.legacy.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
#model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])     

# 모델 훈련
history = model.fit(train_dataset, epochs=10, validation_data=validation_dataset)

# 결과 시각화
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

###Prediction
import numpy as np
from tensorflow.keras.preprocessing import image

# 예측할 이미지 경로
image_path = "/Users/jaege/TestPGM/cat.jpg"

# 이미지 불러오기 및 전처리
img = image.load_img(image_path, target_size=(197, 150))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0  # 이미지를 모델에 입력하기 전에 0에서 1 사이의 값으로 스케일 조정

# 모델 예측
prediction = model.predict(img_array)
print(prediction)

# 결과 출력
if prediction[0][0] > 0.5:
    print(prediction, "예측 결과: 강아지")
else:
    print(prediction, "예측 결과: 고양이")