# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 19:33:28 2024

@author: jaege
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import vgg19
from tensorflow.keras import backend as K
from tensorflow.image import resize
from scipy.optimize import fmin_l_bfgs_b
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input
from tensorflow.keras import models, losses, optimizers

# Load and preprocess images
target_image_path = '/Users/jaege/TestPGM/pattern2.jpg'  # 원본 이미지
style_reference_image_path = '/Users/jaege/TestPGM/pattern4.jpg'
width, height = load_img(target_image_path).size

#VGG19 pre-trained model supports 224x224 size only
img_height = 224
img_width = 224 

optimizer = tf.keras.optimizers.Adam(learning_rate=0.02)

def deprocess_image(x):
    x = x.reshape((img_height, img_width, 3))
    # Reverse of preprocess_input
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x

# Content loss
def content_loss(base, target):
    return tf.reduce_mean(tf.square(base - target))

# Define the neural style loss
def style_loss(style, combination):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    
    # Ensure S has 3 dimensions
    if len(S.shape) == 2:
        S = tf.expand_dims(S, axis=0)

    # Resize style matrix to match the size of the combination matrix
    new_shape = tf.shape(C)
    S_resized = tf.image.resize(S, (new_shape[0], new_shape[1]), method='nearest')

    channels = 3
    size = tf.cast(new_shape[0] * new_shape[1], dtype=tf.float32)

    # Adjust the computation to handle the batch dimension in S_resized
    batch_size = tf.shape(S_resized)[0]
    S_resized = tf.reshape(S_resized, (size, -1))
    C = tf.reshape(C, (size, -1))
 
    return tf.reduce_sum(tf.square(S_resized - C)) / (4.0 * (channels ** 2) * (size ** 2))

# Compute the Gram matrix
def gram_matrix(x):
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram

# Define total variation loss to maintain spatial coherence
def total_variation_loss(x):
#    print("x", x.shape)
    h, w = x.shape[1], x.shape[2]
    a = tf.square(x[:, :h-1, :w-1, :] - x[:, 1:, :w-1, :])
    b = tf.square(x[:, :h-1, :w-1, :] - x[:, :h-1, 1:, :])
    return tf.reduce_sum(tf.pow(a + b, 1.25))

# Combine losses
def total_loss(model, loss_weights, generated_image, content_image, style_image):
    content_weight, style_weight = loss_weights

    model_outputs = model(generated_image)
    content_features = model_outputs[0]
    style_features = model_outputs[1:]

    content_loss_value = content_loss(content_features[0], model(content_image)[0])

    style_loss_value = 0
    for layer, weight in zip(style_features, style_weight):
        style_loss_value += weight * style_loss(layer[0], model(style_image)[0])

    total_variation_loss_value = total_variation_loss(generated_image)

    total_loss = content_weight * content_loss_value + style_loss_value + total_variation_loss_value

    return total_loss

def style_transfer(target_image_path, style_reference_image_path, iterations=20):
    # Load images400
    target_image = tf.keras.preprocessing.image.load_img(target_image_path, target_size=(224, 224))
    style_reference_image = tf.keras.preprocessing.image.load_img(style_reference_image_path, target_size=(224, 224))

    plt.figure(figsize=(10, 10))

    plt.subplot(131)
    plt.imshow(target_image)
    plt.title('Content Image')

    plt.subplot(132)
    plt.imshow(style_reference_image)
    plt.title('Style Image')
    
    # Convert images to arrays and expand dimensions
    target_image = tf.expand_dims(tf.keras.preprocessing.image.img_to_array(target_image), axis=0)
    style_reference_image = tf.expand_dims(tf.keras.preprocessing.image.img_to_array(style_reference_image), axis=0)
    original_image = target_image
    
    # Convert arrays to tensors
    target_image = tf.convert_to_tensor(target_image)
    style_reference_image = tf.convert_to_tensor(style_reference_image)

    combination_image = tf.Variable(original_image)
    base_model = vgg19.VGG19(weights='imagenet', include_top=False)
    input_layer = Input(shape=(None, None, 3))

    # 기존의 VGG19 모델을 새로운 입력에 적용합니다
    vgg_output = base_model(input_layer)
    modified_output = tf.keras.layers.Conv2D(3, (1, 1),
                                   activation='sigmoid')(vgg_output)

    # 업샘플링 레이어를 추가합니다
    upsample_layer = tf.keras.layers.UpSampling2D(size=(32, 32))
    upsampled_output = upsample_layer(modified_output)

    # 새로운 모델을 정의합니다
    model = tf.keras.Model(inputs=input_layer, outputs=upsampled_output)

    content_weight = 1e3
    style_weight = [1e-2, 1e-2, 1e-3, 1e-3, 1e-3]
    loss_weights = (content_weight, style_weight)
    generated_image = combination_image
    content_image = target_image
    style_image = style_reference_image

####
    outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])

    for i in range(iterations):
        print('Iteration', i)
        with tf.GradientTape() as tape:
            loss_value = total_loss(model, loss_weights, generated_image, content_image, style_image)

        grads = tape.gradient(loss_value, combination_image)
        optimizer.apply_gradients([(grads, combination_image)])
        combination_image.assign(tf.clip_by_value(
                                  combination_image,clip_value_min=0.0,
                                  clip_value_max=255.0))

        if i % 100 == 0:
            img = deprocess_image(combination_image.numpy())

    plt.subplot(133)
    plt.imshow(img)
    plt.title('Combination Image')

    plt.show()

# Run style transfer
style_transfer(target_image_path, style_reference_image_path)