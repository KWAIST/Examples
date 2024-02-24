# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 19:27:33 2024

@author: jaege
"""

import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw
import json

# COCO 데이터셋의 클래스 목록
coco_classes = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
    'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
    'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def load_model():
    # 미리 훈련된 Faster R-CNN 모델을 로드합니다.
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    return model

def predict(model, image_path, threshold=0.5):
    # 이미지를 불러오고 전처리합니다.
    image = Image.open(image_path).convert("RGB")
    image_tensor = F.to_tensor(image).unsqueeze(0)

    # 모델에 입력하고 예측을 수행합니다.
    with torch.no_grad():
        prediction = model(image_tensor)

    # 예측 결과를 시각화합니다.
    draw = ImageDraw.Draw(image)
    for score, label, box in zip(prediction[0]['scores'], prediction[0]['labels'], prediction[0]['boxes']):
        if score >= threshold:
            box = [round(i, 2) for i in box.tolist()]
            draw.rectangle(box, outline="red", width=2)
            draw.text((box[0], box[1]), f"{coco_classes[label]}: {round(score.item(), 3)}", fill="red")

    image.show()

image_path = '/Users/Jaege/TestPGM/Yolo/image4.jpeg'

# 모델을 로드합니다.
model = load_model()

# 예측을 수행하고 결과를 시각화합니다.
predict(model, image_path)