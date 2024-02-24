# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 19:01:10 2024

@author: jaege
"""

import cv2
import numpy as np

def apply_nms(boxes, scores, threshold=0.5):
    indices = cv2.dnn.NMSBoxes(boxes, scores, threshold, threshold)
    return indices

# YOLO 설정 파일과 가중치 파일 경로
yolo_config_path = "C:/Users/jaege/TestPGM/Yolo/yolov3.cfg"
yolo_weights_path = "C:/Users/jaege/TestPGM/Yolo/yolov3.weights"
yolo_classes_path = "C:/Users/jaege/TestPGM/Yolo/coco.names"

# YOLO 네트워크 로드
net = cv2.dnn.readNetFromDarknet(yolo_config_path, yolo_weights_path)

# 클래스 이름 로드
with open(yolo_classes_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# 이미지 로드, 여기에 적당한 이미지를 넣으세요
image_path = "C:/Users/jaege/TestPGM/Yolo/image4.jpeg"
image = cv2.imread(image_path)
height, width = image.shape[:2]


# YOLO 입력 이미지 전처리
blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
net.setInput(blob)

# YOLO 출력
output_layers_names = net.getUnconnectedOutLayersNames()
outs = net.forward(output_layers_names)

# 바운딩 박스 정보 추출
conf_threshold = 0.5
boxes = []
confidences = []
class_ids = []

for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > conf_threshold:
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# NMS 적용
indices = apply_nms(boxes, confidences)

# 남은 바운딩 박스에 대한 후속 작업 수행
for i in indices:
    box = boxes[i]
    confidence = confidences[i]
    class_id = class_ids[i]

    # 바운딩 박스 그리기
    color = (0, 255, 0)  # RGB
    cv2.rectangle(image, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), color, 1)

    # 클래스 이름과 신뢰도 표시
    label = f"{classes[class_id]}: {confidence:.2f}"
    cv2.putText(image, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

# 결과 이미지 출력
cv2.imshow("YOLO Object Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()