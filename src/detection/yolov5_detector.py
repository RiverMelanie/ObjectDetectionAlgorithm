import cv2
import numpy as np
import torch
import os
import sys
from ultralytics import YOLO

# 获取项目根目录的路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(project_root)

class YOLOv5Detector:
    def __init__(self, model_path='models/yolov5s.pt', 
                 conf_threshold=0.3,  # 降低置信度阈值，提高召回
                 iou_threshold=0.3,   # 降低 IOU 阈值，减少框抑制
                 target_class=None):  # 先不限制类别，后续看实际再调整
        self.model = YOLO(model_path)
        self.model.conf = conf_threshold
        self.model.iou = iou_threshold
        if target_class is not None:
            self.model.classes = [target_class]

    def detect(self, image):
        results = self.model(image, augment=True)  # 开启数据增强，提升检测
        detections = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
            detections.append({
                "x": x1, "y": y1, 
                "w": x2 - x1, "h": y2 - y1,
                "confidence": float(box.conf),
                "class_id": int(box.cls)
            })
        return detections