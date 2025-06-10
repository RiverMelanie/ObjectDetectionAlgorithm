import cv2
import os
import sys
import json
import numpy as np

# 项目根目录
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.append(project_root)

from src.detection.yolov5_detector import YOLOv5Detector
from src.process.enhance_image import enhance_image
from src.process.calibrate_detections import calibrate_detections
from src.process.confidence_compensate import confidence_compensate
from src.process.visualize_detections import visualize_detections

def process_img(img_path):

    # 读取图像
    img = cv2.imread(img_path)
    if img is None:
        print(f"无法读取图像: {img_path}")
        return []

    # 增强预处理
    img = enhance_image(img)

    # 初始化检测器
    detector = YOLOv5Detector(
        model_path='models/yolov5s.pt',
        conf_threshold=0.3,  # 降低阈值提高召回率
        iou_threshold=0.3,   # 降低IOU减少漏检
        target_class=32      # sports ball类别，形状和草菇类似
    )

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    complexity = np.sum(edges) / (img.shape[0] * img.shape[1])

    # 复杂度高（蘑菇小/多/重叠多）→ 用小尺度+高分辨率
    scales = [(640, 640), (960, 960), (1280, 1280)]
    if complexity > 100:  # 阈值可根据实际调
        scales = [(320, 320), (640, 640), (960, 960)]  # 加小尺度

    detections_list = []
    for scale in scales:
        scaled_img = cv2.resize(img, scale)
        dets = detector.detect(scaled_img)
        detections_list.append(dets)

    # 合并结果
    detections = merge_detections(detections_list, img.shape, scales)

    # 坐标校准，根据结果进行人为矫正
    calibrated = calibrate_detections(detections, img.shape)

    calibrated = confidence_compensate(calibrated, small_thresh=30, boost=0.1)

    # 实现目标检测中的非极大值抑制 (NMS) 算法 (包含 IoU 计算)。
    filtered = dynamic_nms(calibrated, img_shape=img.shape)

    result = [
        {k: v for k, v in det.items() if k in ["x", "y", "w", "h"]}
        for det in filtered
        if all(det.get(key, 0) != 0 for key in ["x", "y", "w", "h"])
    ]

    # 可视化结果
    visualize_detections(img_path, filtered)

    return result

def merge_detections(dets_list, img_shape, scales):
    """合并多尺度检测结果，优化尺度转换"""

    merged = []
    height, width = img_shape[:2]

    # 记录每个检测框的原始尺度，避免过度缩放
    for i, (scale, dets) in enumerate(zip(scales, dets_list)):
        for d in dets:
            # 保留原始尺度信息，用于后续判断
            d["original_scale"] = scale  
            d_scaled = {
                "x": int(d["x"] * (width / scale[0])),
                "y": int(d["y"] * (height / scale[1])),
                "w": int(d["w"] * (width / scale[0])),
                "h": int(d["h"] * (height / scale[1])),
                "confidence": d["confidence"],
                "class_id": d["class_id"],
                "original_scale": scale
            }
            merged.append(d_scaled)

    # 按置信度排序，优先保留高置信度+小尺度检测的框
    merged.sort(key=lambda x: (-x["confidence"], x["original_scale"][0]))

    # 去重（用动态 NMS，区分小蘑菇）
    keep = []
    for det in merged:
        w, h = det["w"], det["h"]
        iou_thresh = 0.3 if max(w, h) < 50 else 0.5  # 小蘑菇宽松去重
        overlap = False
        for kept in keep:
            if calculate_iou(det, kept) > iou_thresh:
                overlap = True
                break
        if not overlap:
            keep.append(det)

    return keep

def dynamic_nms(detections, img_shape):
    """
    detections: 检测结果列表（含 x,y,w,h,confidence）
    small_thresh: 判定小蘑菇的最大边长（像素）
    """
    if not detections:
        return []
    # 按置信度排序，同时保留小尺度检测结果
    detections.sort(key=lambda x: (-x["confidence"], x["original_scale"][0]))
    keep = []
        
    for det in detections:
        w, h = det["w"], det["h"]
        # 小蘑菇（边长<50）允许更高重叠
        iou_thresh = 0.3 if max(w, h) < 50 else 0.5  
            
        overlap = False
        for kept in keep:
            if calculate_iou(det, kept) > iou_thresh:
                overlap = True
                break
        if not overlap:
            keep.append(det)
    return keep

def calculate_iou(box1, box2):
    """实现目标检测中的交并比 (IoU) 计算"""
    x1 = max(box1["x"], box2["x"])
    y1 = max(box1["y"], box2["y"])
    x2 = min(box1["x"] + box1["w"], box2["x"] + box2["w"])
    y2 = min(box1["y"] + box1["h"], box2["y"] + box2["h"])
    if x2 < x1 or y2 < y1:
        return 0.0
    intersection = (x2 - x1) * (y2 - y1)
    area1 = box1["w"] * box1["h"]
    area2 = box2["w"] * box2["h"]
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0

if __name__ == "__main__":
    # 测试一张图像
    img_path = "./imgs/1748333053264.jpg"
    result = process_img(img_path)
    print(img_path,':',result)

    img_path2 = "./imgs/1748333920330.jpg"
    result2 = process_img(img_path2)
    print(img_path,':',result2)

    # 测试多张图像
    project_root2 = os.path.dirname(os.path.dirname(__file__))
    sys.path.append(project_root2)

    imgs_folder = os.path.join(project_root2, 'imgs/')
    img_paths = os.listdir(imgs_folder)

    for img_path in img_paths:
        result = process_img(imgs_folder+img_path)
        print(img_path,':',result)
