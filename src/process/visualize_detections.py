import cv2
import os
import numpy as np


def visualize_detections(img_path, detections, save_dir="visualization"):
    """
    可视化检测框，辅助调试漏检问题
    """
    os.makedirs(save_dir, exist_ok=True)
    img = cv2.imread(img_path)
    for det in detections:
        x, y, w, h = det["x"], det["y"], det["w"], det["h"]
        # 绘制检测框
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # 标记原始尺度（辅助分析多尺度效果）
        scale_info = f"Scale: {det.get('original_scale', (0,0))}"
        cv2.putText(img, scale_info, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    save_path = os.path.join(save_dir, os.path.basename(img_path))
    cv2.imwrite(save_path, img)