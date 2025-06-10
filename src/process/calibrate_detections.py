import cv2
import os
import sys
import json
import numpy as np

def calibrate_detections(detections, img_shape):
    """精确校准边界框"""
    height, width = img_shape[:2]

    # for det in detections:
    #     # 1. 修正x坐标
    #     x_ = int(det["w"] * 0.14)
    #     det["x"] = max(0, det["x"] + x_)

    #     # 2. 修正y坐标
    #     y_shift = int(det["h"] * 0.12)
    #     det["y"] = max(0, det["y"] - y_shift)

    #     # 3. 修正宽度
    #     w_expand = int(det["w"] * 0.24)
    #     det["w"] = min(width - det["x"], det["w"] + w_expand)
            
    #     # 4. 修正高度
    #     h_expand = int(det["h"] * 0.05)
    #     det["h"] = min(height - det["y"], det["h"] + h_expand)

    return detections