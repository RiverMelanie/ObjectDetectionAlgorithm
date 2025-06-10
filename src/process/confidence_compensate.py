import cv2
import os
import sys
import json
import numpy as np


def confidence_compensate(detections, small_thresh=30, boost=0.1):
    """
    detections: 检测结果列表
    small_thresh: 小蘑菇判定阈值
    boost: 小蘑菇置信度提升量
    """
    compensated = []
    for det in detections:
        w, h = det["w"], det["h"]
        conf = det["confidence"]
        # 小蘑菇置信度补偿
        if max(w, h) < small_thresh:
            conf = min(conf + boost, 1.0)  # 不超过 1.0
        # 大蘑菇置信度校验
        else:
            conf = max(conf, 0.3)  # 大蘑菇最低置信度限制
        det["confidence"] = conf
        compensated.append(det)
    return compensated