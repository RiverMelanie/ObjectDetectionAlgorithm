import cv2
import numpy as np

def enhance_image(img):
    """增强预处理"""
    # 转换到HSV空间，增强蘑菇
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 增强饱和度通道，让蘑菇颜色更突出
    hsv[:, :, 1] = cv2.convertScaleAbs(hsv[:, :, 1], alpha=1.5, beta=0)  
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # 小目标超分辨率
    h, w = img.shape[:2]
    if max(h, w) < 400:  # 判定为小图，触发超分
        img = cv2.resize(img, (w*2, h*2), interpolation=cv2.INTER_CUBIC)

    # 局部对比度增强（针对蘑菇区域）
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab_planes = list(cv2.split(lab))
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return img