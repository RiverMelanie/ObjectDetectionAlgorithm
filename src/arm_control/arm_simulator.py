import time
import random
import cv2
import numpy as np
import json
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class ArmSimulator:
    def __init__(self):
        """初始化机械臂模拟器"""
        # 设置机械臂参数
        self.base_position = (0, 0, 0)  # 机械臂基座位置
        self.current_position = self.base_position
        self.speed = 100  # 移动速度
        self.gripper_state = "open"     # 夹爪状态: "open" 或 "closed"
        
        # 设置工作区域限制
        self.workspace_limits = {
            'x_min': -200, 'x_max': 200,
            'y_min': 0, 'y_max': 300,
            'z_min': 0, 'z_max': 200
        }

    def image_to_world_coordinates(x, y, w, h, image_width, image_height, workspace_limits):
        """将图像坐标转换为机械臂坐标"""
        # 图像中心
        image_center_x = image_width / 2
        image_center_y = image_height / 2
        
        # 归一化坐标(范围从-1到1)
        norm_x = (x + w/2 - image_center_x) / image_center_x
        norm_y = (y + h/2 - image_center_y) / image_center_y
        
        # 工作区域尺寸
        workspace_width = workspace_limits['x_max'] - workspace_limits['x_min']
        workspace_height = workspace_limits['y_max'] - workspace_limits['y_min']
        
        # 转换为世界坐标
        world_x = norm_x * (workspace_width / 2)
        world_y = norm_y * (workspace_height / 2) + workspace_limits['y_min'] + workspace_height / 2
        
        # 设置固定的Z高度(物体上方)
        world_z = 100
        
        return world_x, world_y, world_z
    
    def move_to(self, x, y, z, wait=True):
        """移动机械臂到指定位置"""
        # 检查目标位置是否在工作区域内
        if not self._is_position_valid(x, y, z):
            print(f"警告: 目标位置({x}, {y}, {z})超出工作区域，忽略该指令")
            return False
        
        # 计算移动距离和估计时间
        dx = x - self.current_position[0]
        dy = y - self.current_position[1]
        dz = z - self.current_position[2]
        distance = (dx**2 + dy**2 + dz**2) ** 0.5
        estimated_time = distance / self.speed
        
        # 模拟移动过程
        print(f"机械臂从{self.current_position}移动到({x}, {y}, {z})")
        self.current_position = (x, y, z)
        
        # 如果需要等待移动完成
        if wait:
            time.sleep(estimated_time * 0.1)  # 加速模拟，实际时间*0.1
            print(f"机械臂已到达位置({x}, {y}, {z})")
        
        return True
    
    def grasp(self, x, y, z, object_height=50):
        """执行抓取操作"""
        print(f"准备抓取位置({x}, {y}, {z})的物体")
        
        # 移动到物体上方
        self.move_to(x, y, z + 50)
        
        # 下降到抓取高度
        self.move_to(x, y, z)
        
        # 闭合夹爪
        self._close_gripper()
        
        # 提升物体
        self.move_to(x, y, z + 100)
        
        print(f"已成功抓取位置({x}, {y}, {z})的物体")
        return True
    
    def place(self, x, y, z):
        """执行放置操作"""
        print(f"准备放置物体到位置({x}, {y}, {z})")
        
        # 移动到放置位置上方
        self.move_to(x, y, z + 50)
        
        # 下降到放置高度
        self.move_to(x, y, z)
        
        # 打开夹爪
        self._open_gripper()
        
        # 提升机械臂
        self.move_to(x, y, z + 50)
        
        print(f"已成功放置物体到位置({x}, {y}, {z})")
        return True
    
    def return_to_base(self):
        """返回初始位置"""
        print("机械臂返回初始位置")
        self.move_to(*self.base_position)
        return True
    
    def _open_gripper(self):
        """打开夹爪"""
        print("夹爪已打开")
        self.gripper_state = "open"
        # 模拟操作时间
        time.sleep(0.2)
        return True
    
    def _close_gripper(self):
        """闭合夹爪"""
        print("夹爪已闭合")
        self.gripper_state = "closed"
        # 模拟操作时间
        time.sleep(0.2)
        return True
    
    def _is_position_valid(self, x, y, z):
        """检查位置是否在工作区域内"""
        limits = self.workspace_limits
        return (limits['x_min'] <= x <= limits['x_max'] and
                limits['y_min'] <= y <= limits['y_max'] and
                limits['z_min'] <= z <= limits['z_max'])
    
    def get_response(self):
        """获取机械臂操作完成后的响应"""
        responses = [
            "操作已完成，等待下一步指令",
            "任务执行成功，准备下一个任务",
            "已完成指定操作，系统就绪",
            "机械臂操作完毕，请指示"
        ]
        return random.choice(responses)