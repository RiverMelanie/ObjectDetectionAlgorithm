from abc import ABC, abstractmethod

class BaseDetector(ABC):
    """备用检测"""
    @abstractmethod
    def detect(self, image):
        """执行物体检测，返回检测结果列表"""
        pass    
