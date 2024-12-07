from abc import ABC, abstractmethod

import numpy as np

from ..HandData.OneHand import OneHand


class VisualHandDetector(ABC):
    """基于视觉的手部关键点检测器"""

    @abstractmethod
    def __init__(self, max_num_hands: int):
        # 创建手部关键点检测器
        self._detector = None

    @abstractmethod
    def detect(self, image: np.ndarray, hands_dict: dict[str, OneHand]) -> list[str]:
        """使用视觉检测手部关键点"""
        detected_name_ls = []
        # 最后返回检测到的手部的名字
        return detected_name_ls
