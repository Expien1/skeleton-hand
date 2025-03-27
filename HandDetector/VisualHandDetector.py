from abc import ABC, abstractmethod

import numpy as np

from .HandsMatchers import HandsMatcher
from ..HandData.OneHand import OneHand


class VisualHandDetector(ABC):
    """基于视觉的手部关键点检测器"""

    __slots__ = "hands_name_ls", "_detector", "hands_matcher"

    @abstractmethod
    def __init__(self, hands_name_ls: list[str], hands_matcher: type[HandsMatcher]):
        """
        Args:
            hands_name_ls: 需要被检测到的手部名字列表
            hands_matcher: 创建手部匹配器
        """
        # 创建手部关键点检测器
        self._detector = None
        self.hands_name_ls: list[str] = hands_name_ls
        self.hands_matcher: HandsMatcher = hands_matcher(hands_name_ls)

    @abstractmethod
    def detect(self, image: np.ndarray, hands_dict: dict[str, OneHand]) -> list[str]:
        """使用视觉检测手部关键点,返回成功检测到的手部的名称
        Args:
            image: ndarray格式的被检测图片
            hands_dict: 手部名字和手部数据实例的字典,名字为键,数据实例为值
        """
        detected_name_ls = []
        # 最后返回检测到的手部的名字
        return detected_name_ls
