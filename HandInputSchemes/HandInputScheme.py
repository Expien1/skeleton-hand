from abc import ABC, abstractmethod
from typing import Any


class HandInputScheme(ABC):
    __slots__ = "hand_input", "hand_name"

    @abstractmethod
    def __init__(self, hand_input: "HandInput", hand_name: str) -> None:
        """手部操控方案抽象类,通过调用手部数据自定义通用的手部操控方案
        Args:
            hand_input: 手部输入类的实例
            hand_name: 手部名字,指定为哪只手制定手部操控方案
        """
        hand_input.hands_dict[hand_name]  # 获取该手部,没有该手部名字则报错
        self.hand_input = hand_input
        self.hand_name: str = hand_name

    @abstractmethod
    def update(self) -> Any:
        """实时更新手部操控方案,外部通过调用其他类属性或方法来使用手部操控方案"""
        pass

    @property
    @abstractmethod
    def is_activate(self) -> bool:
        """检查该手部操控方案是否有激活,激活则返回True"""
        return True
