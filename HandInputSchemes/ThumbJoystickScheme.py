import numpy as np

from .HandInputScheme import HandInputScheme
from .FingertipButtonScheme import FingertipButtonScheme
from ..HandInput import HandInput


class ThumbJoystickScheme(HandInputScheme):
    __slots__ = "hand_input", "hand_name", "finger_btn", "_fixed_point", "_activate"

    def __init__(
        self, hand_input: HandInput, hand_name: str, finger_btn: FingertipButtonScheme
    ) -> None:
        """拇指摇杆操控方案
        Args:
            hand_input: 手部输入类的实例
            hand_name: 手部名字,指定为哪只手制定手部操控方案
            finger_btn: 一个用于长按设置定点的指尖按钮实例
        """
        hand_input.hands_dict[hand_name]  # 获取该手部,没有该手部名字则报错
        self.hand_input: HandInput = hand_input
        self.hand_name: str = hand_name
        # 需要使用长按来设置定点,所以要传入一个指尖按钮实例
        self.finger_btn: FingertipButtonScheme = finger_btn
        # 创建一个变量来存储定点位置,初始值为None,表示还没有定点
        self._fixed_point: np.ndarray = np.zeros((1, 3))
        # 定义一个变量用于区分,摇杆是否激活,默认为False未激活
        self._activate: bool = False

    def update(self) -> None:
        """实时更新定点的位置,直到长按才定下来"""
        # 获取并判断是否有检测到手部
        base = self.hand_input.base(self.hand_name)
        if base is None:  # 没有检测到改手部,则不激活摇杆
            self._activate = False
            return
        # 判断是否激活摇杆
        if self.finger_btn.is_long_press:  # 持续长按,则继续激活摇杆
            self._activate = True
        else:  # 长按结束,则关闭摇杆,即不激活
            self._activate = False
        # 摇杆未激活,则定点随拇指指尖移动
        if not self._activate:
            self._fixed_point = base.wrist_npos(4)  # 定点随拇指移动

    @property
    def is_activate(self) -> bool:
        """摇杆是否激活或是否启用,激活为True"""
        return self._activate

    @property
    def fixed_point(self) -> np.ndarray | None:
        """返回设置的定点位置,没长按时定点位置随拇指移动,若未设置定点则返回None"""
        return self._fixed_point if self._activate else None

    @property
    def end_point(self) -> np.ndarray | None:
        """返回摇杆的终点坐标,是相对于手腕的归一化后坐标值,没有设定点则返回None"""
        base = self.hand_input.base(self.hand_name)
        if base is not None and self._activate:  # 手部存在且摇杆已激活
            return base.wrist_npos(4)  # 终点坐标就是当前手部的拇指坐标

    @property
    def vector(self) -> np.ndarray | None:
        """返回拇指摇杆的向量(未归一化),即当前拇指的位置和定点之差"""
        end_p = self.end_point
        return (end_p - self._fixed_point) if end_p is not None else None

    @property
    def norm_vec(self) -> np.ndarray | None:
        """返回拇指摇杆的方向向量,即归一化的向量"""
        vec = self.vector
        if vec is None:
            return None
        vec_len = np.linalg.norm(vec)
        vec_len = vec_len if vec_len > 0 else 1
        return vec / vec_len
