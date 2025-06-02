from time import time

import numpy as np

from .HandInputScheme import HandInputScheme
from ..HandInput import HandInput


class FingertipButtonScheme(HandInputScheme):
    __slots__ = (
        "hand_input",
        "hand_name",
        "finger_touch_idx",
        "short_press_time",
        "long_press_time",
        "_start_time",
        "_start_point",
        "_short_press_flag",
        "_press_status",
    )

    def __init__(
        self,
        hand_input: HandInput,
        hand_name: str,
        finger_touch_idx: int = 0,
        short_press_time: float = 0.1,
        long_press_time: float = 0.3,
    ) -> None:
        """指尖按钮操控方案
        Args:
            hand_input: 手部输入类的实例
            hand_name: 手部名字,指定为哪只手制定手部操控方案
            finger_touch_idx: 指尖触碰状态模型的索引,指定使用哪个指尖作为指尖按钮,默认为0(即食指触碰)
            short_press_time: 短按时间阈值,即短按的触发时间,默认为0.1秒
            long_press_time: 长按时间阈值,即长按的触发时间,默认为0.3秒
        """
        hand_input.hands_dict[hand_name]  # 获取该手部,没有该手部名字则报错
        self.hand_input: HandInput = hand_input
        self.hand_name: str = hand_name
        # 指尖按钮操控方案相关参数
        self.finger_touch_idx: int = finger_touch_idx
        self.short_press_time: float = short_press_time
        self.long_press_time: float = long_press_time
        # 开始按下的时间戳
        self._start_time: None | float = None
        # 开始按下的位置
        self._start_point: None | np.ndarray = None
        # 用于记录是否短按的标志
        self._short_press_flag: bool = False
        # 按键状态标志,None表示没有按,False表示长按,True表示短按
        self._press_status: None | bool = None

    def update(self) -> None | bool:
        """实时更新指尖按钮计时判断"""
        # 获取并判断该手部数据是否有被检测到
        hand = self.hand_input.hand(self.hand_name)
        if hand is None:  # 没有检测到对应的手部
            self._press_status = None  # 将按钮状态重置为没有按下
            return None
        # 检测手指指尖是否触碰
        if hand.gestrue.get_fg_touch(self.finger_touch_idx):
            if self._start_time is None:  # 记录开始触碰的时间戳
                self._start_time = time()
            else:  # 记录按下的瞬间大拇指的相对于手腕的归一化坐标
                if self._start_point is None:
                    self._start_point = hand.base.wrist_npos(4)
                # 通过按下的时间来判断是长按还是短按
                press_time = time() - self._start_time
                if press_time > self.long_press_time:  # 判断长按
                    self._short_press_flag = False  # 超过短按时间,不是短按
                    self._press_status = False  # 返回长按
                    return self._press_status
                elif press_time > self.short_press_time:  # 判断短按
                    self._short_press_flag = True  # 记录短按标志
        else:  # 重置开始按下的时间戳和记录的点
            self._start_time = None
            self._start_point = None
            if self._short_press_flag:  # 判断是否确定短按
                self._short_press_flag = False  # 重置短按标志
                self._press_status = True  # 短按
                return self._press_status
            else:
                self._press_status = None  # 没有按下
                return self._press_status

    @property
    def is_activate(self) -> bool:
        """指尖按钮是否有激活,True为激活,False为没有激活"""
        return self._press_status is not None

    @property
    def is_long_press(self) -> bool:
        """是否为长按,True为长按,False为短按或没有按下"""
        return self._press_status is False

    @property
    def is_short_press(self) -> bool:
        """是否为短按,True为短按,False为长按或没有按下"""
        return self._press_status is True

    @property
    def is_press(self) -> bool:
        """是否有按下,True为按下,False为没有按下"""
        return self._press_status is not None

    @property
    def start_point(self) -> np.ndarray | None:
        """返回最开始按下按钮的位置"""
        return self._start_point
