from time import time

import numpy as np

from .HandInput import HandInput


class FingerButton:
    def __init__(
        self,
        hand_input: HandInput,
        short_press_time: float = 0.1,
        long_press_time: float = 0.3,
    ) -> None:
        self.hand_input: HandInput = hand_input
        self.short_press_time: float = short_press_time
        self.long_press_time: float = long_press_time
        # 开始按下的时间戳
        self._start_time: None | float = None
        # 开始按下的位置
        self._start_point: None | np.ndarray = None
        # 用于记录短按的标志
        self._short_press_flag: bool = False
        # 按键状态标志,None表示没有按,False表示长按,True表示短按
        self._press_status: None | bool = None

    @property
    def is_long_press(self) -> bool:
        return self._press_status is False

    @property
    def is_short_press(self) -> bool:
        return self._press_status is True

    @property
    def is_press(self) -> bool:
        return self._press_status is not None

    @property
    def start_point(self) -> np.ndarray | None:
        return self._start_point

    def update(self, name: str, finger_touch_idx: int) -> None | bool:
        gst = self.hand_input.gestrue(name)
        bs = self.hand_input.base(name)
        # 判断该手部是否有被检测到
        if gst is None or bs is None:
            self._press_status = None
            return self._press_status
        # 检测手指指尖是否触碰
        if gst.get_fg_touch(finger_touch_idx):
            if self._start_time is None:
                self._start_time = time()
            else:
                # 记录按下的瞬间大拇指的相对于手腕的归一化坐标
                if self._start_point is None:
                    self._start_point = bs.wrist_npos(4)
                # 通过按下的时间来判断是长按还是短按
                press_time = time() - self._start_time
                if press_time > self.long_press_time:
                    self._short_press_flag = False
                    self._press_status = False  # 长按
                    return self._press_status
                elif press_time > self.short_press_time:
                    self._short_press_flag = True
        else:
            # 重置开始按下的时间戳和记录的点
            self._start_time = None
            self._start_point = None
            if self._short_press_flag:
                self._short_press_flag = False
                self._press_status = True  # 短按
                return self._press_status
            else:
                self._press_status = None  # 没有按下
                return self._press_status


class ThumbJoystick:
    def __init__(self, hand_input: HandInput) -> None:
        self.hand_input: HandInput = hand_input
        self.fixed_point: np.ndarray | None = None

    def get(self, name: str) -> np.ndarray | None:
        bs = self.hand_input.base(name)
        if bs is None or self.fixed_point is None:
            return None
        return bs.wrist_npos(4) - self.fixed_point

    def run(self, name: str, reset_flag: bool) -> np.ndarray | None:
        bs = self.hand_input.base(name)
        if bs is None:
            return None
        if reset_flag:
            self.fixed_point = bs.wrist_npos(4)
            return None
        if self.fixed_point is None:
            return None
        vec = bs.wrist_npos(4) - self.fixed_point
        return vec
        # return vec / np.linalg.norm(vec)
