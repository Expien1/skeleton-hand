from time import time

import numpy as np

from .HandInputScheme import HandInputScheme
from ..HandInput import HandInput


class FingerSwipeScheme(HandInputScheme):
    __slots__ = (
        "hand_input",
        "hand_name",
        "point_id",
        "swipe_velocity",
        "follow_velocity",
        "min_swipe_dist",
        "reset_time",
        "_follow_point",
        "_start_point",
        "_end_point",
        "_swipe_vec",
        "_swipe_dist",
        "_start_time",
    )

    def __init__(
        self,
        hand_input: HandInput,
        hand_name: str,
        point_id: int = 8,
        swipe_velocity: float = 3,
        follow_velocity: float = 1,
        min_swipe_dist: float = 0.3,
        reset_time: float = 2,
    ) -> None:
        """
        Args:
            hand_input: 手部输入类的实例
            hand_name: 手部名字,指定为哪只手制定手部操控方案
            point_id: 设置关键点索引作为滑动点,最好选指尖的点,默认为8(食指指尖)
            swipe_velocity: 手指滑动阈值,超过该阈值才设置跟随点,默认为3
            follow_velocity: 跟随点的速度,默认为1
            min_swipe_dist: 最小的滑动距离,超过该距离才看作滑动,默认为0.3
            reset_time: 重置跟随点的时间,超过该时间没有滑动则重置,默认为2秒
        """
        hand_input.hands_dict[hand_name]  # 获取该手部,没有该手部名字则报错
        self.hand_input: HandInput = hand_input
        self.hand_name: str = hand_name
        self.point_id: int = point_id
        self.swipe_velocity: float = swipe_velocity
        self.follow_velocity: float = follow_velocity
        self.min_swipe_dist: float = min_swipe_dist
        self.reset_time: float = reset_time
        # 创建跟随点,没有跟随设为None
        self._follow_point: np.ndarray | None = None
        # 记录滑动开始位置
        self._start_point: np.ndarray | None = None
        # 记录滑动结束位置
        self._end_point: np.ndarray = np.empty((1, 2))
        # 记录滑动向量
        self._swipe_vec: np.ndarray = np.empty((1, 2))
        # 记录滑动距离
        self._swipe_dist = 0
        # 记录开始跟随的时间,用于判断跟随点是否超时
        self._start_time = time()

    def _reset(self) -> None:
        """重置跟随点和计时器"""
        self._follow_point = None  # 重置跟随点为初始状态
        self._start_time = time()  # 重置计时器

    def update(self) -> None:
        """实时更新是否有滑动"""
        base = self.hand_input.base(self.hand_name)
        # 计算时间差查看是否超时
        if base is None or (time() - self._start_time) > self.reset_time:
            self._reset()  # 超时或没有检测到手部,则重置跟随点
            self._start_point = None  # 结束/关闭滑动的输出
            return None
        # 初始化跟随点
        if self._follow_point is None:
            self._follow_point = base.wrist_npos(self.point_id)[:2]
        # 更新跟随点的位置,跟随当前点的位置
        cur_point = base.wrist_npos(self.point_id)[:2]
        self._follow_point += (  # 更新跟随点位置
            (cur_point - self._follow_point)  # 与当前点位置距离差
            * self.follow_velocity  # 跟随点的速度
            * self.hand_input.delta_time  # 帧间间隔时间
        )
        # 超过速度阈值和跟随点距离阈值才返回输出
        if base.wrist_npos_velocity(self.point_id) > self.swipe_velocity:
            swipe_vec = cur_point - self._follow_point  # 计算滑动向量
            swipe_dist = np.linalg.norm(swipe_vec)  # 计算滑动距离
            # 检测滑动距离是否满足要求
            if swipe_dist > self.min_swipe_dist:
                self._start_point = self._follow_point.copy()  # 记录输出结果
                self._end_point = cur_point
                self._swipe_vec = swipe_vec
                self._swipe_dist = swipe_dist
                self._reset()  # 重置跟随点
                return None
        # 否则更新跟随点
        self._follow_point = base.wrist_npos(self.point_id)[:2]
        self._start_point = None  # 结束/关闭滑动的输出
        return None

    @property
    def is_activate(self) -> bool:
        """是否有滑动,有滑动则为True"""
        return self._start_point is not None

    @property
    def start_point(self) -> np.ndarray | None:
        """滑动开始位置坐标,如果没有滑动则返回None"""
        return self._start_point

    @property
    def end_point(self) -> np.ndarray | None:
        """滑动末尾位置坐标,如果没有滑动则返回None"""
        return self._end_point if self.is_activate else None

    @property
    def vector(self) -> np.ndarray | None:
        """返回滑动向量,如果没有滑动则返回None"""
        return self._swipe_vec if self.is_activate else None

    @property
    def distance(self) -> np.floating | float | None:
        """返回滑动的距离,如果没有滑动则返回None"""
        return self._swipe_dist if self.is_activate else None

    @property
    def norm_vec(self) -> np.ndarray | None:
        """返回滑动的方向向量,如果没有滑动则返回None"""
        return (self._swipe_vec / self._swipe_dist) if self.is_activate else None
