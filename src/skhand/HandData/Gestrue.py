from time import time
import numpy as np
from .FingerModels import out_model, touch_model, close_model
from .OneHand import OneHand


class Gestrue:
    __slots__ = (
        "one_hand",
        "interval_time",
        "fom_output",
        "fom_interval_timer",
        "ftm_output",
        "ftm_interval_timer",
        "fcm_output",
        "fcm_interval_timer",
    )

    def __init__(
        self,
        one_hand: OneHand,
        interval_time: float = 0.3,
    ) -> None:
        """调用手势识别模型API
        Args:
            one_hand: 输入基本的手部数据实例
            interval_time: 调用模型的间隔时间
        """
        self.one_hand: OneHand = one_hand
        # 间隔调用参数
        self.interval_time: float = interval_time
        # 检测手指是否伸出
        self.fom_output: np.ndarray = np.zeros(5, dtype=np.bool_)
        self.fom_interval_timer: list[float] = [0 for _ in range(5)]
        # 检测指尖是否触碰
        self.ftm_output: np.ndarray = np.zeros(4, dtype=np.bool_)
        self.ftm_interval_timer: list[float] = [0 for _ in range(4)]
        # 检测手指是否合拢
        self.fcm_output: np.ndarray = np.zeros(4, dtype=np.bool_)
        self.fcm_interval_timer: list[float] = [0 for _ in range(4)]
        # self.fcm_interval_timer: np.ndarray = np.zeros(4, dtype=np.float32)

    @property
    def fg_all_out(self) -> np.ndarray:
        """检测是否所有手指都伸出"""
        for i in range(self.fom_output.shape[0]):
            self.get_fg_out(i)
        return self.fom_output

    def get_fg_out(self, idx: int) -> np.ndarray:
        """检测指定索引的手指是否伸出
        Args:
            idx: 0到4分别为大拇指,食指,中指,无名指,小拇指
        """
        if idx < 0 or idx > 4:
            raise ValueError(f"There is no finger with index {idx}")
        # 计算当前间隔时间
        cur_interval = time() - self.fom_interval_timer[idx]
        # 长间隔时间:模型输出结果没有变化
        if cur_interval > self.interval_time:
            self.fom_interval_timer[idx] = time()  # 更新计时器
            # 调用模型进行推理
            self.fom_output[idx] = out_model.run_by_idx(idx, self.one_hand.pos_data)
        return self.fom_output[idx]

    @property
    def thumb_out(self) -> np.ndarray:
        """检测大拇指是否伸出"""
        return self.get_fg_out(0)

    @property
    def index_fg_out(self) -> np.ndarray:
        """检测食指是否伸出"""
        return self.get_fg_out(1)

    @property
    def middle_fg_out(self) -> np.ndarray:
        """检测中指是否伸出"""
        return self.get_fg_out(2)

    @property
    def ring_fg_out(self) -> np.ndarray:
        """检测无名指是否伸出"""
        return self.get_fg_out(3)

    @property
    def pinky_out(self) -> np.ndarray:
        """检测小拇指是否伸出"""
        return self.get_fg_out(4)

    @property
    def fg_all_touch(self) -> np.ndarray:
        """检测是否所有手指指尖都触碰"""
        for i in range(self.ftm_output.shape[0]):
            self.get_fg_touch(i)
        return self.ftm_output

    def get_fg_touch(self, idx: int) -> np.ndarray:
        """检测指定索引的手指指尖是否触碰大拇指指尖
        Args:
            idx: 0到3分别为大拇指到食指/中指/无名指/小拇指指尖的索引
        """
        if idx < 0 or idx > 3:
            raise ValueError(f"There is no finger with index {idx}")
        # 计算当前间隔时间
        cur_interval = time() - self.ftm_interval_timer[idx]
        # 长间隔时间:模型输出结果没有变化
        if cur_interval > self.interval_time:
            self.ftm_interval_timer[idx] = time()  # 更新计时器
            # 调用模型进行推理
            self.ftm_output[idx] = touch_model.run_by_idx(
                idx, self.one_hand.finger_data
            )
        return self.ftm_output[idx]

    @property
    def index_fg_touch(self) -> np.ndarray:
        """检测食指指尖是否触碰大拇指指尖"""
        return self.get_fg_touch(0)

    @property
    def middle_fg_touch(self) -> np.ndarray:
        """检测中指指尖是否触碰大拇指指尖"""
        return self.get_fg_touch(1)

    @property
    def ring_fg_touch(self) -> np.ndarray:
        """检测无名指指尖是否触碰大拇指指尖"""
        return self.get_fg_touch(2)

    @property
    def pinky_touch(self) -> np.ndarray:
        """检测小拇指指尖是否触碰大拇指指尖"""
        return self.get_fg_touch(3)

    @property
    def fg_all_close(self) -> np.ndarray:
        """检测是否所有手指都合拢"""
        for i in range(self.fcm_output.shape[0]):
            self.get_fg_out(i)
        return self.fcm_output

    def get_fg_close(self, idx: int) -> np.ndarray:
        """检测指定索引的手指之间是否并拢
        Args:
            idx: 0到3分别为大拇指和食指并拢,食指和中指并拢...的索引
        """
        if idx < 0 or idx > 3:
            raise ValueError(f"There is no finger with index {idx}")
        # 计算当前间隔时间
        cur_interval = time() - self.fcm_interval_timer[idx]
        # 长间隔时间:模型输出结果没有变化
        if cur_interval > self.interval_time:
            self.fcm_interval_timer[idx] = time()  # 更新计时器
            # 调用模型进行推理
            self.fcm_output[idx] = close_model.run_by_idx(idx, self.one_hand.pos_data)
        return self.fcm_output[idx]

    @property
    def tb_if_close(self) -> np.ndarray:
        """检测大拇指和食指是否并拢"""
        return self.get_fg_close(0)

    @property
    def if_mf_close(self) -> np.ndarray:
        """检测食指和中指是否并拢"""
        return self.get_fg_close(1)

    @property
    def mf_rf_close(self) -> np.ndarray:
        """检测中指和无名指是否并拢"""
        return self.get_fg_close(2)

    @property
    def rf_pk_close(self) -> np.ndarray:
        """检测无名指和小拇指是否并拢"""
        return self.get_fg_close(3)
