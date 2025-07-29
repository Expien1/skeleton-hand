from time import time

import numpy as np

from .Models import finger_out_model, finger_touch_model, finger_close_model
from .OneHand import OneHand


class Gestrue:
    __slots__ = (
        "one_hand",
        "interval_time",
        "_fg_out_output",
        "_fout_last_time",
        "_fg_touch_output",
        "_ftouch_last_time",
        "_fg_close_output",
        "_fclose_last_time",
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
        # 手指伸出状态输出状态和计时器
        self._fg_out_output: np.ndarray = np.zeros(
            len(finger_out_model), dtype=np.bool_
        )
        self._fout_last_time: np.ndarray = np.zeros(
            len(finger_out_model), dtype=np.float64
        )
        # 手指指尖触碰状态输出状态和计时器
        self._fg_touch_output: np.ndarray = np.zeros(
            len(finger_touch_model), dtype=np.bool_
        )
        self._ftouch_last_time: np.ndarray = np.zeros(
            len(finger_touch_model), dtype=np.float64
        )
        # 手指并拢状态输出状态和计时器
        self._fg_close_output: np.ndarray = np.zeros(
            len(finger_close_model), dtype=np.bool_
        )
        self._fclose_last_time: np.ndarray = np.zeros(
            len(finger_close_model), dtype=np.float64
        )

    def __repr__(self) -> str:
        return (
            "skhand.HandData.Gestrue("
            f"\n\tinterval_time={self.interval_time},"
            f"\n\t{repr(finger_out_model)},"
            f"\n\t{repr(finger_touch_model)},"
            f"\n\t{repr(finger_close_model)},"
            "\n)"
        )

    def __str__(self) -> str:
        return (
            "Gestrue("
            f"\n\tinterval_time={self.interval_time},"
            f"\n\t{str(finger_out_model)},"
            f"\n\t{str(finger_touch_model)},"
            f"\n\t{str(finger_close_model)},"
            "\n)"
        )

    def _update_fout_output(self, index: int | None = None):
        """更新手指伸出状态输出"""
        if index is not None:
            if index < 0 or index > len(finger_out_model):
                raise ValueError(f"Can't find '{index}' model output")
            cur_interval = time() - self._fout_last_time[index]
            # 超过当前间隔时间,才调用模型更新输出数据
            if cur_interval > self.interval_time:
                self._fout_last_time[index] = time()  # 更新计时器
                self._fg_out_output[index] = finger_out_model[index].run(
                    self.one_hand.pos_data
                )
        else:
            cur_interval = time() - self._fout_last_time
            mask = cur_interval > self.interval_time  # 记录超过间隔时间的布尔索引
            # 重置超过间隔时间的计时器
            self._fout_last_time = np.where(mask, time(), self._fout_last_time)
            # 更新对应的输出数据
            self._fg_out_output[mask] = [
                finger_out_model[i].run(self.one_hand.pos_data)[0]
                for i in np.arange(len(finger_out_model))[mask]
            ]

    @property
    def finger_out(self) -> np.ndarray:
        """检测手指是否伸出,并用索引访问对应输出数据,finger_out[idx]
        idx: 0到4分别为大拇指,食指,中指,无名指,小拇指
        """
        self._update_fout_output()
        return self._fg_out_output

    @property
    def fg_all_out(self) -> np.ndarray:
        """检测是否所有手指都伸出"""
        self._update_fout_output()
        return self._fg_out_output.copy()

    @property
    def thumb_out(self) -> np.ndarray:
        """检测大拇指是否伸出"""
        self._update_fout_output(0)
        return self._fg_out_output[0]

    @property
    def index_fg_out(self) -> np.ndarray:
        """检测食指是否伸出"""
        self._update_fout_output(1)
        return self._fg_out_output[1]

    @property
    def middle_fg_out(self) -> np.ndarray:
        """检测中指是否伸出"""
        self._update_fout_output(2)
        return self._fg_out_output[2]

    @property
    def ring_fg_out(self) -> np.ndarray:
        """检测无名指是否伸出"""
        self._update_fout_output(3)
        return self._fg_out_output[3]

    @property
    def pinky_out(self) -> np.ndarray:
        """检测小拇指是否伸出"""
        self._update_fout_output(4)
        return self._fg_out_output[4]

    def _update_ftouch_output(self, index: int | None = None):
        """更新手指伸出状态输出"""
        if index is not None:
            if index < 0 or index > len(finger_touch_model):
                raise ValueError(f"Can't find '{index}' model output")
            cur_interval = time() - self._ftouch_last_time[index]
            # 超过当前间隔时间,才调用模型更新输出数据
            if cur_interval > self.interval_time:
                self._ftouch_last_time[index] = time()  # 更新计时器
                self._fg_touch_output[index] = finger_touch_model[index].run(
                    self.one_hand.finger_data
                )
        else:
            cur_interval = time() - self._ftouch_last_time
            mask = cur_interval > self.interval_time  # 记录超过间隔时间的布尔索引
            # 重置超过间隔时间的计时器
            self._ftouch_last_time = np.where(mask, time(), self._ftouch_last_time)
            # 更新对应的输出数据
            self._fg_touch_output[mask] = [
                finger_touch_model[i].run(self.one_hand.finger_data)[0]
                for i in np.arange(len(finger_touch_model))[mask]
            ]

    @property
    def finger_touch(self) -> np.ndarray:
        """检测手指指尖是否触碰大拇指指尖,finger_touch[idx]
        idx: 0到3分别为大拇指到食指/中指/无名指/小拇指指尖的索引
        """
        self._update_ftouch_output()
        return self._fg_touch_output

    @property
    def fg_all_touch(self) -> np.ndarray:
        """检测是否所有手指指尖都触碰"""
        self._update_ftouch_output()
        return self._fg_touch_output.copy()

    @property
    def index_fg_touch(self) -> np.ndarray:
        """检测食指指尖是否触碰大拇指指尖"""
        self._update_ftouch_output(0)
        return self._fg_touch_output[0]

    @property
    def middle_fg_touch(self) -> np.ndarray:
        """检测中指指尖是否触碰大拇指指尖"""
        self._update_ftouch_output(1)
        return self._fg_touch_output[1]

    @property
    def ring_fg_touch(self) -> np.ndarray:
        """检测无名指指尖是否触碰大拇指指尖"""
        self._update_ftouch_output(2)
        return self._fg_touch_output[2]

    @property
    def pinky_touch(self) -> np.ndarray:
        """检测小拇指指尖是否触碰大拇指指尖"""
        self._update_ftouch_output(3)
        return self._fg_touch_output[3]

    def _update_fclose_output(self, index: int | None = None):
        """更新手指伸出状态输出"""
        if index is not None:
            if index < 0 or index > len(finger_close_model):
                raise ValueError(f"Can't find '{index}' model output")
            cur_interval = time() - self._fclose_last_time[index]
            # 超过当前间隔时间,才调用模型更新输出数据
            if cur_interval > self.interval_time:
                self._fclose_last_time[index] = time()  # 更新计时器
                self._fg_close_output[index] = finger_close_model[index].run(
                    self.one_hand.pos_data
                )
        else:
            cur_interval = time() - self._fclose_last_time
            mask = cur_interval > self.interval_time  # 记录超过间隔时间的布尔索引
            # 重置超过间隔时间的计时器
            self._fclose_last_time = np.where(mask, time(), self._fclose_last_time)
            # 更新对应的输出数据
            self._fg_close_output[mask] = [
                finger_close_model[i].run(self.one_hand.pos_data)[0]
                for i in np.arange(len(finger_close_model))[mask]
            ]

    @property
    def finger_close(self) -> np.ndarray:
        """检测手指之间是否并拢,finger_close[idx]
        idx: 0到3分别为大拇指和食指并拢,食指和中指并拢...的索引
        """
        self._update_fclose_output()
        return self._fg_close_output

    @property
    def fg_all_close(self) -> np.ndarray:
        """检测是否所有手指都合拢"""
        self._update_fclose_output()
        return self._fg_close_output.copy()

    @property
    def tb_if_close(self) -> np.ndarray:
        """检测大拇指和食指是否并拢"""
        self._update_fclose_output(0)
        return self._fg_close_output[0]

    @property
    def if_mf_close(self) -> np.ndarray:
        """检测食指和中指是否并拢"""
        self._update_fclose_output(1)
        return self._fg_close_output[1]

    @property
    def mf_rf_close(self) -> np.ndarray:
        """检测中指和无名指是否并拢"""
        self._update_fclose_output(2)
        return self._fg_close_output[2]

    @property
    def rf_pk_close(self) -> np.ndarray:
        """检测无名指和小拇指是否并拢"""
        self._update_fclose_output(3)
        return self._fg_close_output[3]
