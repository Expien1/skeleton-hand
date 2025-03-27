from typing import Literal
import numpy as np

from ..HandUtils.Filter import AdaptiveEWMAFilter
from ._global import T_NORM2WRIST, T_NORM2VEC


class OneHand:
    __slots__ = (
        "hand_side",
        "raw_pos",
        "last_rpos",
        "_norm_pos",
        "_box",
        "_normalized_flag",
        "_wrist_npos",
        "_wrist_npos_flag",
        "_fingers_angle",
        "_angle_flag",
        "_thumb_dist",
        "_dist_flag",
        "_data",
        "_data_flag",
        "_pos_data",
        "_pos_data_flag",
        "_finger_data",
        "_finger_data_flag",
        "_wrist_npos_diff",
        "last_wnpos",
        "wrist_npos_diff_filter",
        "_wrist_npos_diff_flag",
        "_angle_diff",
        "_last_angle",
        "_angle_diff_flag",
    )

    def __init__(self):
        """一只手的关键点位置数据"""
        # 用记录当前手是左手还是右手,默认为未知
        self.hand_side: Literal["left", "right", "Unknown"] = "Unknown"
        # 定义一个21x3的全0的二维数组来接收返回的21个手部关键点的xyz坐标
        # raw_pos表示这是原始传入的手部关键点的位置,初始创建一个空的数组,里面初始值是脏数据
        self.raw_pos: np.ndarray = np.zeros((21, 3), dtype=np.float32)
        self.last_rpos: np.ndarray = self.raw_pos.copy()
        # norm_pos表示是再手部box里归一化之后的坐标点坐标
        self._norm_pos: np.ndarray = np.zeros((21, 3), dtype=np.float32)
        # 手部矩形框的四个坐标点,分别为左上角坐标和右下角坐标
        self._box: np.ndarray = np.zeros(4, dtype=np.int32)
        # 标记是否已经计算好归一化和矩形框的四个坐标点
        self._normalized_flag: bool = False
        # _wrist_npos表示以手腕为原点的归一化坐标
        self._wrist_npos: np.ndarray = np.zeros((21, 3), dtype=np.float32)
        # 标记是否已经计算好以手腕为原点的归一化坐标
        self._wrist_npos_flag: bool = False
        # 定义一个5x3的二维数组用来存储5根手指的3个关节点的弯曲角度
        self._fingers_angle: np.ndarray = np.zeros((5, 3), dtype=np.float32)
        # 标记是否计算好关键点的弯曲角度
        self._angle_flag: bool = False
        # 定义一个一维数组来存储大拇指与其他关键点的距离
        self._thumb_dist: np.ndarray = np.zeros(4, dtype=np.float32)
        # 标志是否计算好4个指尖与大拇指的距离
        self._dist_flag: bool = False
        # 创建一个一维数组用于收集所有的一维手部数据
        self._data: np.ndarray = self._norm_pos.copy()
        # 标志当前数据是否为最新的数据
        self._data_flag: bool = False
        # 创建一个一维数组用于收集关键点的一维手部数据
        self._pos_data: np.ndarray = self._norm_pos.ravel()
        # 标志当前数据是否为最新的数据
        self._pos_data_flag: bool = False
        # 创建一个一维数组用于收集角度和指尖距离的一维手部数据
        self._finger_data: np.ndarray = self._fingers_angle.ravel()
        # 标志当前数据是否为最新的数据
        self._finger_data_flag: bool = False
        # 创建一个21x3的二维数组来存储每个关键点的移动差值
        self._wrist_npos_diff: np.ndarray = np.zeros(21, dtype=np.float32)
        self.last_wnpos: np.ndarray = self._wrist_npos.copy()
        # 创建滤波器对关键点两帧间的位移进行滤波
        self.wrist_npos_diff_filter: AdaptiveEWMAFilter = AdaptiveEWMAFilter(0.08, 0.5)
        # 标志当前_wrist_npos_diff变量存的是否为最新的速度
        self._wrist_npos_diff_flag: bool = False
        # 定义一个5x3的二维数组用来存储5根手指的3个关节点的弯曲角度的差值
        self._angle_diff: np.ndarray = self._fingers_angle.copy()
        self._last_angle: np.ndarray = self._fingers_angle.copy()
        # 标志当前_angle_diff变量存的是否为最新的速度
        self._angle_diff_flag: bool = False

    def reset_all_flags(self):
        """重置所有数据的更新标志为False"""
        self._normalized_flag = False
        self._wrist_npos_flag = False
        self._angle_flag = False
        self._dist_flag = False
        self._data_flag = False
        self._finger_data_flag = False
        self._wrist_npos_diff_flag = False
        self._angle_diff_flag = False

    @property
    def norm_pos(self) -> np.ndarray:
        """返回以手部矩形框左上角为原点的归一化关键点坐标"""
        if not self._normalized_flag:
            return self.normalization()
        return self._norm_pos

    def normalization(self) -> np.ndarray:
        """将传入的初始手部关键点坐标进行归一化为手部框内的相对位置"""
        # 计算得到手部矩形框的四个顶点坐标
        min_arr = self.raw_pos.min(axis=0)
        max_arr = self.raw_pos.max(axis=0)
        # 计算顶点的范围,最大减最小
        min_x, min_y, _ = min_arr
        max_x, max_y, _ = max_arr
        # 更新手部矩形框
        self._box[:] = min_x, min_y, max_x, max_y
        # 归一化计算,以手部矩形框左上角为原点,坐标值范围在0到1内
        self._norm_pos[:] = (self.raw_pos - min_arr) / (max_arr - min_arr)
        self._normalized_flag = True
        return self._norm_pos

    @property
    def wrist_npos(self) -> np.ndarray:
        """返回以手腕关键点坐标为原点的归一化关键点坐标"""
        if not self._wrist_npos_flag:
            return self.calc_wrist_npos()
        return self._wrist_npos

    def calc_wrist_npos(self) -> np.ndarray:
        """计算得到以手腕为原点的归一化坐标"""
        # 当前归一化坐标是以手部矩形框左上角为原点,坐标值范围在0到1内
        self._wrist_npos[:] = self.norm_pos
        # 以手腕关键点坐标位置为原点
        self._wrist_npos[1:, :] -= self._wrist_npos[0, :]
        self._wrist_npos_flag = True
        return self._wrist_npos

    def calc_wrist_npos2(self) -> np.ndarray:
        """计算得到以手腕为原点的归一化坐标"""
        self._wrist_npos[:] = np.dot(T_NORM2WRIST, self.norm_pos)
        self._wrist_npos_flag = True
        return self._wrist_npos

    @property
    def box(self) -> np.ndarray:
        """获取手部矩形框4个顶点坐标"""
        if not self._normalized_flag:
            self.normalization()
        return self._box

    @property
    def box_w(self) -> int:
        return self._box[3] - self._box[1]

    @property
    def box_h(self) -> int:
        return self._box[2] - self._box[0]

    @property
    def fingers_angle(self) -> np.ndarray:
        """获取手指每个关节点的弯曲角度"""
        if not self._angle_flag:
            return self.calc_5fingers_angle()
        return self._fingers_angle

    def calc_5fingers_angle(self) -> np.ndarray:
        """计算所有手指的每个关节点的弯曲角度"""
        # 使用行变换计算当前手部关键点坐标组成的向量,每个行向量代表一截手指关节向量
        fingers_vec = np.dot(T_NORM2VEC, self.norm_pos)  # 左乘行变换矩阵
        fingers_vec = fingers_vec[1:, :]  # 去掉第0行,第0行不是向量
        # 将每个手指向量都单位化,变成单位向量
        # 先计算行向量的模长
        vec_length = np.sqrt(np.sum(fingers_vec * fingers_vec, axis=1)).reshape((20, 1))
        fingers_vec = fingers_vec / vec_length  # 每个行向量元素都除以行向量模长
        # 计算每个手指的向量的夹角,这里range(0,20,4)是取每根手指最小的关键点到关键点0的向量
        for f_idx, v_idx in enumerate(range(0, 20, 4)):  # 每4截关节向量为一根手指
            finger_angle = np.acos(
                np.sum(
                    fingers_vec[v_idx : (v_idx + 3), :]
                    * fingers_vec[(v_idx + 1) : (v_idx + 4), :],
                    axis=1,
                )
            )
            # 将计算的角度结果,赋值给用于存储的变量
            self._fingers_angle[f_idx, :] = finger_angle
        self._angle_flag = True
        return self._fingers_angle

    @property
    def thumb_dist(self) -> np.ndarray:
        """获取4个指尖到大拇指的距离"""
        if not self._dist_flag:
            return self.calc_thumb_distance()
        return self._thumb_dist

    def calc_thumb_distance(self) -> np.ndarray:
        """计算大拇指指尖到其他4个手指指尖的距离"""
        thumb_tip_point = self.norm_pos[4, :]
        for i, finger_id in enumerate((8, 12, 16, 20)):
            # 用曼哈顿距离的计算量没有欧式距离大
            # 根据手指的编号获取该点的归一化的xyz坐标
            finger_point = self.norm_pos[finger_id, :]
            # 直接用l1范数来计算,也可以用np.sum(np.abs(point1 - point2))
            self._thumb_dist[i] = np.linalg.norm(thumb_tip_point - finger_point, ord=1)
        self._dist_flag = True
        return self._thumb_dist

    @property
    def data(self) -> np.ndarray:
        """获取所有已整合好的数据"""
        if not self._data_flag:
            return self.integrate_data()
        return self._data

    def integrate_data(self) -> np.ndarray:
        """整合所有手部相关数据并输出为一维数组"""
        self._data = self.norm_pos.copy()
        if self.hand_side != "right":  # 将关键点数据统一成右手数据
            self._data[:, 0] = 1 - self.norm_pos[:, 0]
        # 将数据展平然后合并
        self._data = np.ravel(self._data, order="C")
        self._data = np.concatenate(
            (self._data, np.ravel(self.fingers_angle, order="C"), self.thumb_dist)
        ).reshape(1, -1)
        self._data_flag = True
        return self._data

    @property
    def pos_data(self) -> np.ndarray:
        """获取手指角度和指尖距离数据"""
        if not self._pos_data_flag:
            self._pos_data = np.ravel(self.norm_pos, order="C").reshape(1, -1)
            self._pos_data_flag = True
        return self._pos_data

    @property
    def finger_data(self) -> np.ndarray:
        """获取手指角度和指尖距离数据"""
        if not self._finger_data_flag:
            return self.integrate_finger_data()
        return self._finger_data

    def integrate_finger_data(self) -> np.ndarray:
        """整合手指角度和指尖距离的数据并输出为一维数组"""
        self._finger_data = np.concatenate(
            (np.ravel(self.fingers_angle, order="C"), self.thumb_dist)
        ).reshape(1, -1)
        self._finger_data_flag = True
        return self._finger_data

    @property
    def wrist_npos_diff(self):
        """更新并获取两帧之间的归一化坐标差值"""
        if not self._wrist_npos_diff_flag:
            # 计算上次和本次的手部关键点距离
            self._wrist_npos_diff[:] = self.wrist_npos_diff_filter(
                np.linalg.norm(
                    self.wrist_npos[:, :2] - self.last_wnpos[:, :2], ord=1, axis=1
                )
            )
            self.last_wnpos[:, :] = self.wrist_npos  # 记录本次关键点位置
            self._wrist_npos_diff_flag = True
        return self._wrist_npos_diff

    @property
    def angle_diff(self):
        """更新并获取两帧之间手指关节角度的差值"""
        if not self._angle_diff_flag:
            # 计算上次和本次的手部关键点距离
            self._angle_diff = np.abs(self.fingers_angle - self._last_angle)
            self._last_angle[:, :] = self.fingers_angle  # 记录本次关键点位置
            self._angle_diff_flag = True
        return self._angle_diff
