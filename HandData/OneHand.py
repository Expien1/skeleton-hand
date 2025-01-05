import numpy as np

from ._global_ import T4VEC_MATIX


class OneHand:
    __slots__ = (
        "is_left",
        "raw_pos",
        "_norm_pos",
        "_box",
        "_normalized_flag",
        "_fingers_angle",
        "_angle_flag",
        "_thumb_dist",
        "_dist_flag",
        "_data",
        "_data_flag",
        "_delta_pos",
        "_last_pos",
        "_delta_pos_flag",
        "_delta_angle",
        "_last_angle",
        "_delta_angle_flag",
    )

    def __init__(self):
        """一只手的关键点位置数据"""
        # 用一个bool值记录当前手是左手还是右手
        self.is_left: bool = False  # 初始值为右手
        # 定义一个21x3的全0的二维数组来接收返回的21个手部关键点的xyz坐标
        # raw_pos表示这是原始传入的手部关键点的位置,初始创建一个空的数组,里面初始值是脏数据
        self.raw_pos: np.ndarray = np.zeros((21, 3), dtype=np.float32)

        # norm_pos表示是再手部box里归一化之后的坐标点坐标
        self._norm_pos: np.ndarray = np.zeros((21, 3), dtype=np.float32)
        # 手部矩形框的四个坐标点,初始值为(0,0,0,0)
        self._box: tuple = (0, 0, 0, 0)
        # 标记是否已经计算好归一化和矩形框的四个坐标点
        self._normalized_flag: bool = False

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
        # 标志当前_data变量是否为最新的数据
        self._data_flag: bool = False

        # 创建一个21x3的二维数组来存储每个关键点的移动差值
        self._delta_pos: np.ndarray = self.raw_pos.copy()
        self._last_pos: np.ndarray = self.raw_pos.copy()
        # 标志当前_delta_pos变量存的是否为最新的速度
        self._delta_pos_flag: bool = False

        # 定义一个5x3的二维数组用来存储5根手指的3个关节点的弯曲角度的差值
        self._delta_angle: np.ndarray = self._fingers_angle.copy()
        self._last_angle: np.ndarray = self._fingers_angle.copy()
        # 标志当前_delta_angle变量存的是否为最新的速度
        self._delta_angle_flag: bool = False

    def reset_all_flags(self):
        """重置所有数据的更新标志为False"""
        self._normalized_flag = False
        self._angle_flag = False
        self._dist_flag = False
        self._data_flag = False
        self._delta_pos_flag = False
        self._delta_angle_flag = False

    @property
    def norm_pos(self) -> np.ndarray:
        """返回归一化后的关键点坐标"""
        if not self._normalized_flag:
            self.normalization()
            self._normalized_flag = True
        return self._norm_pos

    @property
    def box(self) -> tuple[int, int, int, int]:
        """获取手部矩形框4个顶点坐标"""
        if not self._normalized_flag:
            self.normalization()
            self._normalized_flag = True
        return self._box

    def normalization(self) -> np.ndarray:
        """将传入的初始手部关键点坐标进行归一化为手部框内的相对位置"""
        # 计算得到手部矩形框的四个顶点坐标
        min_arr = self.raw_pos[:, :2].min(axis=0)
        max_arr = self.raw_pos[:, :2].max(axis=0)
        # 顺便更新手部矩形框
        min_x, min_y = min_arr  # 计算能框住手部的最小矩形框
        max_x, max_y = max_arr
        self._box = tuple(map(int, (min_x, min_y, max_x, max_y)))
        # 计算顶点的范围,最大减最小
        len_x, len_y = max_arr - min_arr
        # 计算归一化之后的xyz
        self._norm_pos[:, 0] = (self.raw_pos[:, 0] - min_x) / len_x
        self._norm_pos[:, 1] = (self.raw_pos[:, 1] - min_y) / len_y
        self._norm_pos[:, 2] = self.raw_pos[:, 2]  # z轴归一化已经在raw_pos计算了
        return self._norm_pos

    @property
    def fingers_angle(self) -> np.ndarray:
        """获取手指每个关节点的弯曲角度"""
        if not self._angle_flag:
            self.calc_5fingers_angle()
            self._angle_flag = True
        return self._fingers_angle

    def calc_5fingers_angle(self) -> np.ndarray:
        """计算所有手指的每个关节点的弯曲角度"""
        # 使用行变换计算当前手部关键点坐标组成的向量,每个行向量代表一截手指关节向量
        fingers_vec = np.dot(T4VEC_MATIX, self.norm_pos)  # 左乘(行变换)转移矩阵
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
        return self._fingers_angle

    @property
    def thumb_dist(self) -> np.ndarray:
        """获取4个指尖到大拇指的距离"""
        if not self._dist_flag:
            self.calc_thumb_distance()
            self._dist_flag = True
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
        return self._thumb_dist

    @property
    def data(self) -> np.ndarray:
        """获取所有已整合好的数据"""
        if not self._data_flag:
            self.integrate_data()
            self._data_flag = True
        return self._data

    def integrate_data(self) -> np.ndarray:
        """整合所有手部相关数据并输出为一维数组"""
        self._data = self.norm_pos.copy()
        if self.is_left:  # 将关键点数据统一成右手数据
            self._data[:, 0] = 1 - self.norm_pos[:, 0]
        # 将数据展平然后合并
        self._data = np.ravel(self._data, order="C")
        self._data = np.concatenate(
            (self._data, np.ravel(self.fingers_angle, order="C"), self.thumb_dist)
        )
        return self._data

    @property
    def delta_pos(self):
        """更新并获取手部关键点在图片中的像素差值"""
        if not self._delta_pos_flag:
            # 计算上次和本次的手部关键点距离
            self._delta_pos = self.raw_pos - self._last_pos
            self._last_pos = self.raw_pos.copy()  # 记录本次关键点位置
            self._delta_pos_flag = True
        return self._delta_pos

    @property
    def delta_angle(self):
        """更新并获取手部关键点在图片中的像素差值"""
        if not self._delta_angle_flag:
            # 计算上次和本次的手部关键点距离
            self._delta_angle = self.fingers_angle - self._last_angle
            self._last_angle = self.fingers_angle.copy()  # 记录本次关键点位置
            self._delta_angle_flag = True
        return self._delta_angle
