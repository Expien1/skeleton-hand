import numpy as np


class OneHand:
    __slots__ = (
        "is_left",
        "raw_pos",
        "_norm_pos",
        "_box",
        "_fingers_angle",
        "T4VEC",
        "_thumb_dist",
    )

    def __init__(self):
        """一只手的关键点位置数据"""
        # 用一个bool值记录当前手是左手还是右手
        self.is_left: bool = False  # 初始值为右手
        # 定义一个21x3的全0的二维数组来接收返回的21个手部关键点的xyz坐标
        # raw_pos表示这是原始传入的手部关键点的位置,初始创建一个空的数组,里面初始值是脏数据
        self.raw_pos: np.ndarray = np.empty((21, 3), dtype=np.float64)

        # norm_pos表示是再手部box里归一化之后的坐标点坐标
        self._norm_pos: np.ndarray = np.empty((21, 3), dtype=np.float64)
        # 手部矩形框的四个坐标点,初始值为(0,0,0,0)
        self._box: tuple = (0, 0, 0, 0)

        # 定义一个5x3的二维数组用来存储5根手指的3个关节点的弯曲角度
        self._fingers_angle: np.ndarray = np.empty((5, 3), dtype=np.float64)
        # 定义一个用于将归一化后的手部坐标点矩阵norm_pos通过行变换为向量的矩阵
        # T4VEC means transition matrix T for(4) vector
        self.T4VEC = np.array(  # 变换后除了第0行不是向量
            [
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [-1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [-1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [-1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0],
                [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1],
            ],
        )

        # 定义一个一维数组来存储大拇指与其他关键点的距离
        self._thumb_dist: np.ndarray = np.empty(7, dtype=np.float64)

    def update(self):
        self.normalization()
        self.calc_5fingers_angle()
        self.calc_thumb_distance()

    @property
    def norm_pos(self) -> np.ndarray:
        return self._norm_pos

    def normalization(self) -> np.ndarray:
        """将传入的初始手部关键点坐标进行归一化为手部框内的相对位置"""
        # 计算得到手部矩形框的四个顶点坐标
        min_arr = self.raw_pos[:, :2].min(axis=0)
        max_arr = self.raw_pos[:, :2].max(axis=0)
        # 顺便计算能框住手部的最小矩形框
        min_x, min_y = min_arr
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
    def box(self) -> tuple[int, int, int, int]:
        """获取手部矩形框4个顶点坐标"""
        return self._box

    @property
    def fingers_angle(self) -> np.ndarray:
        return self._fingers_angle

    def calc_5fingers_angle(self) -> np.ndarray:
        """计算所有手指的每个关节点的弯曲角度"""
        # 使用行变换计算当前手部关键点坐标组成的向量,每个行向量代表一截手指关节向量
        fingers_vec = np.dot(self.T4VEC, self._norm_pos)  # 左乘(行变换)转移矩阵
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
        return self._thumb_dist

    def calc_thumb_distance(self) -> np.ndarray:
        """计算大拇指指尖到其他关键点的距离"""
        thumb_tip_point = self._norm_pos[4, :]
        for i, finger_id in enumerate((5, 6, 7, 8, 12, 16, 20)):
            # 用曼哈顿距离的计算量没有欧式距离大
            # 根据手指的编号获取该点的归一化的xyz坐标
            finger_point = self._norm_pos[finger_id, :]
            # 直接用l1范数来计算
            # 也可以用np.sum(np.abs(point1 - point2))
            self._thumb_dist[i] = np.linalg.norm(thumb_tip_point - finger_point, ord=1)
        return self._thumb_dist

    @property
    def data(self) -> np.ndarray:
        """整合所有手部相关数据并输出为一维数组"""
        data = self._norm_pos.copy()
        if self.is_left:  # 将数据统一成右手数据
            data[:, 0] = 1 - self._norm_pos[:, 0]
        data = np.ravel(data, order="C")
        angle = np.ravel(self._fingers_angle, order="C")
        return np.concatenate((data, angle, self._thumb_dist))
