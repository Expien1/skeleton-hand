import numpy as np


class EWMAFilter:
    __slots__ = "smooth_factor", "x"

    # 基本的指数加权移动平均 (Basic EWMA)
    def __init__(self, smooth_factor: float = 0.5):
        # 平滑因子越接近0平滑效果越强.但响应速度更慢;越接近1响应速度越快,但平滑效果较弱
        self.smooth_factor: float = smooth_factor  # 设定平滑因子
        # 这里的平滑因子可以设置成一个向量,来更新整个矩阵,所以没有指定类型
        self.x = None  # 未设定初始值

    def update(self, new_x, new_smooth: None | float = None):
        if self.x is None:
            self.x = new_x  # 设定初始值
        else:  # 根据公式进行平滑
            if new_smooth is not None:  # 使用新的平滑因子
                self.smooth_factor = new_smooth
            self.x = self.smooth_factor * new_x + (1 - self.smooth_factor) * self.x
        return self.x

    def __call__(self, new_x, new_smooth: None | float = None):
        return self.update(new_x, new_smooth)

    def decorator(self, func):
        """装饰器,将装饰函数的返回结果进行过滤"""

        def wrapper(*args, **kwargs):
            return self.update(func(*args, **kwargs))

        return wrapper


class AdaptiveEWMAFilter:
    __slots__ = "smooth_factor", "learning_rate", "x", "last_error"

    # 自适应指数平滑 (Adaptive Exponential Smoothing)
    def __init__(self, smooth_factor: float = 0.5, learning_rate: float = 0.1):
        self.smooth_factor: float = smooth_factor  # 设定平滑因子的初始值
        # 自适应指数平滑的核心思想是根据当前误差和前一时刻的误差来调整平滑因子
        self.learning_rate: float = learning_rate  # 设定学习率
        # 学习率越大,调整得越快;学习率越小,调整得越慢
        self.x = None  # 未设定初始值
        self.last_error = 0  # 记录上一次的预测误差

    def update(self, new_x):
        if self.x is None:
            self.x = new_x  # 设定初始值
        else:
            # 计算当前误差,当前误差=当前观测值-上一次的预测值(就是self.x)
            cur_error = np.abs(new_x - self.x)
            # 使用学习率和误差来计算平滑因子
            self.smooth_factor += self.learning_rate * (cur_error - self.last_error)
            # 确保更新后的平滑因子的值在0到1间
            self.smooth_factor = np.clip(self.smooth_factor, 0, 1)
            # 更新过滤后的值
            self.x = self.smooth_factor * new_x + (1 - self.smooth_factor) * self.x
            # 更新上一次的误差
            self.last_error = cur_error
        return self.x

    def __call__(self, new_x):
        return self.update(new_x)

    def decorator(self, func):
        """装饰器,将装饰函数的返回结果进行过滤"""

        def wrapper(*args, **kwargs):
            return self.update(func(*args, **kwargs))

        return wrapper
