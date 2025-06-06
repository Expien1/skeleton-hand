"""
封装onnx模型
目前只支持使用onnx模型
"""

from os import path
from collections import namedtuple
from abc import ABC, abstractmethod

import onnxruntime
from numpy import ndarray


def load_onnx(model_file_name):
    """加载onnx模型
    Args:
        model_file_name: onnx模型文件路径
    """
    model_path = path.join(path.dirname(__file__), "models", model_file_name)
    model_path = path.abspath(model_path)
    return onnxruntime.InferenceSession(model_path)


ModelData = namedtuple("ModelData", "model mask")


class FingerModel(ABC):
    @abstractmethod
    def run_by_idx(self, idx: int, X: ndarray) -> ndarray:
        """通过输出索引来指定需要运行的手指状态模型
        Args:
            idx: 模型输出的索引
            X: 模型输入
        """
        pass

    @abstractmethod
    def run_all(self, X: ndarray) -> list[ndarray]:
        """运行该手部的所有手指状态的模型
        Args:
            X: 模型输入
        """
        pass


class FingerOutModel(FingerModel):
    __slots__ = "output"

    def __init__(self) -> None:
        """手指伸出状态模型"""
        # 定义手指伸出状态模型的输出
        FingerOutOutput = namedtuple(
            "FingerOutOutput", "thumb index_fg middle_fg ring_fg pinky"
        )
        # 加载模型和指定模型输入
        self.output = FingerOutOutput(
            thumb=ModelData(
                model=load_onnx("tb_out_model.onnx"),
                mask="pos_data",
            ),
            index_fg=ModelData(
                model=load_onnx("if_out_model.onnx"),
                mask="pos_data",
            ),
            middle_fg=ModelData(
                model=load_onnx("mf_out_model.onnx"),
                mask="pos_data",
            ),
            ring_fg=ModelData(
                model=load_onnx("rf_out_model.onnx"),
                mask="pos_data",
            ),
            pinky=ModelData(
                model=load_onnx("pk_out_model.onnx"),
                mask="pos_data",
            ),
        )

    def run_by_idx(self, idx: int, X: ndarray) -> ndarray:
        """通过输出索引来指定需要运行的手指伸出状态模型
        Args:
            idx: 0到4分别为大拇指,食指,中指,无名指,小拇指
            X: 模型输入,所有手部关键点的位置数据
        """
        # X = X[self.output[idx].mask]
        return self.output[idx].model.run(["label"], {"f32X": X})[0]

    def run_all(self, X: ndarray) -> list[ndarray]:
        """运行该手部的所有手指伸出状态的模型
        Args:
            X: 模型输入,所有手部关键点的位置数据
        """
        # return [fg.model.run(["label"], {"f32X": X[fg.mask]})[0] for fg in self.output]
        return [fg.model.run(["label"], {"f32X": X})[0] for fg in self.output]

    def thumb(self, X: ndarray) -> ndarray:
        """运行大拇指伸出状态模型
        Args:
            X: 模型输入,所有手部关键点的位置数据
        """
        return self.output.thumb.model.run(["label"], {"f32X": X})[0]

    def index_fg(self, X: ndarray) -> ndarray:
        """运行食指伸出状态模型
        Args:
            X: 模型输入,所有手部关键点的位置数据
        """
        return self.output.index_fg.model.run(["label"], {"f32X": X})[0]

    def middle_fg(self, X: ndarray) -> ndarray:
        """运行中指伸出状态模型
        Args:
            X: 模型输入,所有手部关键点的位置数据
        """
        return self.output.middle_fg.model.run(["label"], {"f32X": X})[0]

    def ring_fg(self, X: ndarray) -> ndarray:
        """运行无名指伸出状态模型
        Args:
            X: 模型输入,所有手部关键点的位置数据
        """
        return self.output.ring_fg.model.run(["label"], {"f32X": X})[0]

    def pinky(self, X: ndarray) -> ndarray:
        """运行小拇指伸出状态模型
        Args:
            X: 模型输入,所有手部关键点的位置数据
        """
        return self.output.pinky.model.run(["label"], {"f32X": X})[0]


class FingerTouchModel(FingerModel):
    __slots__ = "output"

    def __init__(self) -> None:
        # 定义指尖触碰状态的输出
        FingerTouchOutput = namedtuple(
            "FingerTouchOutput", "index_fg middle_fg ring_fg pinky"
        )
        # 加载模型和定义模型输入
        self.output = FingerTouchOutput(
            index_fg=ModelData(
                model=load_onnx("if_touch_model.onnx"),
                mask="finger_data",
            ),
            middle_fg=ModelData(
                model=load_onnx("mf_touch_model.onnx"),
                mask="finger_data",
            ),
            ring_fg=ModelData(
                model=load_onnx("rf_touch_model.onnx"),
                mask="finger_data",
            ),
            pinky=ModelData(
                model=load_onnx("pk_touch_model.onnx"),
                mask="finger_data",
            ),
        )

    def run_by_idx(self, idx: int, X: ndarray) -> ndarray:
        """通过输出索引来指定需要运行的指尖触碰状态模型
        Args:
            idx: 0到3分别为大拇指到食指/中指/无名指/小拇指指尖的索引
            X: 模型输入,包括所有手指弯曲和指尖距离数据
        """
        return self.output[idx].model.run(["label"], {"f32X": X})[0]

    def run_all(self, X: ndarray) -> list[ndarray]:
        """运行该手部的所有指尖触碰状态的模型
        Args:
            X: 模型输入,包括所有手指弯曲和指尖距离数据
        """
        return [fg.model.run(["label"], {"f32X": X})[0] for fg in self.output]

    def index_fg(self, X: ndarray) -> ndarray:
        """运行大拇指到食指的指尖触碰状态模型
        Args:
            X: 模型输入,包括所有手指弯曲和指尖距离数据
        """
        return self.output.index_fg.model.run(["label"], {"f32X": X})[0]

    def middle_fg(self, X: ndarray) -> ndarray:
        """运行大拇指到中指的指尖触碰状态模型
        Args:
            X: 模型输入,包括所有手指弯曲和指尖距离数据
        """
        return self.output.middle_fg.model.run(["label"], {"f32X": X})[0]

    def ring_fg(self, X: ndarray) -> ndarray:
        """运行大拇指到无名指的指尖触碰状态模型
        Args:
            X: 模型输入,包括所有手指弯曲和指尖距离数据
        """
        return self.output.ring_fg.model.run(["label"], {"f32X": X})[0]

    def pinky(self, X: ndarray) -> ndarray:
        """运行大拇指到小拇指的指尖触碰状态模型
        Args:
            X: 模型输入,包括所有手指弯曲和指尖距离数据
        """
        return self.output.pinky.model.run(["label"], {"f32X": X})[0]


class FingerCloseModel(FingerModel):
    __slots__ = "output"

    def __init__(self) -> None:
        # 定义手指并拢状态模型的输出
        FingerCloseOutput = namedtuple("FingerCloseOutput", "tb_if if_mf mf_rf rf_pk")
        # 加载模型和指定模型输入
        self.output = FingerCloseOutput(
            tb_if=ModelData(
                model=load_onnx("ti_close_model.onnx"),
                mask="pos_data",
            ),
            if_mf=ModelData(
                model=load_onnx("im_close_model.onnx"),
                mask="pos_data",
            ),
            mf_rf=ModelData(
                model=load_onnx("mr_close_model.onnx"),
                mask="pos_data",
            ),
            rf_pk=ModelData(
                model=load_onnx("rp_close_model.onnx"),
                mask="pos_data",
            ),
        )

    def run_by_idx(self, idx: int, X: ndarray) -> ndarray:
        """通过输出索引来指定需要运行的手指并拢状态模型
        Args:
            idx: 0到3分别为大拇指和食指并拢,食指和中指并拢...的索引
            X: 模型输入,所有手部关键点的位置数据
        """
        return self.output[idx].model.run(["label"], {"f32X": X})[0]

    def run_all(self, X: ndarray) -> list[ndarray]:
        """运行该手部的所有手指并拢状态的模型
        Args:
            X: 模型输入,所有手部关键点的位置数据
        """
        return [fg.model.run(["label"], {"f32X": X})[0] for fg in self.output]

    def tb_if(self, X: ndarray) -> ndarray:
        """运行大拇指和食指并拢状态模型
        Args:
            X: 模型输入,所有手部关键点的位置数据
        """
        return self.output.tb_if.model.run(["label"], {"f32X": X})[0]

    def if_mf(self, X: ndarray) -> ndarray:
        """运行食指和中指并拢状态模型
        Args:
            X: 模型输入,所有手部关键点的位置数据
        """
        return self.output.if_mf.model.run(["label"], {"f32X": X})[0]

    def mf_rf(self, X: ndarray) -> ndarray:
        """运行中指和无名指并拢状态模型
        Args:
            X: 模型输入,所有手部关键点的位置数据
        """
        return self.output.mf_rf.model.run(["label"], {"f32X": X})[0]

    def rf_pk(self, X: ndarray) -> ndarray:
        """运行无名指和小拇指并拢状态模型
        Args:
            X: 模型输入,所有手部关键点的位置数据
        """
        return self.output.rf_pk.model.run(["label"], {"f32X": X})[0]


# 直接定义全局变量来实现单例模式
out_model = FingerOutModel()
touch_model = FingerTouchModel()
close_model = FingerCloseModel()
