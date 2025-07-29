from abc import ABC, abstractmethod
from pathlib import Path
import pickle

import onnxruntime
from numpy import ndarray


# 调用不同格式模型的抽象基类
class ModelRunner(ABC):
    __slots__ = "file_name", "x_type"

    def __init__(self, file_name: str, x_type: str) -> None:
        super().__init__()
        self.file_name: str = file_name
        self.x_type: str = x_type

    def __repr__(self) -> str:
        return f"skhand.HandData.Models.ModelRunner(file={self.file_name}, x_type={self.x_type})"

    def __str__(self) -> str:
        return f"ModelRunner(file={self.file_name}, x_type={self.x_type})"

    @abstractmethod
    def run(self, X: ndarray) -> ndarray:
        """运行模型的统一抽象方法
        Args:
            X: 模型输入数据,统一为ndarray数据
        """
        pass

    def __call__(self, X: ndarray) -> ndarray:
        """运行模型,返回模型输出数据
        Args:
            X: 模型输入数据,统一为ndarray数据
        """
        return self.run(X)


# 加载不同格式模型的抽象基类
class ModelLoader(ABC):
    @classmethod
    @abstractmethod
    def load(cls, file_name: str, x_type: str) -> ModelRunner:
        """加载模型文件的统一抽象方法
        Args:
            file_name: 模型文件在models/文件夹里的文件名,包含后缀
            x_type: 指定该模型的输入数据
        """
        pass


# 封装onnx模型运行类的具体实现
class OnnxModelRunner(ModelRunner):
    __slots__ = "_model"

    def __init__(
        self,
        model: onnxruntime.InferenceSession,
        file_name: str,
        x_type: str,
    ) -> None:
        super().__init__(file_name, x_type)
        self._model: onnxruntime.InferenceSession = model

    @property
    def model(self) -> onnxruntime.InferenceSession:
        return self._model

    def run(self, X: ndarray) -> ndarray:
        """运行模型的统一抽象方法
        Args:
            X: 模型输入数据,统一为ndarray格式的数据
        """
        return self._model.run(["label"], {"f32X": X})[0]


# 封装onnx模型加载类的具体实现
class OnnxModelLoader(ModelLoader):
    @classmethod
    def load(cls, file_name: str, x_type: str) -> ModelRunner:
        """加载models文件夹下的ONNX模型文件
        Args:
            file_name: 模型文件在models/文件夹里的文件名,包含后缀
            x_type: 指定该模型的输入数据
        """
        model_path: Path = Path(__file__).parent / "models" / file_name
        if not model_path.exists():
            raise FileNotFoundError(f"Can't found the model file, path:{model_path}")
        return OnnxModelRunner(
            onnxruntime.InferenceSession(model_path.resolve().absolute()),
            file_name=file_name,
            x_type=x_type,
        )


class SklearnModelRunner(ModelRunner):
    __slots__ = "_model"

    def __init__(
        self,
        model,
        file_name: str,
        x_type: str,
    ) -> None:
        super().__init__(file_name, x_type)
        self._model = model
        raise NotImplementedError("Not implement SklearnModelRunner class yet")

    def run(self, X: ndarray) -> ndarray:
        """该方法仅用于测试和演示
        Args:
            X: 模型输入数据,统一为ndarray格式的数据
        """
        return X


class PickleModelLoader(ModelLoader):
    @classmethod
    def load(cls, file_name: str, x_type: str) -> ModelRunner:
        """加载models文件夹下的ONNX模型文件
        Args:
            file_name: 模型文件在models/文件夹里的文件名,包含后缀
            x_type: 指定该模型的输入数据
        """
        model_path: Path = Path(__file__).parent / "models" / file_name
        if not model_path.exists():
            raise FileNotFoundError(f"Can't found the model file, path:{model_path}")
        with open(model_path, "rb") as file:
            return SklearnModelRunner(pickle.load(file), file_name, x_type)


def load_model(file_name: str, x_type: str) -> ModelRunner:
    """加载模型,返回ModelRunner模型运行器
    Args:
        file_name: 模型文件在models/文件夹里的文件名,包含后缀
    """
    file_suffix: str = file_name.split(".")[-1].lower()
    if file_suffix == "onnx":
        return OnnxModelLoader.load(file_name, x_type)
    elif file_suffix in ["pickle", "pkl"]:
        return PickleModelLoader.load(file_name, x_type)
    raise ValueError(f"Can't load the model file named '{file_name}'")


class BaseHandModel(ABC):
    __slots__ = "_models"

    @abstractmethod
    def __init__(self) -> None:
        super().__init__()
        self._models: tuple[ModelRunner, ...] = ()

    def __len__(self) -> int:
        """返回模型的输出数据的个数"""
        return self._models.__len__()

    @abstractmethod
    def __getitem__(self, key: int) -> ModelRunner:
        """用索引调用相应的模型,返回模型对应的输出数据
        Args:
            key: 指定模型的输出数据的索引或是切片
        """
        return self._models[key]

    def __repr__(self) -> str:
        s = "skhand.HandData.Models.BaseHandModel(\n\t"
        s += "\n\t".join(f"{repr(m)}," for m in self._models)
        return s + "\n)"

    def __str__(self) -> str:
        s = "BaseHandModel(\n\t"
        s += "\n\t".join(f"{str(m)}," for m in self._models)
        return s + "\n)"


class FingerOutModel(BaseHandModel):
    __slots__ = "_models"

    def __init__(self) -> None:
        super().__init__()
        self._models: tuple[
            ModelRunner, ModelRunner, ModelRunner, ModelRunner, ModelRunner
        ] = (
            load_model("tb_out_model.onnx", "pos_data"),
            load_model("if_out_model.onnx", "pos_data"),
            load_model("mf_out_model.onnx", "pos_data"),
            load_model("rf_out_model.onnx", "pos_data"),
            load_model("pk_out_model.onnx", "pos_data"),
        )

    def __getitem__(self, key: int) -> ModelRunner:
        """用索引调用相应的模型,返回对应的模型
        Args:
            key: 0到4分别为大拇指,食指,中指,无名指,小拇指
        """
        if isinstance(key, int) and (key < 0 or key > 4):
            raise IndexError("Finger out model only has 5 output")
        return self._models[key]

    @property
    def thumb(self) -> ModelRunner:
        """运行大拇指伸出状态模型"""
        return self._models[0]

    @property
    def index_fg(self) -> ModelRunner:
        """运行食指伸出状态模型"""
        return self._models[1]

    @property
    def middle_fg(self) -> ModelRunner:
        """运行中指伸出状态模型"""
        return self._models[2]

    @property
    def ring_fg(self) -> ModelRunner:
        """运行无名指伸出状态模型"""
        return self._models[3]

    @property
    def pinky(self) -> ModelRunner:
        """运行小拇指伸出状态模型"""
        return self._models[4]

    def __repr__(self) -> str:
        s = "skhand.HandData.Models.FingerOutModel(\n\t"
        s += "\n\t".join(
            f"{n}={repr(m)},"
            for n, m in zip(
                ("thumb", "index_fg", "middle_fg", "ring_fg", "pinky"), self._models
            )
        )
        return s + "\n)"

    def __str__(self) -> str:
        s = "FingerOutModel(\n\t"
        s += "\n\t".join(
            f"{n}={str(m)},"
            for n, m in zip(
                ("thumb", "index_fg", "middle_fg", "ring_fg", "pinky"), self._models
            )
        )

        return s + "\n)"


class FingerTouchModel(BaseHandModel):
    __slots__ = "_models"

    def __init__(self) -> None:
        super().__init__()
        self._models: tuple[ModelRunner, ModelRunner, ModelRunner, ModelRunner] = (
            load_model("if_touch_model.onnx", "finger_data"),
            load_model("mf_touch_model.onnx", "finger_data"),
            load_model("rf_touch_model.onnx", "finger_data"),
            load_model("pk_touch_model.onnx", "finger_data"),
        )

    def __getitem__(self, key: int) -> ModelRunner:
        """用索引调用相应的模型,返回对应的模型
        Args:
            key: 0到3分别为大拇指到食指/中指/无名指/小拇指指尖的索引
        """
        if isinstance(key, int) and (key < 0 or key > 3):
            raise IndexError("Finger touch model only has 4 output")
        return self._models[key]

    @property
    def index_fg(self) -> ModelRunner:
        """运行大拇指到食指的指尖触碰状态模型"""
        return self._models[0]

    @property
    def middle_fg(self) -> ModelRunner:
        """运行大拇指到中指的指尖触碰状态模型"""
        return self._models[1]

    @property
    def ring_fg(self) -> ModelRunner:
        """运行大拇指到无名指的指尖触碰状态模型"""
        return self._models[2]

    @property
    def pinky(self) -> ModelRunner:
        """运行大拇指到小拇指的指尖触碰状态模型"""
        return self._models[3]

    def __repr__(self) -> str:
        s = "skhand.HandData.Models.FingerTouchModel(\n\t"
        s += "\n\t".join(
            f"{n}={repr(m)},"
            for n, m in zip(("index_fg", "middle_fg", "ring_fg", "pinky"), self._models)
        )
        return s + "\n)"

    def __str__(self) -> str:
        s = "FingerTouchModel(\n\t"
        s += "\n\t".join(
            f"{n}={str(m)},"
            for n, m in zip(("index_fg", "middle_fg", "ring_fg", "pinky"), self._models)
        )
        return s + "\n)"


class FingerCloseModel(BaseHandModel):
    __slots__ = "_models"

    def __init__(self) -> None:
        super().__init__()
        self._models: tuple[ModelRunner, ModelRunner, ModelRunner, ModelRunner] = (
            load_model("ti_close_model.onnx", "pos_data"),
            load_model("im_close_model.onnx", "pos_data"),
            load_model("mr_close_model.onnx", "pos_data"),
            load_model("rp_close_model.onnx", "pos_data"),
        )

    def __getitem__(self, key: int) -> ModelRunner:
        """用索引调用相应的模型,返回对应的模型
        Args:
            index: 0到3分别为大拇指和食指并拢,食指和中指并拢...的索引
        """
        if isinstance(key, int) and (key < 0 or key > 3):
            raise IndexError("Finger close model only has 4 model")
        return self._models[key]

    @property
    def tb_if(self) -> ModelRunner:
        """运行大拇指和食指并拢状态模型"""
        return self._models[0]

    @property
    def if_mf(self) -> ModelRunner:
        """运行食指和中指并拢状态模型"""
        return self._models[1]

    @property
    def mf_rf(self) -> ModelRunner:
        """运行中指和无名指并拢状态模型"""
        return self._models[2]

    @property
    def rf_pk(self) -> ModelRunner:
        """运行无名指和小拇指并拢状态模型"""
        return self._models[3]

    def __repr__(self) -> str:
        s = "skhand.HandData.Models.FingerCloseModel(\n\t"
        s += "\n\t".join(
            f"{n}={repr(m)},"
            for n, m in zip(("tb_if", "if_mf", "mf_rf", "rf_pk"), self._models)
        )
        return s + "\n)"

    def __str__(self) -> str:
        s = "FingerCloseModel(\n\t"
        s += "\n\t".join(
            f"{n}={str(m)},"
            for n, m in zip(("tb_if", "if_mf", "mf_rf", "rf_pk"), self._models)
        )
        return s + "\n)"


# 定义模型变量,保证每个模型只加载一次
finger_out_model: FingerOutModel = FingerOutModel()
finger_touch_model: FingerTouchModel = FingerTouchModel()
finger_close_model: FingerCloseModel = FingerCloseModel()
