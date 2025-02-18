from os import path
from collections import namedtuple
from abc import ABC, abstractmethod

import onnxruntime
from numpy import ndarray


class ModelLoader:
    __slots__ = ()

    @staticmethod
    def load_onnx(model_file_name):
        model_path = path.join(path.dirname(__file__), "models", model_file_name)
        model_path = path.abspath(model_path)
        return onnxruntime.InferenceSession(model_path)


ModelData = namedtuple("ModelData", "model mask")


class FingerModel(ABC):
    @abstractmethod
    def run_by_idx(self, idx: int, X: ndarray) -> ndarray:
        pass

    @abstractmethod
    def run_all(self, X: ndarray) -> list[ndarray]:
        pass


class FingerOutModel(FingerModel):
    __slots__ = "output"

    def __init__(self) -> None:
        FingerOutOutput = namedtuple(
            "FingerOutOutput", "thumb index_fg middle_fg ring_fg pinky"
        )

        self.output = FingerOutOutput(
            thumb=ModelData(
                model=ModelLoader.load_onnx("tb_out_lgbm0.onnx"),
                mask="pos_data",
            ),
            index_fg=ModelData(
                model=ModelLoader.load_onnx("if_out_lgbm0.onnx"),
                mask="pos_data",
            ),
            middle_fg=ModelData(
                model=ModelLoader.load_onnx("mf_out_lgbm0.onnx"),
                mask="pos_data",
            ),
            ring_fg=ModelData(
                model=ModelLoader.load_onnx("rf_out_lgbm0.onnx"),
                mask="pos_data",
            ),
            pinky=ModelData(
                model=ModelLoader.load_onnx("pk_out_lgbm0.onnx"),
                mask="pos_data",
            ),
        )

    def run_by_idx(self, idx: int, X: ndarray) -> ndarray:
        # X = X[self.output[idx].mask]
        return self.output[idx].model.run(["label"], {"f32X": X})[0]

    def run_all(self, X: ndarray) -> list[ndarray]:
        # return [fg.model.run(["label"], {"f32X": X[fg.mask]})[0] for fg in self.output]
        return [fg.model.run(["label"], {"f32X": X})[0] for fg in self.output]

    def thumb(self, X: ndarray) -> ndarray:
        return self.output.thumb.model.run(["label"], {"f32X": X})[0]

    def index_fg(self, X: ndarray) -> ndarray:
        return self.output.index_fg.model.run(["label"], {"f32X": X})[0]

    def middle_fg(self, X: ndarray) -> ndarray:
        return self.output.middle_fg.model.run(["label"], {"f32X": X})[0]

    def ring_fg(self, X: ndarray) -> ndarray:
        return self.output.ring_fg.model.run(["label"], {"f32X": X})[0]

    def pinky(self, X: ndarray) -> ndarray:
        return self.output.pinky.model.run(["label"], {"f32X": X})[0]


class FingerTouchModel(FingerModel):
    __slots__ = "output"

    def __init__(self) -> None:
        FingerTouchOutput = namedtuple(
            "FingerTouchOutput", "index_fg middle_fg ring_fg pinky"
        )
        self.output = FingerTouchOutput(
            index_fg=ModelData(
                model=ModelLoader.load_onnx("if_touch_lgbm0.onnx"),
                mask="finger_data",
            ),
            middle_fg=ModelData(
                model=ModelLoader.load_onnx("mf_touch_lgbm0.onnx"),
                mask="finger_data",
            ),
            ring_fg=ModelData(
                model=ModelLoader.load_onnx("rf_touch_lgbm0.onnx"),
                mask="finger_data",
            ),
            pinky=ModelData(
                model=ModelLoader.load_onnx("pk_touch_lgbm0.onnx"),
                mask="finger_data",
            ),
        )

    def run_by_idx(self, idx: int, X: ndarray) -> ndarray:
        return self.output[idx].model.run(["label"], {"f32X": X})[0]

    def run_all(self, X: ndarray) -> list[ndarray]:
        return [fg.model.run(["label"], {"f32X": X})[0] for fg in self.output]

    def index_fg(self, X: ndarray) -> ndarray:
        return self.output.index_fg.model.run(["label"], {"f32X": X})[0]

    def middle_fg(self, X: ndarray) -> ndarray:
        return self.output.middle_fg.model.run(["label"], {"f32X": X})[0]

    def ring_fg(self, X: ndarray) -> ndarray:
        return self.output.ring_fg.model.run(["label"], {"f32X": X})[0]

    def pinky(self, X: ndarray) -> ndarray:
        return self.output.pinky.model.run(["label"], {"f32X": X})[0]


class FingerCloseModel(FingerModel):
    __slots__ = "output"

    def __init__(self) -> None:
        FingerCloseOutput = namedtuple("FingerCloseOutput", "tb_if if_mf mf_rf rf_pk")
        self.output = FingerCloseOutput(
            tb_if=ModelData(
                model=ModelLoader.load_onnx("if_touch_lgbm0.onnx"),
                mask="pos_data",
            ),
            if_mf=ModelData(
                model=ModelLoader.load_onnx("mf_touch_lgbm0.onnx"),
                mask="pos_data",
            ),
            mf_rf=ModelData(
                model=ModelLoader.load_onnx("rf_touch_lgbm0.onnx"),
                mask="pos_data",
            ),
            rf_pk=ModelData(
                model=ModelLoader.load_onnx("pk_touch_lgbm0.onnx"),
                mask="pos_data",
            ),
        )

    def run_by_idx(self, idx: int, X: ndarray) -> ndarray:
        return self.output[idx].model.run(["label"], {"f32X": X})[0]

    def run_all(self, X: ndarray) -> list[ndarray]:
        return [fg.model.run(["label"], {"f32X": X})[0] for fg in self.output]

    def tb_if(self, X: ndarray) -> ndarray:
        return self.output.tb_if.model.run(["label"], {"f32X": X})[0]

    def if_mf(self, X: ndarray) -> ndarray:
        return self.output.if_mf.model.run(["label"], {"f32X": X})[0]

    def mf_rf(self, X: ndarray) -> ndarray:
        return self.output.mf_rf.model.run(["label"], {"f32X": X})[0]

    def rf_pk(self, X: ndarray) -> ndarray:
        return self.output.rf_pk.model.run(["label"], {"f32X": X})[0]


out_model = FingerOutModel()
touch_model = FingerTouchModel()
# close_model= FingerCloseModel()
