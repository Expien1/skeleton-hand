import numpy as np
from pandas import DataFrame, Series

from .HandData.OneHand import OneHand
from .HandDetector.VisualHandDetector import VisualHandDetector
from .HandDetector.MediaPipeHandDetector import MediaPipeHandDetector


class HandInput:
    __slots__ = "hands_dict", "detector", "columns_name"

    def __init__(
        self,
        hands_name_ls: list[str] = ["h0"],
        hands_detector: type[VisualHandDetector] = MediaPipeHandDetector,
        **detector_kwargs,
    ) -> None:
        # 利用传入的手部名字来创建对应的手部数据字典
        self.hands_dict: dict[str, OneHand] = {
            name: OneHand() for name in hands_name_ls
        }
        # 根据传入的检测器类创建视觉手部检测器实例,添加手部名字列表的数量参数
        detector_kwargs["max_num_hands"] = len(hands_name_ls)
        self.detector: VisualHandDetector = hands_detector(**detector_kwargs)
        # 定义集成数据之后的列名
        self.columns_name = Series(
            (
                "norm_pos0_x",
                "norm_pos0_y",
                "norm_pos0_z",
                "norm_pos1_x",
                "norm_pos1_y",
                "norm_pos1_z",
                "norm_pos2_x",
                "norm_pos2_y",
                "norm_p os2_z",
                "norm_pos3_x",
                "norm_pos3_y",
                "norm_pos3_z",
                "norm_pos4_x",
                "norm_pos4_y",
                "norm_pos4_z",
                "norm_pos5_x",
                "norm_pos5_y",
                "norm_pos5_z",
                "norm_pos6_x",
                "norm_pos6_y",
                "norm_pos6_z",
                "norm_pos7_x",
                "norm_pos7_y",
                "norm_pos7_z",
                "norm_pos8_x",
                "norm_pos8_y",
                "norm_pos8_z",
                "norm_pos9_x",
                "norm_pos9_y",
                "norm_pos9_z",
                "norm_pos10_x",
                "norm_pos10_y",
                "norm_pos10_z",
                "norm_pos11_x ",
                "norm_pos11_y",
                "norm_pos11_z",
                "norm_pos12_x",
                "norm_pos12_y",
                "norm_pos12_z",
                "norm_pos13_x",
                "norm_pos13_y",
                "norm_pos13_z ",
                "norm_pos14_x",
                "norm_pos14_y",
                "norm_pos14_z",
                "norm_pos15_x",
                "norm_pos15_y",
                "norm_pos15_z",
                "norm_pos16_x",
                "norm_pos16_y ",
                "norm_pos16_z",
                "norm_pos17_x",
                "norm_pos17_y",
                "norm_pos17_z",
                "norm_pos18_x",
                "norm_pos18_y",
                "norm_pos18_z",
                "norm_pos19_x ",
                "norm_pos19_y",
                "norm_pos19_z",
                "norm_pos20_x",
                "norm_pos20_y",
                "norm_pos20_z",
                "p1_angle",
                "p2_angle",
                "p3_angle",
                "p5_angle",
                "p6_angle",
                "p7_angle",
                "p9_angle",
                "p10_angle",
                "p11_angle",
                "p13_angle",
                "p14 _angle",
                "p15_angle",
                "p17_angle",
                "p18_angle",
                "p19_angle",
                "pos5_2thumb",
                "pos6_2thumb",
                "pos7_2thumb",
                "pos8_2thumb",
                "pos12_2thumb",
                "pos16_2thumb",
                "pos20_2thumb",
            )
        )

    def run(self, image: np.ndarray) -> list[str]:
        """
        运行手部关键点检测器
        检测成功则返回成功检测到的手部的名称
        没有检测到手部或检测失败则返回空列表
        """
        # 获取检测结果
        return self.detector.detect(image, self.hands_dict)

    def get_hand_img(
        self, name: str, original_img: np.ndarray, padx: int = 10, pady: int = 10
    ) -> np.ndarray | None:
        """截取对应名字的手部图片,返回截取后的图片"""
        # 获取手部矩形框
        x0, y0, x1, y1 = self.get_hand_box(name)
        # 保证xxyy都为正整数
        x0 -= padx
        x0 = x0 if x0 >= 0 else 0
        x1 += padx
        x1 = x1 if x1 >= 0 else 0
        y0 -= pady
        y0 = y0 if y0 >= 0 else 0
        y1 += pady
        y1 = y1 if y1 >= 0 else 0
        # 保证所裁切出来的图片有效
        if x0 == x1 or y0 == y1:
            return None
        return original_img[y0:y1, x0:x1, :]

    def get_hand_box(self, name: str) -> tuple[int, int, int, int]:
        """获取对应名字的手部最小矩形框的在图片中的xxyy坐标"""
        return self.hands_dict[name].box

    def get_img_pos(self, name: str, id: int = -1) -> np.ndarray:
        """获取对应名字的手部在图片中的关键点xyz坐标,坐标id错误则报错"""
        one_hand = self.hands_dict[name]
        if 20 >= id >= 0:
            return one_hand.raw_pos[id, :]
        elif id == -1:
            return one_hand.raw_pos
        raise ValueError(f"There is no coordinate data with id {id}")

    def get_norm_pos(self, name: str, id: int = -1) -> np.ndarray:
        """获取对应名字的手部在归一化后的关键点xyz坐标,坐标id错误则报错"""
        one_hand = self.hands_dict[name]
        if 20 >= id >= 0:
            return one_hand.norm_pos[id, :]
        elif id == -1:
            return one_hand.norm_pos
        raise ValueError(f"There is no coordinate data with id {id}")

    def get_norm_pos_to_img(
        self, name: str, id: int, img_w: int, img_h: int, padx: int, pady: int
    ) -> tuple[int, int, float]:
        if id == -1:
            raise ValueError(f"There is no coordinate data with id {id}")
        res = list(self.get_norm_pos(name, id))
        res[0] = int(res[0] * (img_w - (2 * padx)) + padx)
        res[1] = int(res[1] * (img_h - (2 * pady)) + pady)
        return tuple(res)

    def get_angle(self, name: str, id: int = -1) -> np.ndarray:
        """获取对应名字的手部关键点弧度制角度"""
        if id == -1:  # 没有输入id则返回全部角度
            one_hand = self.hands_dict[name]
            return one_hand.fingers_angle
        if id > 20 or id < 0:
            raise ValueError(f"There is no angle data with id {id}")
        # 计算关键点对应的角度数组中的索引
        finger_id, angle_id = divmod(id, 4)
        if angle_id != 0:
            one_hand = self.hands_dict[name]
            return one_hand.fingers_angle[finger_id, (angle_id - 1)]
        raise ValueError(f"There is no angle data with id {id}")

    def get_thumb_dist(self, name: str, other_point_id: int = -1) -> np.ndarray:
        """获取对应名字的手部,从拇指到其他手指关键点的曼哈顿距离"""
        one_hand = self.hands_dict[name]
        if other_point_id in (5, 6, 7, 8, 12, 16, 20):
            # 计算对应的数组的索引
            finger_id, knuckle_id = divmod(other_point_id, 4)
            arr_id = (knuckle_id - 1) if knuckle_id != 0 else (finger_id + 1)
            return one_hand.thumb_dist[arr_id]
        elif other_point_id == -1:  # 没有输入id则返回全部到拇指的距离
            return one_hand.thumb_dist
        raise ValueError(f"There is no distance data with id {id}")

    def get_hand_data(self, name: str) -> np.ndarray:
        """整合并返回对应名字的手部的所有连续型数据的一维数组"""
        one_hand = self.hands_dict[name]
        return one_hand.data

    def get_hand_DataFrame(self, name: str) -> DataFrame:
        """整合并返回对应名字的手部的所有连续型数据的一维数组"""
        return DataFrame(
            self.hands_dict[name].data.reshape(1, -1),
            columns=self.columns_name,
        )
