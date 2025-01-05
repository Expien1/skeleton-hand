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
        self.columns_name: Series = Series(
            (
                "norm_pos0_x",
                "norm_pos0_y",
                "norm_pos0_z",
                "norm_pos1_x",
                "norm_pos1_y",
                "norm_pos1_z",
                "norm_pos2_x",
                "norm_pos2_y",
                "norm_pos2_z",
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
                "pos8_2thumb",
                "pos12_2thumb",
                "pos16_2thumb",
                "pos20_2thumb",
            )
        )

    def detect(self, image: np.ndarray) -> list[str]:
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
        """
        截取对应名字的手部图片
        返回截取后的图片
        """
        x0, y0, x1, y1 = self.get_hand_box(name)  # 获取手部矩形框
        x0 -= padx  # 设置边缘
        x1 += padx
        y0 -= pady
        y1 += pady
        # 保证xxyy都为正整数
        x0, y0, x1, y1 = map(lambda x: x if x >= 0 else 0, (x0, y0, x1, y1))
        # 保证所裁切出来的图片有效
        if x0 == x1 or y0 == y1:
            return None
        return original_img[y0:y1, x0:x1, :]

    def get_hand_box(self, name: str) -> tuple[int, int, int, int]:
        """
        获取对应名字的手部最小矩形框的在图片中的xxyy坐标
        """
        return self.hands_dict[name].box

    def get_img_pos(self, name: str, point_id: int = -1) -> np.ndarray:
        """
        获取名字为name的手部的第point_id个关键点在图片中的像素坐标
        name: 输入手部名称
        point_id: 输入关键点编号,默认值-1则返回所有关键点的像素坐标
        """
        one_hand = self.hands_dict[name]
        if 20 >= point_id >= 0:  # 返回指定关键点的像素坐标
            return one_hand.raw_pos[point_id, :2].astype(np.int32)
        elif point_id == -1:  # 返回所有关键点的像素坐标
            return one_hand.raw_pos[:, :2].astype(np.int32)
        raise ValueError(f"There is no coordinate data with point_id {point_id}")

    def get_norm_pos(self, name: str, point_id: int = -1) -> np.ndarray:
        """
        获取名字为name的手部的第point_id个关键点的xyz归一化坐标
        name: 输入手部名称
        point_id: 输入关键点编号,默认值-1则返回所有关键点的归一化坐标
        """
        one_hand = self.hands_dict[name]
        if 20 >= point_id >= 0:  # 返回指定的关键点归一化坐标
            return one_hand.norm_pos[point_id, :]
        elif point_id == -1:  # 返回该手部所有的关键点的归一化坐标
            return one_hand.norm_pos[:, :]  # 返回nrom_pos的副本,防止原来的数据被修改
        raise ValueError(f"There is no coordinate data with point_id {point_id}")

    def get_delta_img_pos(self, name: str, point_id: int = -1) -> np.ndarray:
        """
        获取名字为name的手部的第point_id个关键点两帧差的像素坐标
        name: 输入手部名称
        point_id: 输入关键点编号,默认值-1则返回所有关键点两帧差的像素坐标
        """
        one_hand = self.hands_dict[name]
        if 20 >= point_id >= 0:  # 返回指定关键点的两帧像素差坐标
            return one_hand.delta_pos[point_id, :]
        elif point_id == -1:  # 返回所有关键点的两帧像素差坐标
            return one_hand.delta_pos[:, :]
        raise ValueError(f"There is no coordinate data with point_id {point_id}")

    def get_norm2img_pos(
        self,
        name: str,
        point_id: int = -1,
        img_w: int = 100,
        img_h: int = 100,
        padx: int = 10,
        pady: int = 10,
    ) -> np.ndarray:
        """
        将归一化后的坐标转化为特定大小的图片位置坐标
        """
        npos = self.get_norm_pos(name, point_id)
        npos[:, 0] = npos[:, 0] * (img_w - (2 * padx)) + padx
        npos[:, 1] = npos[:, 1] * (img_h - (2 * pady)) + pady
        return npos

    def get_angle(self, name: str, point_id: int = -1) -> np.ndarray:
        """
        获取对应名字的手部关键点弧度制角度
        其中指尖和手腕没有角度数据
        """
        one_hand = self.hands_dict[name]
        finger_id, angle_id = divmod(point_id, 4)  # 计算关键点对应的角度数组中的索引
        if 20 >= point_id > 0 and angle_id != 0:  # 指尖和手腕没有角度数据
            return one_hand.fingers_angle[finger_id, (angle_id - 1)]
        elif point_id == -1:  # 没有输入point_id则返回全部角度
            return one_hand.fingers_angle[:, :]
        raise ValueError(f"There is no angle data with point_id {point_id}")

    def get_delta_angle(self, name: str, point_id: int = -1) -> np.ndarray:
        """
        获取对应名字的手部关键点弧度制角度差
        其中指尖和手腕没有角度数据
        """
        one_hand = self.hands_dict[name]
        finger_id, angle_id = divmod(point_id, 4)  # 计算关键点对应的角度数组中的索引
        if 20 >= point_id > 0 and angle_id != 0:  # 指尖和手腕没有角度数据
            return one_hand.delta_angle[finger_id, (angle_id - 1)]
        elif point_id == -1:  # 没有输入point_id则返回全部角度差
            return one_hand.delta_angle[:, :]
        raise ValueError(f"There is no angle data with point_id {point_id}")

    def get_thumb_dist(self, name: str, other_point_id: int = -1) -> np.ndarray:
        """
        获取对应名字的手部的从拇指到其他手指关键点的曼哈顿距离
        """
        one_hand = self.hands_dict[name]
        if other_point_id in (8, 12, 16, 20):
            # 计算对应的数组的索引
            finger_id, knuckle_id = divmod(other_point_id, 4)
            arr_id = (knuckle_id - 1) if knuckle_id != 0 else (finger_id + 1)
            return one_hand.thumb_dist[arr_id]
        elif other_point_id == -1:  # 没有输入id则返回全部到拇指的距离
            return one_hand.thumb_dist[:]
        raise ValueError(f"There is no distance data with point_id {other_point_id}")

    def get_hand_data(self, name: str | None = None) -> np.ndarray:
        """
        获取对应名字的手部的所有连续型数据的数组
        """
        if name is None:  # name没有填入参数,则返回全部手部数据
            return np.vstack(
                [self.hands_dict[n].data.reshape(1, -1) for n in self.hands_dict.keys()]
            )
        # 返回对应名字的手部数据
        one_hand = self.hands_dict[name]
        return one_hand.data.reshape(1, -1)

    def get_hand_DataFrame(self, name: str | None = None) -> DataFrame:
        """
        获取对应名字的手部的所有连续型数据的DataFrame格式的数据
        """
        return DataFrame(self.get_hand_data(name), columns=self.columns_name)
