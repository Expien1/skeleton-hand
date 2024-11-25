import cv2
import numpy as np
from pandas import DataFrame, Series

from .HandsData.OneHand import OneHand
from .HandsDetector.VisualHandsDetector import VisualHandsDetector
from .HandsDetector.MediaPipeHandsDetector import MediaPipeHandsDetector


class HandsInput:
    __slots__ = "hands_dict", "detector", "columns_name"

    def __init__(
        self,
        hands_name_ls: list[str] = ["h0"],
        hands_detector: type[VisualHandsDetector] = MediaPipeHandsDetector,
        **detector_kwargs,
    ) -> None:
        # 利用传入的手部名字来创建对应的手部数据字典
        self.hands_dict: dict[str, OneHand] = {
            name: OneHand() for name in hands_name_ls
        }
        # 根据传入的检测器类创建视觉手部检测器实例,添加手部名字列表的数量参数
        detector_kwargs["max_num_hands"] = len(hands_name_ls)
        self.detector: VisualHandsDetector = hands_detector(**detector_kwargs)
        # 定义集成数据之后的列名
        self.columns_name = (
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

    def get_all_data(self, name: str) -> np.ndarray:
        """整合并返回对应名字的手部的所有连续型数据的一维数组"""
        one_hand = self.hands_dict[name]
        return one_hand.integrated()

    def get_data_frame(self, name: str) -> DataFrame:
        """整合并返回对应名字的手部的所有连续型数据的一维数组"""
        one_hand = self.hands_dict[name]
        return DataFrame(
            one_hand.integrated().reshape(1, -1),
            columns=Series(self.columns_name),
        )

    def draw_hand(
        self,
        name: str,
        original_img: np.ndarray,
        point_radius: int = 4,
        color: tuple[int, int, int] = (255, 255, 255),
        thickness: int = 3,
        base_rgb: int = 4,
    ) -> np.ndarray:
        """在原图中绘制手部"""
        x0, y0 = map(int, self.get_img_pos(name, 0)[:2])
        xp, yp = x0, y0
        for i in range(21):
            # 绘制关键点
            point_x, point_y = map(int, self.get_img_pos(name, i)[:2])
            point_z = self.get_img_pos(name, i)[2]
            # 用颜色深度表示z轴
            point_color = tuple(
                map(
                    lambda c: int((c / base_rgb) * (1 + (base_rgb - 1) * point_z)),
                    color,
                )
            )
            cv2.circle(
                original_img, (point_x, point_y), point_radius, point_color, thickness
            )
            # 绘制手部连线
            xi, yi = map(int, self.get_img_pos(name, i)[:2])
            cv2.line(original_img, (xp, yp), (xi, yi), point_color, thickness)
            if i % 4 == 0:
                xp, yp = x0, y0
            else:
                xp, yp = xi, yi
        return original_img

    def draw_norm_hand(
        self,
        name: str,
        original_img: np.ndarray,
        padx: int = 10,
        pady: int = 10,
        point_radius: int = 4,
        hand_colorBGR: tuple[int, int, int] = (255, 255, 255),
        thickness: int = 3,
        base_rgb: int = 4,
    ) -> np.ndarray | None:
        """绘制归一化后的手部坐标"""
        hand_img = self.get_hand_img(name, original_img, padx, pady)
        # 保证获取的图片有效
        if hand_img is None:
            return None
        else:  # 复制一张图,防止绘制到同一张图片上
            hand_img = hand_img.copy()
        img_h, img_w = hand_img.shape[:2]
        x0, y0, _ = self.get_norm_pos_to_img(name, 0, img_w, img_h, padx, pady)
        xp, yp = x0, y0
        for i in range(21):
            # 绘制关键点
            point_x, point_y, point_z = self.get_norm_pos_to_img(
                name, i, img_w, img_h, padx, pady
            )
            # 用颜色深度表示z轴
            point_color = tuple(
                map(
                    lambda c: int((c / base_rgb) * (1 + (base_rgb - 1) * point_z)),
                    hand_colorBGR,
                )
            )
            cv2.circle(
                hand_img, (point_x, point_y), point_radius, point_color, thickness
            )
            # 绘制手部连线
            xi, yi, _ = self.get_norm_pos_to_img(name, i, img_w, img_h, padx, pady)
            cv2.line(hand_img, (xp, yp), (xi, yi), point_color, thickness)
            if i % 4 == 0:
                xp, yp = x0, y0
            else:
                xp, yp = xi, yi
        return hand_img

    def draw_norm_on_bg(
        self,
        name: str,
        bg_img_w: int = 300,
        bg_img_h: int = 300,
        padx: int = 30,
        pady: int = 30,
        point_radius: int = 4,
        bg_colorBGR: tuple[int, int, int] = (0, 0, 0),
        hand_colorBGR: tuple[int, int, int] = (255, 255, 255),
        thickness: int = 3,
        base_rgb: int = 4,
    ) -> np.ndarray:
        """绘制归一化后的手部坐标"""
        # 创建新的背景图
        bg_img = np.ones((bg_img_h, bg_img_w, 3), dtype=np.uint8)
        bg_img[:, :, 0] *= bg_colorBGR[0]  # 为背景图上色
        bg_img[:, :, 1] *= bg_colorBGR[1]
        bg_img[:, :, 2] *= bg_colorBGR[2]
        x0, y0, _ = self.get_norm_pos_to_img(name, 0, bg_img_w, bg_img_h, padx, pady)
        xp, yp = x0, y0
        for i in range(21):
            # 绘制关键点
            point_x, point_y, point_z = self.get_norm_pos_to_img(
                name, i, bg_img_w, bg_img_h, padx, pady
            )
            # 用颜色深度表示z轴
            point_color = tuple(
                map(
                    lambda c: int((c / base_rgb) * (1 + (base_rgb - 1) * point_z)),
                    hand_colorBGR,
                )
            )
            cv2.circle(bg_img, (point_x, point_y), point_radius, point_color, thickness)
            # 绘制手部连线
            xi, yi, _ = self.get_norm_pos_to_img(
                name, i, bg_img_w, bg_img_h, padx, pady
            )
            cv2.line(bg_img, (xp, yp), (xi, yi), point_color, thickness)
            if i % 4 == 0:
                xp, yp = x0, y0
            else:
                xp, yp = xi, yi
        return bg_img

    def draw_box(
        self,
        name: str,
        image: np.ndarray,
        padding: int = 10,
        colorBGR: tuple[int, int, int] = (255, 0, 0),
        thickness: int = 3,
    ):
        x0, y0, x1, y1 = self.get_hand_box(name)
        cv2.rectangle(
            image,
            (x0 - padding, y0 - padding),
            (x1 + padding, y1 + padding),
            colorBGR,
            thickness,
        )
