import numpy as np
import cv2

from ..HandInput import HandInput


class HandDrawing:
    def __init__(self, hand_input: HandInput):
        self.hand_input = hand_input

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
        x0, y0 = map(int, self.hand_input.get_img_pos(name, 0)[:2])
        xp, yp = x0, y0
        for i in range(21):
            # 绘制关键点
            point_x, point_y = map(int, self.hand_input.get_img_pos(name, i)[:2])
            point_z = self.hand_input.get_img_pos(name, i)[2]
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
            xi, yi = map(int, self.hand_input.get_img_pos(name, i)[:2])
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
        hand_img = self.hand_input.get_hand_img(name, original_img, padx, pady)
        # 保证获取的图片有效
        if hand_img is None:
            return None
        else:  # 复制一张图,防止绘制到同一张图片上
            hand_img = hand_img.copy()
        img_h, img_w = hand_img.shape[:2]
        x0, y0, _ = self.hand_input.get_norm_pos_to_img(
            name, 0, img_w, img_h, padx, pady
        )
        xp, yp = x0, y0
        for i in range(21):
            # 绘制关键点
            point_x, point_y, point_z = self.hand_input.get_norm_pos_to_img(
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
            xi, yi, _ = self.hand_input.get_norm_pos_to_img(
                name, i, img_w, img_h, padx, pady
            )
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
        x0, y0, _ = self.hand_input.get_norm_pos_to_img(
            name, 0, bg_img_w, bg_img_h, padx, pady
        )
        xp, yp = x0, y0
        for i in range(21):
            # 绘制关键点
            point_x, point_y, point_z = self.hand_input.get_norm_pos_to_img(
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
            xi, yi, _ = self.hand_input.get_norm_pos_to_img(
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
        x0, y0, x1, y1 = self.hand_input.get_hand_box(name)
        cv2.rectangle(
            image,
            (x0 - padding, y0 - padding),
            (x1 + padding, y1 + padding),
            colorBGR,
            thickness,
        )
