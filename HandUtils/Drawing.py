import numpy as np
import cv2

from ..HandData.OneHand import OneHand


class HandImage:
    __slots__ = "data", "_size", "_padding"

    def __init__(self, data: np.ndarray, padx: int = 0, pady: int = 0) -> None:
        self.data: np.ndarray = data
        self._size: np.ndarray = np.array((self.width, self.height))
        self._padding: np.ndarray = np.array((padx, pady))

    @property
    def width(self) -> int:
        return self.data.shape[1]

    @property
    def height(self) -> int:
        return self.data.shape[0]

    @property
    def size(self) -> np.ndarray:
        return self._size

    @property
    def padding(self) -> np.ndarray:
        return self._padding

    def calc_norm2img_pos(self, norm_pos: np.ndarray) -> np.ndarray:
        if len(norm_pos.shape) == 1 and norm_pos.shape[0] > 2:
            norm_pos = norm_pos[:2]
        img_pos = (norm_pos * (self._size - (2 * self._padding))) + self._padding
        return img_pos.astype(np.int32)


class HandBackground(HandImage):
    __slots__ = ()

    def __init__(
        self,
        width: int,
        height: int,
        color: tuple[int, int, int] = (0, 0, 0),
        padx: int = 0,
        pady: int = 0,
    ) -> None:
        # 创建纯色背景图
        data = np.empty((height, width, 3), dtype=np.uint8)
        data[:, :, :] = color
        super().__init__(data, padx, pady)


class HandDrawing:
    __slots__ = (
        "one_hand",
        "point_radius",
        "point_color",
        "point_thickness",
        "line_color",
        "line_thickness",
        "raw_img",
    )

    def __init__(
        self,
        one_hand: OneHand,
    ):
        self.one_hand = one_hand
        # 绘制手部关键点参数
        self.point_radius: int = 4
        self.point_color: tuple[int, int, int] = (255, 255, 255)
        self.point_thickness: int = 3
        # 绘制手部关键点连线参数
        self.line_color: tuple[int, int, int] = (255, 255, 255)
        self.line_thickness: int = 3
        # 记录帧图像
        self.raw_img: None | np.ndarray = None

    def update(self, raw_img: np.ndarray):
        """更新帧图像"""
        self.raw_img = raw_img

    def reset(self):
        """重置帧图像,将帧图像设为None"""
        self.raw_img = None

    def set_param(
        self,
        *,
        point_radius: int = 4,
        point_color: tuple[int, int, int] = (255, 255, 255),
        point_thickness: int = 3,
        line_color: tuple[int, int, int] = (255, 255, 255),
        line_thickness: int = 3,
    ):
        """设置手部绘制相关参数
        Args:
            point_radius: 关键点半径
            point_color: 关键点颜色
            point_thickness: 关键点粗细
            line_color: 关键点连接线的线条颜色
            line_thickness: 关键点连接线的粗细
        """
        # 绘制手部关键点参数
        self.point_radius: int = point_radius
        self.point_color: tuple[int, int, int] = point_color
        self.point_thickness: int = point_thickness
        # 绘制手部关键点连线参数
        self.line_color: tuple[int, int, int] = line_color
        self.line_thickness: int = line_thickness

    def set_point(
        self,
        *,
        radius: int = 4,
        color: tuple[int, int, int] = (255, 255, 255),
        thickness: int = 3,
    ):
        """设置手部关键点的相关参数
        Args:
            radius: 关键点半径
            color: 关键点颜色
            thickness: 关键点粗细
        """
        # 绘制手部关键点参数
        self.point_radius: int = radius
        self.point_color: tuple[int, int, int] = color
        self.point_thickness: int = thickness

    def set_line(
        self,
        *,
        color: tuple[int, int, int] = (255, 255, 255),
        thickness: int = 3,
    ):
        """设置关键点连接线的相关参数
        Args:
            color: 关键点连接线的线条颜色
            thickness: 关键点连接线的粗细
        """
        # 绘制手部关键点连线参数
        self.line_color: tuple[int, int, int] = color
        self.line_thickness: int = thickness

    def draw_hand(
        self,
        *,
        point_radius: int = 4,
        point_color: tuple[int, int, int] = (255, 255, 255),
        point_thickness: int = 3,
        line_color: tuple[int, int, int] = (255, 255, 255),
        line_thickness: int = 3,
    ) -> None | np.ndarray:
        """在原图中绘制手部,可同时设置手部绘制相关参数
        Args:
            point_radius: 关键点半径
            point_color: 关键点颜色
            point_thickness: 关键点粗细
            line_color: 关键点连接线的线条颜色
            line_thickness: 关键点连接线的粗细
        """
        # 检查是否存在有帧图像
        if self.raw_img is None:
            return None
        # 设置绘制相关参数
        self.set_param(
            point_radius=point_radius,
            point_color=point_color,
            point_thickness=point_thickness,
            line_color=line_color,
            line_thickness=line_thickness,
        )
        # 绘制
        x0, y0 = self.one_hand.raw_pos[0, :2].astype(np.int32)
        xp, yp = x0, y0
        for i in range(21):
            # 绘制关键点
            xi, yi = self.one_hand.raw_pos[i, :2].astype(np.int32)
            # 用颜色深度表示z轴
            point_z = self.one_hand.norm_pos[i, 2]
            z_color = tuple(
                map(
                    # lambda c: int((c / base_rgb) * (1 + (base_rgb - 1) * point_z)),
                    lambda c: int((c / 5) * (1 + 4 * point_z)),
                    self.point_color,
                )
            )
            cv2.circle(
                self.raw_img, (xi, yi), self.point_radius, z_color, self.point_thickness
            )
            # 绘制手部连线
            cv2.line(self.raw_img, (xp, yp), (xi, yi), z_color, self.line_thickness)
            if i % 4 == 0:
                xp, yp = x0, y0
            else:
                xp, yp = xi, yi
        return self.raw_img

    def get_hand_img(self, *, padx: int = 0, pady: int = 0) -> None | np.ndarray:
        """截取并返回对应名字的手部图片
        Args:
            padx: 横坐标方向的填充像素大小
            pady: 纵坐标方向的填充像素大小
        """
        # 检查是否存在有帧图像
        if self.raw_img is None:
            return None
        x0, y0, x1, y1 = self.one_hand.box  # 获取手部矩形框
        x0 -= padx  # 计算边缘
        x1 += padx
        y0 -= pady
        y1 += pady
        # 保证xxyy都为正整数
        x0, y0, x1, y1 = map(lambda x: x if x >= 0 else 0, (x0, y0, x1, y1))
        if x0 > x1 or y0 > y1:
            raise ValueError("It can't cut out a valid picture.")
        return self.raw_img[y0:y1, x0:x1, :].copy()

    def draw_norm_on_bg(self, bg_img: HandImage) -> np.ndarray:
        """绘制归一化后的手部坐标
        Args:
            bg_img: 背景图片,需要传入HandImage实例
        """
        x0, y0 = bg_img.calc_norm2img_pos(self.one_hand.norm_pos[0, :2])
        xp, yp = x0, y0
        for i in range(21):
            # 绘制关键点
            xi, yi = bg_img.calc_norm2img_pos(self.one_hand.norm_pos[i, :2])
            # 用颜色深度表示z轴
            point_z = self.one_hand.norm_pos[i, 2]
            z_color = tuple(
                map(
                    lambda c: int((c / 5) * (1 + 4 * point_z)),
                    self.point_color,
                )
            )
            cv2.circle(
                bg_img.data, (xi, yi), self.point_radius, z_color, self.point_thickness
            )
            # 绘制手部连线
            xi, yi = bg_img.calc_norm2img_pos(self.one_hand.norm_pos[i, :2])
            cv2.line(bg_img.data, (xp, yp), (xi, yi), z_color, self.point_thickness)
            if i % 4 == 0:
                xp, yp = x0, y0
            else:
                xp, yp = xi, yi
        return bg_img.data

    def draw_box(
        self,
        box_padx: int = 10,
        box_pady: int = 10,
        box_color: tuple[int, int, int] = (255, 0, 0),
        box_thickness: int = 3,
    ) -> None | np.ndarray:
        """在原图中绘制框住手部的矩形框
        Args:
            box_padx: 矩形框的横坐标方向的填充像素大小
            box_pady: 矩形框的纵坐标方向的填充像素大小
            box_color: 矩形框的线条颜色
            box_thickness: 矩形框的线条粗细
        """
        if self.raw_img is None:
            return None
        x0, y0, x1, y1 = self.one_hand.box
        cv2.rectangle(
            self.raw_img,
            (x0 - box_padx, y0 - box_pady),
            (x1 + box_padx, y1 + box_pady),
            box_color,
            box_thickness,
        )
        return self.raw_img


def draw_hand_data(
    data: np.ndarray,
    bg_img_w: int = 300,
    bg_img_h: int = 300,
    bg_colorBGR: tuple[int, int, int] = (0, 0, 0),
    padx: int = 30,
    pady: int = 30,
    point_radius: int = 4,
    point_colorBGR: tuple[int, int, int] = (255, 255, 255),
    point_thickness: int = 3,
    base_rgb: int = 4,
) -> np.ndarray:
    """根据一维手部数据绘制归一化后的手部坐标
    Args:
        data: 一维手部关键点数据
        bg_img_w: 纯色背景图片的宽
        bg_img_h: 纯色背景图片的高
        bg_colorBGR: 背景图片颜色
        padx: 横坐标方向的填充像素大小
        pady: 纵坐标方向的填充像素大小
        point_radius: 手部关键点的半径
        point_colorBGR: 手部关键点的颜色
        point_thickness: 手部关键点的粗细
        base_rgb: z轴基础颜色占比
    """
    # 创建纯色背景图
    bg_img = np.ones((bg_img_h, bg_img_w, 3), dtype=np.uint8)
    bg_img[:, :, :] = bg_colorBGR
    # data的前63列为手部数据的一维数组
    x0, y0 = data[:2]
    x0 = int(x0 * (bg_img_w - (2 * padx)) + padx)
    y0 = int(y0 * (bg_img_h - (2 * pady)) + pady)
    xp, yp = x0, y0
    for i in range(21):
        # 绘制关键点
        point_x, point_y, point_z = data[i * 3 : i * 3 + 3]
        point_x = int(point_x * (bg_img_w - (2 * padx)) + padx)
        point_y = int(point_y * (bg_img_h - (2 * pady)) + pady)
        # 用颜色深度表示z轴
        point_color = tuple(
            map(
                lambda c: int((c / base_rgb) * (1 + (base_rgb - 1) * point_z)),
                point_colorBGR,
            )
        )
        cv2.circle(
            bg_img, (point_x, point_y), point_radius, point_color, point_thickness
        )
        # 绘制手部连线
        xi, yi = data[i * 3 : i * 3 + 2]
        xi = int(xi * (bg_img_w - (2 * padx)) + padx)
        yi = int(yi * (bg_img_h - (2 * pady)) + pady)
        cv2.line(bg_img, (xp, yp), (xi, yi), point_color, point_thickness)
        if i % 4 == 0:
            xp, yp = x0, y0
        else:
            xp, yp = xi, yi
    return bg_img
