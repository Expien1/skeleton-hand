from collections import deque
from time import time
from typing import Generator

import numpy as np
import cv2


class Camera:
    __slots__ = "capture", "prev_time", "fps_deque"

    def __init__(
        self, camera_idx: int = 0, max_fps: int = 30, cam_w: int = 640, cam_h: int = 480
    ) -> None:
        self.capture = cv2.VideoCapture(camera_idx)  # 获取摄像头
        # 设置摄像头的打开参数,如果参数有修改才进行设置,否则跳过设置阶段
        # 设置参数操作很会大大延长打开摄像头的时间
        if max_fps != self.capture.get(cv2.CAP_PROP_FPS) and max_fps > 0:
            self.capture.set(cv2.CAP_PROP_FPS, max_fps)
        if cam_w != self.capture.get(cv2.CAP_PROP_FRAME_WIDTH) and cam_w > 0:
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, cam_w)
        if cam_h != self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT) and cam_h > 0:
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_h)
        # 相机循环标记
        self.prev_time = time()  # 记录当前一帧结束的时间
        # 用移动窗口平滑法,计算多个帧率平均数,让显示的帧率数更加稳定
        self.fps_deque = deque(maxlen=max_fps)

    @property
    def frame_width(self):
        """获取帧的宽"""
        return self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)

    @property
    def frame_height(self):
        """获取帧的高"""
        return self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def set_camera_index(self, camera_idx: int) -> None:
        """重新设置摄像头索引"""
        self.capture = cv2.VideoCapture(camera_idx)

    def read(
        self,
        hflip: bool = True,
        key_ackii: int = -1,
        delay_ms: int = 1,
    ) -> Generator[np.ndarray, None, None]:
        """使用生成器获取摄像头的当前帧"""
        while True:
            # 读取摄像头的图片
            success, image = self.capture.read()
            if not success:  # 是否读取帧成功
                break  # 读取失败则结束循环
            self.update_fps()  # 更新帧率
            # 如果hflip为True则每次返回翻转后的图片
            yield cv2.flip(image, 1) if hflip else image
            # 检查按键输入来结束循环,waitKey返回-1表示没有按键输入
            if cv2.waitKey(delay_ms) != key_ackii:
                break  # 结束循环

    @property
    def shape(self) -> tuple[int, int]:
        """获取当前摄像头打开的宽高"""
        cam_w = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        cam_h = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return cam_w, cam_h

    @property
    def fps(self) -> int:
        """计算当前帧率"""
        # 用之前的帧率的平均数来代替实际帧率,显示更加稳定
        fps = int(sum(self.fps_deque) / len(self.fps_deque))
        return fps

    def update_fps(self):
        """更新帧率,记录每一帧之间的时间"""
        cur_time = time()  # 获取当前时间
        delta_time = cur_time - self.prev_time  # 计算两帧之间的用时
        self.prev_time = cur_time  # 更新最后时间
        fps = int(1 / delta_time)  # 每帧的用时的倒数就是帧率
        self.fps_deque.append(fps)  # 存入队列用于显示稳定帧率

    def draw_fps(
        self,
        image: np.ndarray,
        location: tuple[int, int] = (10, 70),
        color: tuple[int, int, int] = (255, 0, 0),
    ):
        """将帧率绘制到图片上"""
        # 将帧率写到图片上
        cv2.putText(image, str(self.fps), location, cv2.FONT_HERSHEY_PLAIN, 3, color, 3)

    def __del__(self):
        """销毁当前实例后,释放摄像头"""
        self.capture.release()

    def gaussian_box_mask(
        self,
        img: np.ndarray,
        box_ls: list[tuple[int, int, int, int]],
        kernel_size: int = 51,
        padding: int = 10,
    ):
        """对图片中box矩形框外的图片进行模糊效果"""
        # 对图片进行大核的高斯滤波来实现模糊效果
        blurred = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
        mask = np.zeros_like(img)  # 创建初始的掩码
        for box in box_ls:  # 遍历传入的box_ls来创建掩码
            x0, y0, x1, y1 = box
            mask[y0 - padding : y1 + padding, x0 - padding : x1 + padding, :] = 255
        # 应用掩码,只有掩码为黑色的区域应用模糊效果
        img[np.where(mask == (0, 0, 0))] = blurred[np.where(mask == (0, 0, 0))]
        return img

    def median_box_mask(
        self,
        img: np.ndarray,
        box_ls: list[tuple[int, int, int, int]],
        kernel_size: int = 51,
        padding: int = 10,
    ):
        """对图片中box矩形框外的图片进行模糊效果"""
        # 对图片进行大核的中值滤波来实现模糊效果
        blurred = cv2.medianBlur(img, kernel_size)
        mask = np.zeros_like(img)  # 创建初始的掩码
        for box in box_ls:  # 遍历传入的box_ls来创建掩码
            x0, y0, x1, y1 = box
            mask[y0 - padding : y1 + padding, x0 - padding : x1 + padding, :] = 255
        # 应用掩码,只有掩码为黑色的区域应用模糊效果
        img[np.where(mask == (0, 0, 0))] = blurred[np.where(mask == (0, 0, 0))]
        return img

    def color_box_mask(
        self,
        img: np.ndarray,
        box_ls: list[tuple[int, int, int, int]],
        mask_color: tuple[int, int, int] = (0, 0, 0),
        padding: int = 10,
    ):
        """对图片中box矩形框外的图片进行遮挡效果"""
        # 创建纯色的遮挡图
        shadowed = np.ones_like(img)
        shadowed[:, :, 0] *= mask_color[2]  # 为背景图上色
        shadowed[:, :, 1] *= mask_color[1]
        shadowed[:, :, 2] *= mask_color[0]
        mask = np.zeros_like(img)  # 创建初始的掩码
        for box in box_ls:  # 遍历传入的box_ls来创建掩码
            x0, y0, x1, y1 = box
            mask[y0 - padding : y1 + padding, x0 - padding : x1 + padding, :] = 255
        # 应用掩码,只有掩码为黑色的区域应用遮挡效果
        img[np.where(mask == (0, 0, 0))] = shadowed[np.where(mask == (0, 0, 0))]
        return img


if __name__ == "__main__":
    t1 = time()
    cam = Camera()
    print(cam.shape)
    print(time() - t1)
    for img in cam.read():
        cam.draw_fps(img)
        cv2.imshow("test", img)
