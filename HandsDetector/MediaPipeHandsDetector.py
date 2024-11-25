import numpy as np
import cv2
import mediapipe as mp

from .VisualHandsDetector import VisualHandsDetector
from ..OneHand import OneHand


class MediaPipeHandsDetector(VisualHandsDetector):
    __slots__ = "_detector"

    def __init__(
        self,
        *,
        max_num_hands: int = 2,
        static_image_mode: bool = False,
        min_detect_confi: float = 0.8,
        min_track_confi: float = 0.6,
    ):
        """
        static_image_mode:使用静态检测模式,False为使用动态
        max_num_hands: 最大检测到的手的数量
        min_detect_confi: 检测手部的置信度
        min_track_confi: 跟踪手部的置信度
        """
        super().__init__(max_num_hands)
        # 初始化手部检测器
        self._detector = mp.solutions.hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detect_confi,
            min_tracking_confidence=min_track_confi,
        )

    def detect(self, image: np.ndarray, hands_dict: dict[str, OneHand]) -> list[str]:
        """
        具体实现使用MediaPipe来检测手部关键点
        检测成功,则返回成功检测到的手部的名称
        没有检测到手部或检测失败,则返回空列表
        """
        # 用一个列表保存检测到的手部的名字
        detected_name_ls = []
        # MediaPipe检测手部关键点需要转换为RGB
        imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 检测手部关键点
        results = self._detector.process(imgRGB)
        multi_hand_landmarks = results.multi_hand_landmarks
        multi_handedness = results.multi_handedness
        # 判断是否检测到手部
        if multi_hand_landmarks:
            img_h, img_w, _ = image.shape  # 获取图片宽高,用于处理归一化后的关键点数据
            multi_hand_landmarks.reverse()  # 反转检测到的关键点数据,保证先来的顺序
            # 遍历检测到的原始手部数据并进行处理
            for name, landmarks, handedness in zip(
                hands_dict.keys(), multi_hand_landmarks, multi_handedness
            ):
                # 记录该手部是否为左手
                hands_dict[name].is_left = (
                    True if handedness.classification[0].label == "Left" else False
                )
                # 处理检测到的手部关键点数据
                self.process_data(hands_dict[name], landmarks, img_w, img_h)
                detected_name_ls.append(name)
        # 最后返回检测到的手部的名字
        return detected_name_ls

    def process_data(self, one_hand: OneHand, hand_landmarks, img_w: int, img_h: int):
        """
        将检测到的手部关键点数据统一为OneHand数据格式,
        同时调用update方法计算并更新其他手部相关数据,
        将检测结果存入OneHand实例里统一数据格式,
        最后调用OneHand.update()方法来计算其他数据
        """
        # 获取并处理检测到的手部数据
        for id, landmark in enumerate(hand_landmarks.landmark):
            # 将归一化的手部关键点转化为图片中的位置
            x = int(landmark.x * img_w)
            y = int(landmark.y * img_h)
            one_hand.raw_pos[id, :] = x, y, landmark.z
        # z轴用归一化坐标表示
        min_z = one_hand.raw_pos[:, 2].min()
        max_z = one_hand.raw_pos[:, 2].max()
        one_hand.raw_pos[:, 2] = (one_hand.raw_pos[:, 2] - min_z) / (max_z - min_z)
        # 将原始数据处理完毕后,调用OneHand中的update方法计算并更新所有手部数据
        one_hand.update()
        return one_hand.raw_pos
