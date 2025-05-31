import numpy as np
import cv2
import mediapipe as mp

from .VisualHandDetector import VisualHandDetector
from .HandsMatchers import HandsMatcher
from ..HandData.OneHand import OneHand


class MediaPipeHandDetector(VisualHandDetector):
    __slots__ = "hands_name_ls", "_detector", "hands_matcher", "matched_counter"

    def __init__(
        self,
        hands_name_ls: list[str],
        hands_matcher: type[HandsMatcher],
        *,
        static_image_mode: bool = False,
        min_detect_confi: float = 0.8,
        min_track_confi: float = 0.6,
    ):
        """MediaPipe手部关键点检测器
        Args:
            hands_name_ls: 需要被检测到的手部名字列表
            hands_matcher: 手部匹配器,用于匹配/追踪多只手部的名字
                以下为手部检测器VisualHandDetector的参数
                static_image_mode: 使用静态检测模式,False为使用动态
                min_detect_confi: 检测手部的置信度
                min_track_confi: 跟踪手部的置信度
        """
        super().__init__(hands_name_ls, hands_matcher)
        self.hands_name_ls: list[str] = hands_name_ls
        # 初始化手部检测器
        self._detector = mp.solutions.hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=len(hands_name_ls),
            min_detection_confidence=min_detect_confi,
            min_tracking_confidence=min_track_confi,
        )
        # 创建手部匹配器
        self.hands_matcher: HandsMatcher = hands_matcher(hands_name_ls)
        # 创建手部匹配计时器,用于记录手部识别了多少帧,初始值为10
        self.matched_counter: dict[str, int] = {n: 30 for n in hands_name_ls}

    def detect(self, image: np.ndarray, hands_dict: dict[str, OneHand]) -> list[str]:
        """使用MediaPipe检测手部关键点,返回成功检测到的手部的名称
        Args:
            image: ndarray格式的被检测图片
            hands_dict: 手部名字和手部数据实例的字典,名字为键,数据实例为值
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
            # 计算手部相似度矩阵,行是检测到的手部编号(索引),列是需要配对的手部名称
            r_num, c_num = len(multi_hand_landmarks), len(self.hands_name_ls)
            if r_num > c_num:  # 检测的手部数量过多
                return []
            sim_matix: np.ndarray = np.empty((r_num, c_num), dtype=np.float16)
            # 计算相似度,填入对应矩阵的位置
            for detected_idx in range(r_num):
                for name_idx, name in enumerate(self.hands_name_ls):
                    one_hand = hands_dict[name]
                    landmark = multi_hand_landmarks[detected_idx].landmark
                    handedness = multi_handedness[detected_idx]
                    # 1. 手腕关键点距离相似度
                    # 计算下一帧预测位置,使用五指指尖和手腕的关键点位置来计算相似度
                    pred_ipos0 = one_hand.raw_pos[0, :2] * 2 - one_hand.last_rpos[0, :2]
                    dist_sim = np.linalg.norm(
                        (pred_ipos0 / image.shape[:2] - (landmark[0].x, landmark[0].y)),
                        ord=1,
                    )
                    # 2. 左右手标签相似度
                    # 根据左右手标签对相似度乘以比例
                    hand_side_sim = 10
                    if one_hand.hand_side == handedness.classification[0].label:
                        hand_side_sim = 0
                    elif one_hand.hand_side == "Unknown":
                        hand_side_sim = 20
                    # 3. 匹配的帧的数量
                    # 优先匹配之前匹配过的手部
                    matched_num = self.matched_counter[name]
                    # 写入对应的相似度
                    cur_similarity = 10 * dist_sim + hand_side_sim + matched_num
                    sim_matix[detected_idx, name_idx] = cur_similarity
            # 计算得到匹配手部字典
            matched_dict: dict[int, str] = self.hands_matcher.run(sim_matix)

            # 更新匹配的帧计数器
            for name in self.hands_name_ls:
                if name in matched_dict.values():
                    if self.matched_counter[name] > 0:
                        self.matched_counter[name] -= 1  # 匹配到且计数器当前大于0就减1
                else:
                    self.matched_counter[name] = 30  # 重置计数器

            # 根据匹配结果处理对应的手部数据
            img_h, img_w, _ = image.shape  # 获取图片宽高,用于处理归一化后的关键点数据
            for detected_idx, name in matched_dict.items():
                one_hand = hands_dict[name]
                # 记录该手部是左手还是右手
                one_hand.hand_side = (
                    multi_handedness[detected_idx].classification[0].label
                )
                # 记录上一帧的图片关键点位置
                one_hand.last_rpos[:, :] = one_hand.raw_pos
                # 获取并处理检测到的手部数据
                for id, landmark in enumerate(
                    multi_hand_landmarks[detected_idx].landmark
                ):
                    # 将归一化的手部关键点转化为图片中的位置
                    x = int(landmark.x * img_w)
                    y = int(landmark.y * img_h)
                    one_hand.raw_pos[id, :] = x, y, landmark.z
                # 将原始数据处理完毕后,重置所有数据计算标志
                one_hand.reset_all_flags()
                # 更新检测到的手部的名字
                detected_name_ls.append(name)
        else:
            # 重置匹配的帧计数器
            for name in self.hands_name_ls:
                self.matched_counter[name] = 30  # 重置计数器

        return detected_name_ls
