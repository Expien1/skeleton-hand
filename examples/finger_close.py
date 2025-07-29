"""检测手指并拢"""

# 引入所需的类和包
import cv2
from skhand import HandInput, Camera
# from skhand.HandUtils.Camera import Camera
# from skhand.HandInput import HandInput


hi = HandInput(["h0"])  # 创建只检测一只手的手部输入实例,并将手命名为"h0"
camera = Camera()  # 创建摄像头
for img in camera.read():  # 开启并读取摄像头图片
    # 使用名字列表来检查是否有检测到手部
    detected_hand_names = hi.run(img)  # 输入被检测的图片

    if detected_hand_names:  # 如果detected_hand_names列表非空,有检测到手部"h0"
        for i in range(4):
            # 用gestrue.finger_close获取手指并拢状态的输出
            if hi["h0"].gestrue.finger_close[i]:
                px, py = hi["h0"].base.img_pos((i + 1) * 4)
                cv2.circle(img, (px, py), 8, (50, 50, 255), 3)
                px, py = hi["h0"].base.img_pos((i + 2) * 4)
                cv2.circle(img, (px, py), 8, (50, 50, 255), 3)

    camera.draw_fps(img)
    cv2.imshow("finger close", img)
