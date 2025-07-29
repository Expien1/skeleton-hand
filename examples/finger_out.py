"""检测手指伸出"""

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
        for i, is_out in enumerate(hi["h0"].gestrue.finger_out):
            # 如果手指伸出就绘制一个圆圈在对应的指尖上
            if is_out:
                # 获取对应指尖的xy坐标
                px, py = hi["h0"].base.img_pos((i + 1) * 4)
                cv2.circle(img, (px, py), 10, (50, 50, 255), 5)

    camera.draw_fps(img)
    cv2.imshow("finger out", img)


# 对于只检测一只手部的情况可以不使用手部名字列表来检测是否有检测到手部
# 下面的代码是没有使用手部名字列表的示例,两段代码效果一样
"""
hi = HandInput(["h0"])  # 创建只检测一只手的手部输入实例,并将手命名为"h0"
camera = Camera()  # 创建摄像头
for img in camera.read():  # 开启并读取摄像头图片
    hi.run(img)  # 输入被检测的图片

    # gestrue函数没有检测到"h0",则返回None
    h0_gestrue = hi.gestrue("h0")
    # 保证了下面的代码都是能检测到手部"h0"的
    if h0_gestrue is not None:
        # 遍历5根手指是否有伸出
        for i, is_out in enumerate(h0_gestrue.finger_out):
            # 如果手指伸出就绘制一个圆圈在对应的指尖上
            if is_out:
                # 获取对应指尖的xy坐标
                px, py = hi["h0"].base.img_pos((i + 1) * 4)
                cv2.circle(img, (px, py), 10, (50, 50, 255), 5)

    camera.draw_fps(img)
    cv2.imshow("finger out", img)
"""
