import cv2
from skhand import HandInput, Camera, FingertipButtonScheme, ThumbJoystickScheme
from skhand.HandUtils.Drawing import HandBackground


hi = HandInput(["h0"])  # 创建手部输入实例
# 使用拇指摇杆方案需要输入一个指尖按钮交互方案的实例作为参数
hi.schemes["fbtn0"] = FingertipButtonScheme(hi, "h0", 0)  # 食指指尖按钮
hi.schemes["tjoy0"] = ThumbJoystickScheme(hi, "h0", hi.schemes["fbtn0"])

camera = Camera()
for img in camera.read():
    if hi.run(img):
        # 截取出手部所在的图片,pad参数表示边缘扩大12像素
        hand_img = hi["h0"].drawing.draw_hand_only(padx=12, pady=12)
        # 同时绘制归一化后的手部数据进行可视化,要先定义一个背景实例
        bg = HandBackground(300, 300, padx=12, pady=12)
        norm_img = hi["h0"].drawing.draw_norm_hand(bg)

        # 调用手指摇杆交互方案
        if hi.schemes["tjoy0"].is_activate:
            # 还可以调用摇杆的方向向量
            joy_nvec = hi.schemes["tjoy0"].norm_vec
            # print("摇杆方向向量:", joy_nvec)
            # 注意:手指摇杆返回的坐标点是相对于手腕的归一化坐标
            # 所以要可视化前得先转换回以左上角为原点的归一化坐标
            wrist_point = hi["h0"].base.norm_pos(0)
            fixed_point = hi.schemes["tjoy0"].fixed_point + wrist_point
            thumb_point = hi["h0"].base.norm_pos(4)

            # 可视化
            fp = bg.calc_norm2img_pos(fixed_point)
            tp = bg.calc_norm2img_pos(thumb_point)
            fp = tuple(map(int, fp))
            tp = tuple(map(int, tp))
            cv2.circle(norm_img, fp, 8, (50, 50, 250), 3)
            cv2.line(norm_img, fp, tp, (250, 50, 50), 4)

        # 最后绘制可视化图片
        if hand_img is not None:
            camera.draw_fps(hand_img)
            cv2.imshow("hand_img", hand_img)
        cv2.imshow("norm_img", norm_img)
