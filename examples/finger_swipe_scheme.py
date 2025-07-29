import cv2
from skhand import HandInput, Camera, FingerSwipeScheme
from skhand.HandUtils.Drawing import HandBackground


hi = HandInput(["h0"])  # 创建手部输入实例
# 注册食指指尖滑动的交互方案,输入参数为:手部输入实例,对应的手部名字,手指索引(本例为食指指尖)
hi.schemes["fs"] = FingerSwipeScheme(hi, "h0", 8)

# 定义两个变量用于存储滑动点的坐标和颜色,保留滑动痕迹
fsp = (0, 0)
fep = (0, 0)
colors = [(100, 150, 100), (100, 200, 100), (100, 250, 100)]

# 主循环
camera = Camera()
for img in camera.read():
    if hi.run(img):  # 判断是否有检测到有手部
        # 截取出手部所在的图片,pad参数表示边缘扩大12像素
        hand_img = hi["h0"].drawing.draw_hand_only(padx=12, pady=12)
        # 同时绘制归一化后的手部数据进行可视化,要先定义一个背景实例
        bg = HandBackground(300, 300, padx=12, pady=12)
        norm_img = hi["h0"].drawing.draw_norm_hand(bg)

        # 调用手指滑动交互方案
        if hi.schemes["fs"].is_activate:  # 判断手指是否有滑动
            # 注意:手指滑动返回的坐标点是相对于手腕的归一化坐标
            # 所以要可视化前得先转换回以左上角为原点的归一化坐标
            wrist_point = hi["h0"].base.norm_pos(0)[:2]
            # 通过手指滑动交互方案的属性获取所需的数据
            start_point = hi.schemes["fs"].start_point + wrist_point
            end_point = hi.schemes["fs"].end_point + wrist_point
            # 将坐标转换到可视化的背景图片上
            fsp = bg.calc_norm2img_pos(start_point)
            fep = bg.calc_norm2img_pos(end_point)
            fsp = tuple(map(int, fsp))
            fep = tuple(map(int, fep))
        # 绘制滑动轨迹
        cv2.circle(norm_img, fsp, 8, colors[0], 3)
        cv2.line(norm_img, fsp, fep, colors[1], 4)
        cv2.circle(norm_img, fep, 8, colors[2], 3)

        # 最后绘制可视化图片
        if hand_img is not None:
            camera.draw_fps(hand_img)
            cv2.imshow("hand_img", hand_img)
        cv2.imshow("norm_img", norm_img)
