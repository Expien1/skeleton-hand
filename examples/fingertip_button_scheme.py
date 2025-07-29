import cv2

# 引入手部输入类和摄像头类以及手指按钮交互方案
from skhand import HandInput, Camera, FingertipButtonScheme

# 也可以使用下面的方式引入相关的类
# from skhand.HandInput import HandInput
# from skhand.HandUtils.Camera import Camera
# from skhand.HandInputSchemes.FingertipButtonScheme import FingertipButtonScheme

# 以下是从手部数据可视化模块里引入背景类,用于定义可视化背景的大小,方便绘制
from skhand.HandUtils.Drawing import HandBackground


hi = HandInput(["h0"])  # 创建手部输入实例
# 注册手指指尖按钮交互方案,就可以不用显式的调用交互方案的update方案来更新
# 交互方案的输入参数为:手部输入实例,对应的手部名字,手指索引(本例为食指)
hi.schemes["if-btn1"] = FingertipButtonScheme(hi, "h0", 0)

# 主循环
camera = Camera()
for img in camera.read():
    if hi.run(img):  # 判断是否有检测到有手部
        # 截取出手部所在的图片,pad参数表示边缘扩大12像素
        hand_img = hi["h0"].drawing.draw_hand_only(padx=12, pady=12)
        # 同时绘制归一化后的手部数据进行可视化,要先定义一个背景实例
        bg = HandBackground(300, 300, padx=12, pady=12)
        norm_img = hi["h0"].drawing.draw_norm_hand(bg)

        # 将交互方案的效果绘制到归一化手部数据图片上进行可视化
        # 获取大拇指指尖关键点坐标
        ift_point = hi["h0"].base.norm_pos(4)[:2]
        # 将坐标转换到可视化的背景图片上
        nift_point = bg.calc_norm2img_pos(ift_point)
        # 效果相当于应用下面的公式来计算新的坐标点
        # nift_point = ift_point * (300 - (2 * 12)) + 12

        # 调用手指指尖按钮交互方案的属性获取所需的交互数据
        if hi.schemes["if-btn1"].is_long_press:  # 判断食指指尖是否长按
            # 用opencv绘制红色的圆圈代表长按
            cv2.circle(norm_img, tuple(map(int, nift_point)), 10, (70, 70, 255), 5)
        elif hi.schemes["if-btn1"].is_short_press:  # 判断食指指尖是否短按
            # 用opencv绘制蓝色的圆圈代表短按
            cv2.circle(norm_img, tuple(map(int, nift_point)), 10, (255, 70, 70), 5)

        # 最后绘制可视化图片
        if hand_img is not None:
            camera.draw_fps(hand_img)
            cv2.imshow("hand_img", hand_img)
        cv2.imshow("norm_img", norm_img)
