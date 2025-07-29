import cv2  # 用于绘制手部名字
from skhand import HandInput, Camera  # 引入所需的类
# 可以这样引用
# from skhand.HandInput import HandInput
# from skhand.HandUtils.Camera import Camera


hi = HandInput(["hand1", "hand2"])  # 创建HandInput实例
camera = Camera()  # 创建Camera实例,用于获取摄像头图像(也可以使用opencv来获取)
for img in camera.read():  # 调用read()方法,是生成器函数,返回每帧的摄像头图像
    # 这里通过获取检测到的手部名字列表来判断检测到了哪只手部
    detected_hand_names = hi.run(img)  # 运行手部检测器,返回检测到的手部的名字
    if detected_hand_names:  # 如果detected_hand_names列表非空,有检测到手部
        # 遍历检测到的手部名字,只调用被检测到的手部
        for hand_name in detected_hand_names:
            # 绘制对应名字的手部关键点骨架在摄像头的帧图片上
            hi[hand_name].drawing.draw_hand()
            # 获取手腕在摄像头画面上的坐标
            px, py = hi[hand_name].base.img_pos(0)  # 索引0为手腕关键点
            box_color = (0, 255, 0)  # 框的默认颜色
            # 绘制不同颜色的名字在不同的手腕关键点上
            if hand_name == "hand1":
                cv2.putText(img, hand_name, (px, py), 1, 2, (255, 0, 0), 2)
                box_color = (255, 0, 0)
            if hand_name == "hand2":
                cv2.putText(img, hand_name, (px, py), 1, 2, (0, 0, 255), 2)
                box_color = (0, 0, 255)
            # 绘制不同颜色的框在手上
            hi[hand_name].drawing.draw_box(box_color=box_color)
    camera.draw_fps(img)  # 绘制帧率
    cv2.imshow("matcher", img)  # 显示图片
