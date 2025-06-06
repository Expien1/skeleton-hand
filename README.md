# skeleton-hand 🦴

**基于手部关键点检测的手势交互开发框架**

---

## 🖐️ 项目简介

**skeleton-hand** 是一个基于手部关键点检测的**手势交互开发框架**，旨在通过整合现有技术和统一手部数据接口来简化手部关键点检测技术的应用，降低开发者在手势交互领域的开发门槛。促进手势交互在游戏、VR、教育等领域的探索，让交互变得**更有趣**。

### 项目特点

- **统一手部数据接口**：提供标准化的手部数据调用接口，封装手部关键点检测技术。

- **可自定义手势**：用3种手指状态信息组合成不同的手势，只需对手指状态进行建模，即可定义出多种手势，避免了对特定手势单独建模。

- **内置交互方案**：提供指尖按钮、手指滑动、拇指摇杆3种手势交互方案。

- **可扩展**：允许自定义新的手势交互方案，允许接入不同的手部关键点检测技术。

### 效果展示

- 提供三种内置的手部交互方案，如下图分别时指尖按钮、拇指摇杆和手指滑动

<div>
    <img src="./docs/images/FingertipButton.gif" alt="指尖按钮">
    <img src="./docs/images/ThumbJoystick.gif" alt="拇指摇杆">
    <img src="./docs/images/FingerSwipe.gif" alt="手指滑动">
</div>


> 三种手部交互方案都是使用归一化后的手部数据来实现的，所以在进行交互时只会关注手的内部的运动和手指状态，而不关注整个手部在摄像头画面中的绝对位置，这样在长时间交互时就不用那么累。当然，框架里也可以调用手部关键点在摄像头画面中的绝对位置，使用哪些手部数据都取决于你要如何设计。:)

- 下面是使用手部交互方案来操控的小游戏，使用拇指摇杆来控制游戏中的坦克移动，使用指尖按钮的短按来控制坦克开炮



![交互方案应用测试](./docs/images/GestrueGame.gif)

### 源代码目录结构

```
skhand/
├── HandInput.py              # 核心手部输入管理类
├── HandData/                 # 手部数据处理模块
│   ├── FingerModels.py       # 手指状态模型封装
│   ├── Gestrue.py            # 手势识别API
│   ├── OneHand.py            # 单只手数据管理类
│   └── models/               # 手指状态模型文件夹
├── HandDetector/             # 手部关键点检测模块
│   ├── HandsMatchers.py      # 多手匹配器
│   ├── MediaPipeHandDetector.py # MediaPipe检测器实现
│   └── VisualHandDetector.py # 检测器抽象基类
├── HandInputSchemes/         # 手势交互方案模块
│   ├── FingerSwipeScheme.py  # 手指滑动方案实现
│   ├── FingertipButtonScheme.py # 指尖按钮方案实现
│   ├── ThumbJoystickScheme.py # 拇指摇杆方案实现
│   └── HandInputSchemes.py   # 交互方案抽象基类
└── HandUtils/                # 工具模块
    ├── Camera.py             # 摄像头工具
    ├── Drawing.py            # 手部数据绘制工具
    └── Filter.py             # 数据滤波工具   
```

---



## 🚀 快速开始

### 安装

```
pip install skeleton-hand
```

### 使用流程简述

1. 初始化
   - 创建HandInput实例，输入想要检测的手部名字列表，手部名字可以任意起，之后通过手部名字调用对应的数据
   - 创建Camera实例，本质上是把OpenCV的调用摄像头的功能封装成python生成器，可以用for循环调用每帧的图像
   - 若使用交互方案，则先创建Scheme交互方案的实例，然后将该实例加入HandInput实例里的schemes字典里即可
2. 主循环
   - 调用HandInput的run方法，输入摄像头图片，运行手部检测器，返回检测到的手部的名字
   - 通过HandInput实例调用所需要的手部数据，在调用手部数据时一般的流程如下：（和调用交互方案不一样）
     1) 先确定所要调用的数据是属于哪个大类（分为base基础手部数据、data展平后的手部数据、gestrue手指状态描述数据）
     2) 然后确定调用的是哪个名字的手的数据，如`hand_input.base(hand_name)`获取对应的单只手部的数据管理实例
     3) 最后再调用具体的数据方法来获取对应的数据，如`hand_input.base(hand_name).img_pos(0)`

### 具体使用示例

**1. 用简单的绘制两只手部的代码作为示例**

```python
from skhand import HandInput, Camera  					# 导入所需要的类
import cv2  											# 用于绘制手部名字

# 初始化
hi = HandInput(["hand1", "hand2"])  					# 创建HandInput实例
camera = Camera()										# 创建Camera实例,用于获取摄像头图像(可选)
# 主循环
for img in camera.read():								# 调用read()方法,是生成器函数,返回每帧的摄像头图像
    detected_hands = hi.run(img)						# 运行手部检测器,返回检测到的手部的名字
    if detected_hands:									# 如果detected_hands列表非空,有检测到手部
        for hand_name in detected_hands:				# 遍历检测到的手部名字,只调用被检测到的手部
            hi.drawing_unwrap(hand_name).draw_hand()	# 绘制对应名字的手部在摄像头的帧图片上

            # 绘制不同颜色的名字在不同的手腕关键点上
            px, py = hi.base_unwrap(hand_name).img_pos(0)  # 获取手腕在摄像头画面上的坐标
            if hand_name == "hand1":
                cv2.putText(img, hand_name, (px, py), 1, 2, (255, 0, 0), 2)
            if hand_name == "hand2":
                cv2.putText(img, hand_name, (px, py), 1, 2, (0, 0, 255), 2)

        # 绘制不同颜色的框在手上
        dr_h1 = hi.drawing("hand1")
        if dr_h1 is not None:  # 判断该手部是否有被检测到
            dr_h1.draw_box(box_color=(255, 0, 0))
        dr_h2 = hi.drawing("hand2")
        if dr_h2 is not None:
            dr_h2.draw_box(box_color=(0, 0, 255))

    camera.draw_fps(img)  				# 绘制帧率在摄像头画面的左上角
    cv2.imshow("test matcher", img)		# 使用opencv显示摄像头画面和绘制好的手部关键点
```

运行效果如下（目前的手部匹配器在手部高速移动时可能会匹配错误）

![手部名字匹配代码运行效果](./docs/images/HandsMatcher.gif)

这里简单解释一下代码，代码中的HandInput实例`hi`的`drawing`方法是用于绘制手部数据的，`drawing`方法如果调用了没有检测到的手部名字就会返回None，由于下面代码因为已经确定了使用的手部名字一定是有检测到的，所以可以用`drawing_unwrap`方法来代替，`drawing_unwrap`方法在调用了没有检测到的手部名字时会报错，不会返回None。`drawing`和`drawing_unwrap`方法的主要区别就是对于没有检测到的手部是如何处理的，`drawing`方法会返回None，而`drawing_unwrap`方法会直接报错。

```python
for img in camera.read():								# 调用read()方法,是生成器函数,返回每帧的摄像头图像
    detected_hands = hi.run(img)						# 运行手部检测器,返回检测到的手部的名字
    if detected_hands:									# 如果detected_hands列表非空,有检测到手部
        for hand_name in detected_hands:				# 遍历检测到的手部名字,只调用被检测到的手部
            hi.drawing_unwrap(hand_name).draw_hand()	# 绘制对应名字的手部在摄像头的帧图片上
```

如果使用`drawing`方法则需要判断以下返回值是否为None，即看看是否有检测到该名字的手，就像下面这段代码一样。

```python
        # 绘制不同颜色的框在手上
        dr_h1 = hi.drawing("hand1")
        if dr_h1 is not None:  # 判断该手部是否有被检测到
            dr_h1.draw_box(box_color=(255, 0, 0))
        dr_h2 = hi.drawing("hand2")
        if dr_h2 is not None:
            dr_h2.draw_box(box_color=(0, 0, 255))
```

**2. 调用拇指摇杆交互方案的示例**

下面代码主要演示如何创建并注册手部交互方案，以及如何调用交互方案的数据，由于不同的手部交互方案的输出结果都不一样，所以需要具体查看能调用哪些API。

```python
# 导入必要的类
from skhand import HandInput, Camera, FingertipButtonScheme, ThumbJoystickScheme
# 用于绘制手部数据
from skhand.HandUtils.Drawing import HandBackground
import cv2

# 初始化
hi = HandInput(["hand0"])
camera = Camera()
hi.schemes["fbtn0"] = FingertipButtonScheme(hi, "hand0", 0)  # 创建实例并加入schemes字典
hi.schemes["tjoy0"] = ThumbJoystickScheme(hi, "hand0", hi.schemes["fbtn0"])
# 主循环
for img in camera.read():
    if hi.run(img):  # 规定只检测一只手部,则可以直接判断列表是否为空
        # 如果检测到手部,则绘制手部
        dr = hi.drawing_unwrap("hand0")  # 获取该手部的绘制器
        bg = HandBackground(300, 300, padx=12, pady=12)  # 创建纯黑色的300x300的背景图
        norm_img = dr.draw_norm_hand(bg)  # 在这张背景上绘制归一化后的手部数据，返回绘制完的图片
        if hi.schemes["tjoy0"].is_activate:  # 判断拇指摇杆是否被激活
            joy_nvec = hi.schemes["tjoy0"].norm_vec  # 调用拇指摇杆的norm_vec属性来获取摇杆的方向向量
            print("摇杆方向向量", joy_nvec)
            # 将拇指摇杆的定点和拇指指尖转化成归一化图片上的坐标后,再绘制
            bs = hi.base_unwrap("hand0")
            dfp = hi.schemes["tjoy0"].fixed_point + bs.norm_pos(0)
            dtp = bs.norm_pos(4)
            cv2.circle(norm_img, bg.calc_norm2img_pos(dfp), 8, (50, 50, 250), 3)
            cv2.line(norm_img, bg.calc_norm2img_pos(dfp), bg.calc_norm2img_pos(dtp), (250, 50, 50), 4)
        cv2.imshow("n", norm_img)
    camera.draw_fps(img)
```

> 在实际应用时，建议为手部检测单独开一个线程（或进程），然后将所需的结果数据传到主线程使用，防止在交互时发生阻塞。

其他的具体API可参考源代码（推荐）或者<a href="./docs/API参考文档.md" title="查看API参考文件">API参考文件</a>

---



## 🔧 自定义开发指南

### 自定义手势交互方案

继承<a href="./src/skhand/HandInputSchemes/HandInputScheme.py" title="查看具体的手势交互方案抽象基类">HandInputScheme</a>抽象类并实现以下两个方法，然后通过定义属性方法返回结果给外部使用。`update`方法会在每次检测后调用（在HandInput的run方法里调用），实时更新该交互方案的状态，如果不需要实时更新的话，可以用pass代替`update`方法的内容；`is_activate`方法是方便使用该交互方案时判断什么时候该交互方案能有正常的输出，另外还需要定义属性方法来让外部调用改交互方案的返回值。

下面以拇指摇杆为例，演示如何自定义手势交互方案。其中，`__init__`的参数除了前面两个`hand_input`手部输入类实例和`hand_name`手部名字不可以改以外，其他都可以自定义，你的交互方案需要哪些外部数据就传入哪些数据就可以了，像这里的拇指摇杆交互方案需要传入一个指尖按钮实例作为参数，是因为拇指摇杆需要利用指尖按钮的长按来作为摇杆的激活标志，注意这里的指尖按钮实例即使是作为参数传入，再创建时也要将其加入`HandInput.schemes`字典里，否则无法自动调用按钮的`update`方法了（如果你想的话，你也可以自行在主循环里面调用）。

```python
from ..HandInput import HandInput
from .HandInputScheme import HandInputScheme
from .FingertipButtonScheme import FingertipButtonScheme

# 创建一个继承于HandInputScheme的类
class ThumbJoystickScheme(HandInputScheme):
    def __init__(self, hand_input: HandInput, hand_name: str, finger_btn: FingertipButtonScheme):
        """拇指摇杆操控方案
        Args:
            hand_input: 手部输入类的实例
            hand_name: 手部名字,指定为哪只手制定手部操控方案
            finger_btn: 一个用于长按设置定点的指尖按钮实例
        """
        hand_input.hands_dict[hand_name]  # 获取该手部,没有该手部名字则报错
        self.hand_input: HandInput = hand_input
        self.hand_name: str = hand_name
        # 需要使用长按来设置定点,所以要传入一个指尖按钮实例
        self.finger_btn: FingertipButtonScheme = finger_btn
        # 创建一个变量来存储定点位置,初始值为None,表示还没有定点
        self._fixed_point: np.ndarray = np.zeros((1, 3))
        # 定义一个变量用于区分,摇杆是否激活,默认为False未激活
        self._activate: bool = False

    """规定需要实现`update`和`is_activate`两个抽象方法"""
    def update(self) -> None:
        """实时更新定点的位置,直到长按才定下来"""
        # 获取并判断是否有检测到手部
        base = self.hand_input.base(self.hand_name)
        if base is None:  # 没有检测到改手部,则不激活摇杆
            self._activate = False
            return
        # 判断是否激活摇杆
        if self.finger_btn.is_long_press:  # 持续长按,则继续激活摇杆
            self._activate = True
        else:  # 长按结束,则关闭摇杆,即不激活
            self._activate = False
        # 摇杆未激活,则定点随拇指指尖移动
        if not self._activate:
            self._fixed_point = base.wrist_npos(4)  # 定点随拇指移动

    @property
    def is_activate(self) -> bool:
        """摇杆是否激活或是否启用,激活为True"""
        return self._activate

    """下面的属性方法都是提供给外部使用的,按需要自行提供,没有具体要求和限制"""
    @property
    def fixed_point(self) -> np.ndarray | None:
        """返回设置的定点位置,没长按时定点位置随拇指移动,若未设置定点则返回None"""
        return self._fixed_point if self._activate else None

    @property
    def end_point(self) -> np.ndarray | None:
        """返回摇杆的终点坐标,是相对于手腕的归一化后坐标值,没有设定点则返回None"""
        base = self.hand_input.base(self.hand_name)
        if base is not None and self._activate:  # 手部存在且摇杆已激活
            return base.wrist_npos(4)  # 终点坐标就是当前手部的拇指坐标

    @property
    def vector(self) -> np.ndarray | None:
        """返回拇指摇杆的向量(未归一化),即当前拇指的位置和定点之差"""
        end_p = self.end_point
        return (end_p - self._fixed_point) if end_p is not None else None

    @property
    def norm_vec(self) -> np.ndarray | None:
        """返回拇指摇杆的方向向量,即归一化的向量"""
        vec = self.vector
        if vec is None:
            return None
        vec_len = np.linalg.norm(vec)
        vec_len = vec_len if vec_len > 0 else 1
        return vec / vec_len
```

### 接入其他手部关键点检测器

继承<a href="./src/skhand/HandDetector/VisualHandDetector.py" title="查看具体的VisualHandDetector代码">VisualHandDetector</a>并实现其抽象方法，`__init__`方法参数必须有`hands_name_ls`所有手部的名字列表和`hands_matcher`手部匹配器实例，其他手部关键点检测器的可选参数可以加在后面，但是一定要加上默认值。

对于`detect`方法是用于接收图片并检测手部关键点数据，然后将上一次检测出来的关键点位置数据对应名字的OneHand中的`last_rpos`变量，把本次检测出来的数据传给`raw_pos`变量，如果改检测器可以区分出左右手的话，可以将左右手结果传给`hand_side`变量，该变量的取值为"left", "right", "Unknown"的其中之一，具体可以查看<a href="./src/skhand/HandDetector/MediaPipeHandDetector.py" title="查看具体的MediaPipeHandDetector代码">MediaPipeHandDetector</a>的实现。

---



## 🤝 关于项目

本项目是我的毕业设计，是个人项目，因时间问题，该项目仍存在很多问题：

- 目前手指并拢状态模型效果并不是很好，请谨慎使用。:(

- 多手匹配器<a href="./src/skhand/HandDetector/HandsMatchers.py" title="具体看HandsMatchers.py">HandsMatchers</a>目前还没有完善好，匹配器整体架构感觉还不够好，是否需要独立出来一个模块，像手部关键点检测器一样。

- 是否需要封装socket用于在不同的编程语言之间传输数据？

- 是否可以开设一个平台给开发者下载其他开发者制作的手部交互方案呢？

---

## 致谢 

在本项目离不开开源社区和前沿技术的贡献。以下是对相关技术开发者和团队的诚挚感谢：  

- **Python** 
  - **官方网站**: [https://www.python.org/](https://www.python.org/) 


- **MediaPipe（手部关键点识别）** 

  - **GitHub**: [https://github.com/google/mediapipe](https://github.com/google/mediapipe) 

  - **论文**: [MediaPipe Hands: On-device Real-time Hand Tracking](https://arxiv.org/abs/2006.10173) 


- **LightGBM（手指状态模型训练）** 

  - **GitHub**: [https://github.com/microsoft/LightGBM](https://github.com/microsoft/LightGBM) 

  - **论文**: [LightGBM: A Highly Efficient Gradient Boosting Decision Tree](https://proceedings.neurips.cc/paper_id/9588.pdf) 


- **ONNX & ONNX Runtime（模型保存与部署）** 

  - **ONNX GitHub**: [https://github.com/onnx/onnx](https://github.com/onnx/onnx)

  - **ONNX Runtime GitHub**: [https://github.com/microsoft/onnxruntime](https://github.com/microsoft/onnxruntime)

- **NumPy & Pandas（数据处理与计算）** 

  - **NumPy GitHub**: [https://github.com/numpy/numpy](https://github.com/numpy/numpy) 

  - **Pandas GitHub**: [https://github.com/pandas-dev/pandas](https://github.com/pandas-dev/pandas) 


- **OpenCV（图像处理与可视化）** 

  - **GitHub**: [https://github.com/opencv/opencv](https://github.com/opencv/opencv) 

  - **论文**: [OpenCV: Open Source Computer Vision Library](https://ieeexplore.ieee.org/document/1067301) 


- **Tkinter（数据收集工具开发）** 
  - **Python官方文档**: [https://docs.python.org/3/library/tkinter.html](https://docs.python.org/3/library/tkinter.html) 