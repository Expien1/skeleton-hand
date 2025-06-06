# 核心功能类
from .HandInput import HandInput

# 子包核心类(常用)
from .HandUtils.Camera import Camera
from .HandInputSchemes.FingertipButtonScheme import FingertipButtonScheme
from .HandInputSchemes.ThumbJoystickScheme import ThumbJoystickScheme
from .HandInputSchemes.FingerSwipeScheme import FingerSwipeScheme

# 子包入口(进阶,保留子包结构)
from . import HandData, HandDetector, HandInputSchemes, HandUtils

# 定义版本
__version__ = "0.1.0"
# 定义公共接口
__all__ = [
    # 常用类
    "HandInput",
    "Camera",
    "FingertipButtonScheme",
    "ThumbJoystickScheme",
    "FingerSwipeScheme",
    # 子包入口
    "HandData",
    "HandDetector",
    "HandInputSchemes",
    "HandUtils",
]
