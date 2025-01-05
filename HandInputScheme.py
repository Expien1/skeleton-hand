from abc import ABC, abstractmethod
import multiprocessing

from .HandUtils.Camera import Camera
from .HandInput import HandInput
from .HandDetector.MediaPipeHandDetector import MediaPipeHandDetector


class HandInputScheme(ABC, multiprocessing.Process):
    def __init__(self, camera_idx=0):
        self.camera = Camera(camera_idx)
        self.hands_input = HandInput(["h0", "h1"], MediaPipeHandDetector)
        self.socket = None

    @abstractmethod
    def mainloop(self):
        """手部检测主循环"""
        for img in self.camera.read():
            if self.hands_input.detect(img):
                ...

    def unix_domain_socket(self):
        """进程间传输数据"""
        pass

    @abstractmethod
    def run_socket(self):
        """启动socket传输"""
        pass

    @abstractmethod
    def run(self):
        """运行该手部方案"""
        self.mainloop()
        self.run_socket()
