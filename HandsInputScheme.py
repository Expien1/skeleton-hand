from abc import ABC, abstractmethod
import multiprocessing

from .Camera import Camera
from .HandsInput import HandsInput
from .HandsDetector.MediaPipeHandsDetector import MediaPipeHandsDetector


class HandsInputScheme(ABC, multiprocessing.Process):
    def __init__(self, camera_idx=0):
        self.camera = Camera(camera_idx)
        self.hands_input = HandsInput(["h0", "h1"], MediaPipeHandsDetector)
        self.socket = None

    @abstractmethod
    def mainloop(self):
        for img in self.camera.read():
            if self.hands_input.run(img):
                ...

    def unix_domain_socket(self):
        pass

    @abstractmethod
    def run_socket(self):
        pass

    @abstractmethod
    def run(self):
        self.mainloop()
        self.run_socket()
