import pdb
import cv2
import numpy as np

from utils.utils import total_time, sort_bbs
from modules.object_detection.yolov8.yolo_detector import YOLODetector


class ContainerDetector(YOLODetector):
    instance = None
    
    def __init__(self, common_config, model_config):
        super(ContainerDetector, self).__init__(common_config, model_config)
        self.labels = ['container']

    
    @staticmethod
    def get_instance(common_config, model_config):
        if ContainerDetector.instance is None:
            ContainerDetector.instance = ContainerDetector(common_config, model_config)
        return ContainerDetector.instance