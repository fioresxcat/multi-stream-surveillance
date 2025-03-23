import pdb
import cv2
import numpy as np

from utils.utils import total_time, sort_bbs
from modules.object_detection.yolov8.yolo_detector import YOLODetector


class DefectDetector(YOLODetector):
    instance = None
    
    def __init__(self, common_config, model_config):
        super(DefectDetector, self).__init__(common_config, model_config)
        self.labels = ['container', 'scratch', 'rust', 'dent']

    
    @staticmethod
    def get_instance(common_config, model_config):
        if DefectDetector.instance is None:
            DefectDetector.instance = DefectDetector(common_config, model_config)
        return DefectDetector.instance