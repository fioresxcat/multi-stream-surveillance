import pdb
import cv2
import numpy as np

from utils.utils import total_time, sort_bbs
from modules.object_detection.yolov8.yolo_detector import YOLODetector


class ContainerInfoDetector(YOLODetector):
    instance = None
    
    def __init__(self, common_config, model_config):
        super(ContainerInfoDetector, self).__init__(common_config, model_config)
        self.labels = ['owner_code', 'container_number', 'check_digit', 'container_type', 'rear_license_plate', 'rear_license_plate_2', 'front_license_plate']

    
    @staticmethod
    def get_instance(common_config, model_config):
        if ContainerInfoDetector.instance is None:
            ContainerInfoDetector.instance = ContainerInfoDetector(common_config, model_config)
        return ContainerInfoDetector.instance