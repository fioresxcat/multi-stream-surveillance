import numpy as np
import cv2
import os
import time
import pdb
import threading
from line_profiler import profile
from queue import Queue
from typing_extensions import List, Dict, Tuple, Union, Any, Literal
from collections import deque, OrderedDict
from modules.trackers import BYTETracker, BOTSORT
from easydict import EasyDict
from utils.utils import sort_box_by_score, xyxy2xywh, compute_image_blurriness, clip_bbox
from container_info import BaseContainerInfo, ContainerOCRInfo, ContainerDefectInfo


TRACKER_ARGS = EasyDict({
    "track_high_thresh": 0.75,  # threshold for the first association. the scores from detection model is really good, so set this to high
    
    "track_low_thresh": 0.5,  # threshold for the second association
    
    "new_track_thresh": 0.75,  # threshold for init new track if the detection does not match any tracks
    
    # iou (combined with detection scores) threshold for matching tracks. a detection is associcated with a track if iou > match_thresh
    # tested with frame_skip = 1 (hps and nct), iou always higher than 0.90 when moving. only false detection make iou < 0.90
    # with frame_skip = 4 (hps, nct, htt), iou usually > 0.7
    # "match_thresh": 0.75, # if frame_skip = 1
    "match_thresh": 0.7, # if frame_skip = 4

    # BoT-SORT settings
    "gmc_method": None,  # method of global motion compensation
    "proximity_thresh": 0.5, # only used when with_reid = True
    "appearance_thresh": 0.25, # only used when with_reid = True
    "with_reid": False,
    "fps": "${camera.fps}"
})


class BaseCameraProcessor:
    def __init__(self, cam_id, fps, frame_size: tuple, skip_frame: int,
                 frame_queue: Queue, result_queue: deque, container_detected_event: Dict):
        self.cam_id = cam_id
        self.fps = fps
        self.im_w, self.im_h = frame_size
        self.skip_frame = skip_frame
        self.frame_queue = frame_queue
        self.result_queue = result_queue
        self.container_detected_event = container_detected_event
        self.database = OrderedDict()
        self.frame_cnt = 0
        self.is_running = False

        # setup tracker
        self.max_time_lost = 1.5 # seconds
        self.max_frame_lost = int(self.max_time_lost * fps) / self.skip_frame
        self.tracker = BOTSORT(args=TRACKER_ARGS, max_frame_lost=self.max_frame_lost)
        self.tracker.reset()

    
    def _get_next_frame(self):
        try:
            return self.frame_queue.get(block=True, timeout=0.1)
        except Exception:
            return None

    
    def is_last_valid_container_pushed(self, current_container: BaseContainerInfo):
        if len(self.database) <= 1:
            return True
        keys = list(self.database.keys())
        index = keys.index(current_container.id)
        last_container_info = self.database[keys[index-1]]
        if last_container_info.is_valid_container and not last_container_info.is_pushed:
            return False
        return True


