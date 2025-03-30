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
from easydict import EasyDict
import queue
import logging

from modules.trackers import BYTETracker, BOTSORT
from utils.utils import sort_box_by_score, xyxy2xywh, compute_image_blurriness, clip_bbox, clear_file
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

CAMERA_MODE = 'video'


class BaseCameraProcessor:
    def __init__(self, cam_id, cam_src: str, skip_time: float,
                 result_queue: deque, container_detected_event: Dict):
        self.cam_id = cam_id
        self.cam_src = cam_src
        self.skip_time = skip_time
        self.result_queue = result_queue
        self.container_detected_event = container_detected_event
        self.database = OrderedDict()
        self.frame_cnt = 0
        self.is_running = False

        self._setup_logging()
        # setup tracker
        self.fps = 25
        self.skip_frame = 4
        self.max_time_lost = 1.5 # seconds
        self.max_frame_lost = int(self.max_time_lost * self.fps) / self.skip_frame
        self.tracker = BOTSORT(args=TRACKER_ARGS, max_frame_lost=self.max_frame_lost)
        self.tracker.reset()

        # frame queue
        self.frame_queue = queue.Queue(maxsize=10)


    def _setup_logging(self):
        self.logger = logging.getLogger(f'camera-{self.cam_id}')
        self.logger.info(f"Initializing Camera Processor for camera {self.cam_id}")
        self.log_dir = os.path.join(logging.getLogger().log_dir, f'camera-{self.cam_id}')
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_path = os.path.join(self.log_dir, 'log.log')
        clear_file(self.log_path)
        if 'defect' in self.cam_id:
            image_log_dir = os.path.join(self.log_dir, 'image_buffer')
            for fn in os.listdir(image_log_dir):
                os.remove(os.path.join(image_log_dir, fn))


    def _get_next_frame(self):
        try:
            return self.frame_queue.get(block=True, timeout=0.1)
        except Exception:
            return None
        

    def get_frames(self):
        self.cap = cv2.VideoCapture(self.cam_src)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.im_w, self.im_h = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.skip_frame = int(self.skip_time * self.fps)

        is_stopped = False
        frame_index = 0

        # # Set the initial position of the video capture
        # for cam_id, cap in self.caps.items():
        #     cap.set(cv2.CAP_PROP_POS_MSEC, 15)

        while self.is_running:
            # self.logger.debug('reading frames ...')
            if CAMERA_MODE == 'video' and self.frame_queue.full():  # if mode is video, process all frames
                time.sleep(0.05)
                # self.logger.debug('frame queue is full, continue')
                continue
            if is_stopped:
                frame = np.full((self.im_h, self.im_w, 3), 255, dtype=np.uint8)
            else:
                # self.logger.debug('about to read frame ...')
                # pdb.set_trace()
                ret, frame = self.cap.read()
                if not ret:
                    self.logger.info(f"Failed to read frame from {self.cam_id}")
                    is_stopped = True
                    continue
            frame_index += 1
            if frame_index % self.skip_frame != 0:
                continue
            if not is_stopped:
                timestamp = round(self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0, 2)  # Convert to seconds
            else:
                timestamp = frame_index / self.fps
            # self.logger.debug('prepare to push ...')
            self.frame_queue.put({'timestamp': timestamp, 'frame': frame, 'frame_index': frame_index})  # notice this

    
    def is_last_valid_container_pushed(self, current_container: BaseContainerInfo):
        if len(self.database) <= 1:
            return True
        keys = list(self.database.keys())
        index = keys.index(current_container.id)
        last_container_info = self.database[keys[index-1]]
        if last_container_info.is_valid_container and not last_container_info.is_pushed:
            return False
        return True


