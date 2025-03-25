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
from trackers import BYTETracker, BOTSORT
from easydict import EasyDict
from utils.utils import sort_box_by_score, xyxy2xywh, compute_image_blurriness

from container_info import ContainerDefectInfo
from methods import ContainerDetector, DefectDetector


TRACKER_ARGS = EasyDict({
    "track_high_thresh": 0.3,  # threshold for the first association
    "track_low_thresh": 0.1,  # threshold for the second association
    "new_track_thresh": 0.6,  # threshold for init new track if the detection does not match any tracks
    "track_buffer": 45,  # buffer to calculate the time when to remove tracks
    "match_thresh": 0.75,  # threshold for matching tracks
    # "min_box_area": 10,  # threshold for min box areas(for tracker evaluation, not used for now)
    # "mot20": False,  # for tracker evaluation(not used for now)

    # BoT-SORT settings
    "gmc_method": None,  # method of global motion compensation
    # ReID model related thresh (not supported yet)
    "proximity_thresh": 0.5,
    "appearance_thresh": 0.25,
    "with_reid": False,
    "fps": "${camera.fps}"
})


class DefectCameraProcessor:
    def __init__(self, cam_id, fps, frame_size: tuple, 
                 frame_queue: Queue, result_queue: deque, container_detected_event: Dict, 
                 required_cameras: List[str], config_env: dict, config_model: dict):
        self.cam_id = cam_id
        self.fps = fps
        self.frame_queue = frame_queue
        self.result_queue = result_queue
        self.container_detected_event = container_detected_event
        self.required_cameras = required_cameras
        self.im_w, self.im_h = frame_size
        self.database = OrderedDict()

        self.container_detector = ContainerDetector.get_instance(config_env, config_model['container_detection'])
        self.defect_detector = DefectDetector.get_instance(config_env, config_model['container_defect_detection'])
        self.tracker = BOTSORT(args=TRACKER_ARGS, frame_rate=fps)
        self.tracker.reset()

        self.is_running = False
        self.frame_cnt = 0
        self.max_time_lost = 1.2 # seconds
        self.max_frame_result = 3


    def is_having_container(self):
        """
            all cameras must have container detected
        """
        for cam_id in self.required_cameras:
            if not self.container_detected_event[cam_id].is_set():
                return False
        return True
    

    def is_bbox_valid(self, bbox, container_info):
        raise NotImplementedError()
    

    def extract_defect_info(self, images):
        results = self.defect_detector.predict(images)
        return results
    

    def process(self):
        self.is_running = True
        while self.is_running:
            if not self.is_having_container() or self.frame_queue.empty():
                time.sleep(0.1)
                print(f'skip because having no container')
                continue
            frame_info = self.frame_queue.get()
            timestamp, frame = frame_info['timestamp'], frame_info['frame']
            
            boxes, scores, cl_names = self.container_detector.predict([frame])[0]
            tracked_ids = []
            if len(boxes) > 0:
                class_ids = [0] * len(boxes)
                xywh_boxes = [xyxy2xywh(box) for box in boxes]
                dets = {'conf': np.array(scores), 'xywh': np.array(xywh_boxes), 
                        'cls': np.array(class_ids), 'xyxy': np.array(boxes)}
                self.tracker.update(results=dets, img=frame)

                # extract info for ids
                for track in self.tracker.tracked_stracks:  # for all activated tracks in this frame
                    obj_id = track.track_id
                    tracked_ids.append(obj_id)
                    # bbox = np.array(track.bb_history[-1]).astype(int)  # cái box này đúng rồi, nhưng ko hiểu sao ocr lại ra sai
                    bbox = np.array(track.xyxy).astype(int)  # cái này là bb xấp xỉ predict bằng kalman filter
                    bbox[::2] = np.clip(bbox[::2], 0, self.im_w)
                    bbox[1::2] = np.clip(bbox[1::2], 0, self.im_h)
                    bbox = bbox.tolist()

                    # update container_info in database
                    if obj_id not in self.database:
                        container_info = ContainerDefectInfo(self.cam_id, obj_id, self.fps, (self.im_w, self.im_h), self.max_frame_result)
                        container_info.start_time = timestamp
                        container_info.update_history(timestamp, bbox)
                        self.database[obj_id] = container_info
                    else:
                        container_info: ContainerDefectInfo = self.database[obj_id]
                        container_info.update_history(timestamp, bbox)

                    # check if this image is valid
                    if not container_info.is_full and self.is_bbox_valid(bbox, container_info):
                        container_im = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                        container_info.update_image(timestamp, container_im)
                    
                    # if full, extract defect info
                    if not container_info.is_done and container_info.is_full:
                        defect_info = self.extract_defect_info(container_info.images)
                        container_info.update_info(defect_info)

            else:
                # theo dung logic la can phai update tracker de update frame_id chu nhi
                # nhung yolo11 cung ko lam the ma chi track khi detect duoc object
                # tam thoi de the nay vay
                # theo logic trong byte_tracker thi 1 track được coi là removed nếu tracker.frame_id - strack.last_frame_id > max_time_lost
                # do đó frame_id của tracker phải cần được cập nhật nếu ko detect được chứ ??
                # self.tracker.frame_id += 1
                pass

            # remove linhtinh tracks
            obj_ids = list(self.database.keys())
            for obj_index, obj_id in enumerate(obj_ids):
                container_info = self.database[obj_id]
                if obj_id in tracked_ids and container_info.is_done and not container_info.pushed_to_queue:
                    # check if the last valid container is pushed or not
                    will_push = True
                    for i in range(obj_index-1, -1, -1):
                        prev_container_info = self.database[obj_ids[i]]
                        if prev_container_info.is_valid_container and not prev_container_info.pushed_to_queue:
                            will_push = False
                            break
                    if will_push:
                        self.result_queue.append({
                            'type': f'{self.cam_id}_defect_info',
                            'start_time': container_info.start_time,
                            'push_time': timestamp,
                            'info': container_info.info,
                            'is_matched': False
                        })
                        container_info.pushed_to_queue = True
                        with open(f'logs/{self.cam_id}_queue.txt', 'a') as f:
                            f.write(f'time: {time.time()} - {self.result_queue[-1]}\n')

                else:  # non tracked ids
                    # check to remove non tracked containers
                    container_info.time_since_update += 1
                    if container_info.time_since_update > self.max_time_lost * self.fps:
                        if container_info.is_valid_container and not container_info.pushed_to_queue:
                            result = {
                                'type': f'{self.cam_id}_defect_info',
                                'start_time': container_info.start_time,
                                'push_time': timestamp,
                                'info': container_info.info,
                                'is_matched': False
                            }
                            self.result_queue.append(result)
                            with open(f'logs/{self.cam_id}_queue.txt', 'a') as f:
                                f.write(f'time: {time.time()} - {self.result_queue[-1]}\n')
                        self.database.pop(obj_id)
            
            print(f'------- FRAME {self.frame_cnt} - TIME {timestamp} - {self.cam_id.upper()} DATABASE -------')
            print(f'Find {len(boxes)} boxes in this frame')
            for container_id, container_info in self.database.items():
                print_info = [el['cl_names'] for el in container_info.info]
                is_full = 'FULL' if container_info.is_full else 'NOT FULL'
                is_done = 'DONE' if container_info.is_done else 'NOT DONE'
                print(f'CONTAINER {container_id}: direction: {container_info.moving_direction} {is_full} {is_done} {print_info}')
                if container_info.is_full:
                    pdb.set_trace()
            print()


    def run(self):
        self.process()


class HPSCameraProcessor(DefectCameraProcessor):
    def is_bbox_valid(self, bbox, container_info: ContainerDefectInfo):
        xmin, ymin, xmax, ymax = bbox
        if container_info.moving_direction == 'l2r':
            return xmin >= 1/4 * self.im_w and xmax > 0.85 * self.im_w
        elif container_info.moving_direction == 'r2l':
            return xmax <= 3/4 * self.im_w and xmin <= 0.15 * self.im_w
        else:
            return False