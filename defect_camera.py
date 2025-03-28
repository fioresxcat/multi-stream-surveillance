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
import logging

from utils.utils import sort_box_by_score, xyxy2xywh, compute_image_blurriness, clip_bbox
from base_camera import BaseCameraProcessor
from container_info import ContainerDefectInfo
from methods import ContainerDetector, DefectDetector



class DefectCameraProcessor(BaseCameraProcessor):
    def __init__(self, cam_id, fps, frame_size: tuple, skip_frame: int, 
                 frame_queue: Queue, result_queue: deque, container_detected_event: Dict, 
                 depend_cameras: List[str], config_inference_server: dict, config_model: dict):
        super().__init__(cam_id, fps, frame_size, skip_frame, frame_queue, result_queue, container_detected_event)
        self._setup_logging()

        self.depend_cameras = depend_cameras
        self.max_frame_result = 3

        # setup models
        self.container_detector = ContainerDetector.get_instance(config_inference_server, config_model['container_detection'])
        self.defect_detector = DefectDetector.get_instance(config_inference_server, config_model['container_defect_detection'])


    def _setup_logging(self):
        self.logger = logging.getLogger(f'camera-{self.cam_id}')
        self.logger.info(f"Initializing Defect Camera Processor for camera {self.cam_id}")
        self.log_path = os.path.join(logging.getLogger().log_dir, f'camera-{self.cam_id}.log')
        with open(self.log_path, 'w') as f:
            f.write('')


    def process(self):
        self.is_running = True
        last_frame = None
        last_boxes, last_scores, last_cl_names = [], [], []
        while self.is_running:
            if (not self.is_having_container() or self.frame_queue.empty()) and len(self.database) == 0:
                time.sleep(0.05)
                continue

            frame_info = self._get_next_frame()
            if not frame_info:
                continue

            self.frame_cnt += 1
            timestamp, frame = frame_info['timestamp'], frame_info['frame']

            # check frame diff
            if last_frame is None or self.is_frame_different(frame, last_frame):
                boxes, scores, cl_names = self.container_detector.predict([frame])[0]
                last_frame = frame.copy()
                last_boxes, last_scores, last_cl_names = boxes, scores, cl_names
            else:  # frame is the same and last frame has boxes
                boxes, scores, cl_names = last_boxes, last_scores, last_cl_names

            # process active tracks
            tracked_ids = []
            if len(boxes) > 0:
                tracked_ids = self._process_detections(frame, timestamp, boxes, scores)
            # process inactive tracks
            inactive_ids = [id for id in self.database.keys() if id not in tracked_ids]
            self._process_inactive_tracks(inactive_ids)
            # log database state
            self._log_database_state(timestamp, boxes)
        


    def _is_frame_candidate(self, frame, timestamp, container_bbox, container_info):
        moving_direction = container_info.moving_direction
        if moving_direction is None:
            return False
        
        # check timestamp
        MIN_TIMESTAMP_DIFF = 0.2 # seconds
        last_timestamp, last_image = container_info.images[-1]
        if timestamp - last_timestamp < MIN_TIMESTAMP_DIFF:
            return False
        
        xmin, ymin, xmax, ymax = container_bbox

        # Hong phai sau:
        # Logic: 
        # + the tail of container must be quite away from the edge (with the assumption )
        # + and not too far away from the edge
        if self.cam_id in ['hps']:
            if moving_direction == 'l2r':
                return 0.12 * self.im_w <= xmin <= 0.5 * self.im_w
            elif moving_direction == 'r2l':
                return 0.5 * self.im_w <= xmax <= 0.88 * self.im_w
                # return xmin < 0.1 * self.im_w and 0.5 * self.im_w <= xmax <= 1. * self.im_w  # debug
            
        
        # Hong trai truoc
        elif self.cam_id in ['htt']:
            if moving_direction == 'l2r':  # in
                return 0.12 * self.im_w <= xmin <= 0.4 * self.im_w
            elif moving_direction == 'r2l':  # out
                return xmin <= 0.12 * self.im_w and xmax >= 0.5 * self.im_w
        
        # Hongtraisau
        # 
        elif self.cam_id in ['hts']:
            if moving_direction == 'l2r': # out
                # we leave xmin to be far from edge to detect defect in rear side
                return xmin <= 0.5 * self.im_w and xmax >= 0.88 * self.im_w
            elif moving_direction == 'r2l': # in
                # we set xmax to be really close to edge to detect defect in front side
                return xmax <= 0.95 * self.im_w  and xmax >= 0.65 * self.im_w

        # Noccongtruoc
        elif self.cam_id in ['nct']:
            if moving_direction == 't2b':
                return ymin >= 0.1 * self.im_h and ymax >= 0.96 * self.im_h
            elif moving_direction == 'b2t':
                return ymin <= 0.15 * self.im_h and ymax >= 0.96 * self.im_h
        
        
        else:
            raise NotImplementedError(f"Camera ID {self.cam_id} not supported")
        

    def is_having_container(self):
        """
        Checks if all dependent cameras have detected a container.
        """
        return all(self.container_detected_event[cam_id].is_set() for cam_id in self.depend_cameras)


    def _extract_defect_info(self, images):
        """
        Extracts defect information from a list of images.
        """
        return self.defect_detector.predict(images)


    def _process_detections(self, frame, timestamp, boxes, scores):
        """
        Tracks objects, updates the database, and processes defect information.
        """
        tracked_ids = []
        class_ids = [0] * len(boxes)
        xywh_boxes = [xyxy2xywh(box) for box in boxes]
        dets = {'conf': np.array(scores), 'xywh': np.array(xywh_boxes), 'cls': np.array(class_ids), 'xyxy': np.array(boxes)}
        self.tracker.update(results=dets, img=frame)

        for track in self.tracker.tracked_stracks:
            obj_id = track.track_id
            tracked_ids.append(obj_id)
            bbox = clip_bbox(np.array(track.xyxy).astype(int), self.im_w, self.im_h).tolist()
            container_info = self._update_container_history(obj_id, timestamp, bbox)

            if not container_info.is_done:
                if not container_info.is_full_candidate and self._is_frame_candidate(frame, timestamp, bbox, container_info):
                    container_im = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                    container_info.add_candidate_images(timestamp, container_im)

                if container_info.is_full_candidate:
                    self._process_cand_images(container_info)
                    self._finish_container_info_if_full(container_info)
            else:
                if (not container_info.is_pushed) and container_info.is_valid_container:
                    if self.is_last_valid_container_pushed(container_info):
                        self._push_info(container_info)

        return tracked_ids


    def _update_container_history(self, obj_id, timestamp, bbox):
        if obj_id not in self.database:
            container_info = ContainerDefectInfo(self.cam_id, obj_id, self.fps, (self.im_w, self.im_h), self.max_frame_result)
            container_info.start_time = timestamp
            self.database[obj_id] = container_info
        else:
            container_info: ContainerDefectInfo = self.database[obj_id]
        container_info.update_history(timestamp, bbox)
        return container_info


    def _push_info(self, container_info):
        result = {
            'type': f'defect_info',
            'camera_id': self.cam_id,
            'start_time': container_info.start_time,
            'push_time': container_info.history[-1][0],
            'info': container_info.info,
            'is_matched': False
        }
        self.result_queue.append(result)

        print_info = [el['names'] for el in container_info.info]
        with open(self.log_path, 'a') as f:
            f.write(f'--------------- Container {container_info.id} ---------------\n')
            f.write(f'OCR Info: {print_info}\n\n')
        container_info.is_pushed = True
    

    def _process_inactive_tracks(self, inactive_ids):
        """
        Removes inactive tracks from the tracker and the database.
        """
        for id in inactive_ids:
            container_info: ContainerDefectInfo = self.database[id]
            container_info.time_since_update += 1  # history update
            if container_info.time_since_update <= self.max_frame_lost:
                continue
            # process for container that does not appear for a long time
            if container_info.is_valid_container and not container_info.is_pushed:
                self._push_info(container_info)
            self.database.pop(id)
            for track in self.tracker.lost_stracks:
                if track.track_id == id:
                    track.mark_removed()
                    self.tracker.lost_stracks.remove(track)
                    self.tracker.removed_stracks.append(track)
                    break


    def _log_database_state(self, timestamp, boxes):
        """
        Logs the current state of the database for debugging purposes.
        """
        self.logger.debug(f'------- FRAME {self.frame_cnt} - TIME {timestamp} - {self.cam_id.upper()} DATABASE -------')
        self.logger.debug(f'Find {len(boxes)} boxes in this frame')
        for container_id, container_info in self.database.items():
            print_info = [el['names'] for el in container_info.info]
            self.logger.debug(
                f'CONTAINER {container_id}: appear: {container_info.num_appear}, '
                f'moving_direction: {container_info.moving_direction}, '
                f'camera_direction: {container_info.camera_direction}, '
                f'num images: {len(container_info.images)}, '
                f'is_full: {container_info.is_full}, is_done: {container_info.is_done}, '
                f'info: {print_info}'
            )


    def run(self):
        self.process()