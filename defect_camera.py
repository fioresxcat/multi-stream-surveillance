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

from utils.utils import *
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

        # setup models
        self.container_detector = ContainerDetector.get_instance(config_inference_server, config_model['container_detection'])
        self.defect_detector = DefectDetector.get_instance(config_inference_server, config_model['container_defect_detection'])


    def _setup_logging(self):
        self.logger = logging.getLogger(f'camera-{self.cam_id}')
        self.logger.info(f"Initializing Defect Camera Processor for camera {self.cam_id}")
        self.log_dir = os.path.join(logging.getLogger().log_dir, f'camera-{self.cam_id}')
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_path = os.path.join(self.log_dir, 'log.log')
        clear_file(self.log_path)


    def process(self):
        self.is_running = True
        last_frame = None
        last_boxes, last_scores, last_cl_names = [], [], []
        is_different_from_last_frame = True
        while self.is_running:
            if (not self.is_having_container() or self.frame_queue.empty()) and len(self.database) == 0:
                time.sleep(0.05)
                self.logger.debug('skipping 1')
                continue

            frame_info = self._get_next_frame()
            if not frame_info:
                self.logger.debug('skipping 2')
                continue

            self.frame_cnt += 1
            timestamp, frame = frame_info['timestamp'], frame_info['frame']

            # check frame diff
            # risk: if the first frame of the streak, bboxes is not detected or detected wrong -> all subsequent duplicate frame will inherit the wrong boxes
            if last_frame is None or is_frame_different(frame, last_frame):
                boxes, scores, cl_names = self.container_detector.predict([frame])[0]
                last_frame = frame.copy()
                last_boxes, last_scores, last_cl_names = boxes, scores, cl_names
                is_different_from_last_frame = True
            else:  # frame is the same
                boxes, scores, cl_names = last_boxes, last_scores, last_cl_names
                is_different_from_last_frame = False

            # process active tracks
            tracked_ids = []
            if len(boxes) > 0:
                tracked_ids = self._process_detections(frame, is_different_from_last_frame, timestamp, boxes, scores)
            # process inactive tracks
            inactive_ids = [id for id in self.database.keys() if id not in tracked_ids]
            self._process_inactive_tracks(inactive_ids)
            # log database state
            self._log_database_state(timestamp, boxes)
        


    def _process_detections(self, frame, is_different_from_last_frame, timestamp, boxes, scores):
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
            container_info = self._update_or_create_container(obj_id, timestamp, bbox)

            if not container_info.is_done and is_different_from_last_frame:
                if not container_info.is_full_buffer and self._is_frame_candidate(frame, timestamp, bbox, container_info):
                    container_im = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                    container_info.add_image_to_buffer(timestamp, container_im)

                if container_info.is_full_buffer:
                    self._process_image_buffer(container_info)
                
            if container_info.is_full_result and not container_info.is_done and container_info.is_valid_container:
                container_info.gather_final_results()
                container_info.is_done = True
            
            if container_info.is_done and not container_info.is_pushed:
                if self.is_last_valid_container_pushed(current_container=container_info):
                    self._push_info(container_info)
                    # do not remove id here, otherwise the tracker will create new id for the container we've just removed
        return tracked_ids


    def _process_image_buffer(self, container_info: ContainerDefectInfo):
        images = [im for timestamp, im in container_info.image_buffer]
        results = self.defect_detector.predict(images)
        log_dir = os.path.join(self.log_dir, 'image_buffer')
        os.makedirs(log_dir, exist_ok=True)
        for i, (boxes, scores, names) in enumerate(results):
            timestamp, im = container_info.image_buffer[i]
            # logging
            draw_im = im.copy()
            for box, score, name in zip(boxes, scores, names):
                cv2.rectangle(draw_im, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)
                cv2.putText(draw_im, f'{name}-{score:.2f}', (int(box[0]), int(box[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.imwrite(os.path.join(log_dir, f'container_{container_info.id}-{timestamp}.jpg'), draw_im)

            if 'trailer' not in names or set(names) == {'trailer'}:
                continue
            boxes, scores, names = sort_box_by_score(boxes, scores, names)
            trailer_bbox = boxes[names.index('trailer')]
            _, _, iou = iou_bbox(trailer_bbox, (0, 0, im.shape[1], im.shape[0]))
            if iou > 0.7:  # trailer bbox overlaps big with container bbox
                resized_im = resize_image_to_height(im, min(480, im.shape[0]))
                norm_boxes = [normalize_bbox(box, im.shape[1], im.shape[0]) for box in boxes]
                # remove all "trailer" boxes
                new_boxes, new_scores, new_names = [], [], []
                for box, score, name in zip(norm_boxes, scores, names):
                    if name != 'trailer':
                        new_boxes.append(box)
                        new_scores.append(score)
                        new_names.append(name)
                container_info.add_cand_result(resized_im, new_boxes, new_scores, new_names, timestamp)
            
        container_info.image_buffer = []
        

    def is_having_container(self):
        """
        Checks if all dependent cameras have detected a container.
        """
        return all(self.container_detected_event[cam_id].is_set() for cam_id in self.depend_cameras)


    def _update_or_create_container(self, obj_id, timestamp, bbox):
        if obj_id not in self.database:
            container_info = ContainerDefectInfo(self.cam_id, obj_id, self.fps, (self.im_w, self.im_h))
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
            f.write(f'Defect Info: {print_info}\n\n')
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
            if container_info.is_valid_container and not container_info.is_done:
                if len(container_info.image_buffer) > 0:
                    self._process_image_buffer(container_info)
                container_info.gather_final_results()
                container_info.is_done = True
            
            if container_info.is_done and not container_info.is_pushed:
                self._push_info(container_info)
                
            self.database.pop(id)
            self.tracker.remove_lost_track(id)


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
                f'num images in buffer: {len(container_info.image_buffer)}, '
                f'num cand results: {len(container_info.cand_results)}, '
                f'is_full: {container_info.is_full_result}, is_done: {container_info.is_done}, '
                f'info: {print_info}'
                f'\n'
            )


    def _is_frame_candidate(self, frame, timestamp, container_bbox, container_info: ContainerDefectInfo):
        """
            heuristic check if the frame is a candidate for defect detection
            final check (include trailer bbox and container bbox iou will happen concurrently with defect detection)
            doing in this way to avoid having to call to defect detection model for every frame to check (do not utilize batch size)
        """
        moving_direction = container_info.moving_direction
        if moving_direction is None:
            return False
        camera_direction = container_info.camera_direction
        assert camera_direction in ['in', 'out']

        # if timestamps are too close, skip the frame
        MIN_TIMESTAMP_DIFF = 0.2 # seconds
        if container_info.last_timestamp is not None and timestamp - container_info.last_timestamp < MIN_TIMESTAMP_DIFF:
            return False
        
        xmin, ymin, xmax, ymax = container_bbox
        
        # Hong phai sau:
        if self.cam_id in ['hps-defect']:
            MIN_APPEAR_TIME_FROM_START = 4 # seconds. Containers often take time for the trailer to be clearly visible in the camera
            if timestamp - container_info.start_time  < MIN_APPEAR_TIME_FROM_START:
                return False
            if moving_direction == 'r2l' and xmax < 0.25 * self.im_w:  # we leave the container really close to the edge to capture defect in rear side
                return False
            elif moving_direction == 'l2r' and xmin > 0.75 * self.im_w:
                return False
        
        # Hong trai truoc
        elif self.cam_id in ['htt-defect']:
            if camera_direction == 'in':  # in
                if not (0.12 * self.im_w <= xmin <= 0.4 * self.im_w): # bbox condition
                    return False
                MIN_APPEAR_TIME_FROM_START = 4.5
            elif moving_direction == 'r2l':  # out
                if not (xmin <= 0.12 * self.im_w and xmax >= 0.5 * self.im_w): # bbox condition
                    return False
                MIN_APPEAR_TIME_FROM_START = 1.5
            if timestamp - container_info.start_time < MIN_APPEAR_TIME_FROM_START:  # time condition
                return False
        
        # Hongtraisau
        elif self.cam_id in ['hts-defect']:
            if camera_direction == 'out': # out
                MIN_APPEAR_TIME_FROM_START = 3
                # we leave xmin to be far from edge to detect defect in rear side
                if not (xmin <= 0.5 * self.im_w and xmax >= 0.88 * self.im_w):
                    return False
            elif camera_direction == 'in': # in
                MIN_APPEAR_TIME_FROM_START = 4.5
                # we set xmax to be really close to edge to detect defect in front side
                if not (xmax <= 0.95 * self.im_w  and xmax >= 0.65 * self.im_w):
                    return False
            if timestamp - container_info.start_time < MIN_APPEAR_TIME_FROM_START:  # time condition
                return False
        
        # Noccongtruoc
        elif self.cam_id in ['nct-defect']:
            if camera_direction == 'in':  # moving t2b
                MIN_APPEAR_TIME_FROM_START = 5
                if not (ymin >= 0.1 * self.im_h and ymax >= 0.96 * self.im_h):
                    return False
            elif camera_direction == 'out':
                MIN_APPEAR_TIME_FROM_START = 2
                if not (ymin <= 0.15 * self.im_h and ymax >= 0.96 * self.im_h):
                    return False
            if timestamp - container_info.start_time < MIN_APPEAR_TIME_FROM_START:
                return False
        
        else:
            raise NotImplementedError(f"Camera ID {self.cam_id} not supported")
        
        return True


    def run(self):
        self.process()