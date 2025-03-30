import numpy as np
import cv2
import os
import time
import pdb
import threading
from line_profiler import profile
from queue import Queue
from typing_extensions import List, Dict, Tuple, Union, Any, Literal
from collections import deque
from easydict import EasyDict
import logging

from modules.trackers import BYTETracker, BOTSORT
from utils.utils import *
from methods import ContainerDetector, ContainerInfoDetector, LicensePlateOCR, PPOCRv4Rec, ParseqLicensePlateOCR, ParseqGeneralOCR
from container_info import ContainerOCRInfo
from base_camera import BaseCameraProcessor



class OCRCameraProcessor(BaseCameraProcessor):
    def __init__(self, cam_id, fps, frame_size: tuple, skip_frame: int,
                 frame_queue: Queue, result_queue: deque, container_detected_event: Dict, 
                 config_inference_server: dict, config_model: dict):
        super().__init__(cam_id, fps, frame_size, skip_frame, frame_queue, result_queue, container_detected_event)
        self._setup_logging()

        # setup models
        self.container_detector = ContainerDetector.get_instance(config_inference_server, config_model['container_detection'])
        self.container_info_detector = ContainerInfoDetector.get_instance(config_inference_server, config_model['container_info_detection'])
        self.license_plate_ocr = ParseqLicensePlateOCR.get_instance(config_inference_server, config_model['vietnamese_lp_ocr'])
        self.general_ocr = PPOCRv4Rec.get_instance(config_inference_server, config_model['ppocrv4_rec'])
        # self.general_ocr = ParseqGeneralOCR.get_instance(config_inference_server, config_model['parseq_tiny_general_ocr'])

        self.blur_threshold = 1  # threshold to check for blurriness when extract info
        self.min_appearance_to_count_as_detected = 0.5  # seconds


    def _setup_logging(self):
        self.logger = logging.getLogger(f'camera-{self.cam_id}')
        self.logger.info(f"Initializing OCR Camera Processor for camera {self.cam_id}")
        self.log_dir = os.path.join(logging.getLogger().log_dir, f'camera-{self.cam_id}')
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_path = os.path.join(self.log_dir, 'log.log')
        clear_file(self.log_path)


    def process(self):
        self.is_running = True
        last_frame = None
        last_boxes, last_scores, last_cl_names = [], [], []
        while self.is_running:
            frame_info = self._get_next_frame()
            if not frame_info:
                continue

            self.frame_cnt += 1
            timestamp, frame = frame_info['timestamp'], frame_info['frame']

            # check frame diff
            # risk: if the first frame of the streak, bboxes is not detected or detected wrong -> all subsequent duplicate frame will inherit the wrong boxes
            if last_frame is None or is_frame_different(frame, last_frame):
                boxes, scores, cl_names = self.container_detector.predict([frame])[0]
                last_frame = frame.copy()
                last_boxes, last_scores, last_cl_names = boxes, scores, cl_names
            else:  # frame is the same and last frame has boxes
                boxes, scores, cl_names = last_boxes, last_scores, last_cl_names

            # boxes, scores, cl_names = self.container_detector.predict([frame])[0]
            # is_frame_different = True

            tracked_ids = []
            if len(boxes) > 0:
                tracked_ids = self._process_detections(frame, timestamp, boxes, scores)
            # remove inactive tracks
            inactive_ids = [id for id in self.database.keys() if id not in tracked_ids]
            self._process_inactive_tracks(inactive_ids)
            # clear event
            if len(self.database) == 0 and self.container_detected_event[self.cam_id].is_set():
                self.container_detected_event[self.cam_id].clear()
            # log database state
            self._log_database_state(timestamp, boxes)

            
    def _extract_container_info(self, container_im, extract_labels=[]):
        extracted_info = {lb: (None, 0) for lb in extract_labels}
        if len(extract_labels) == 0:
            return extracted_info

        # detect info
        boxes, scores, classes = self.container_info_detector.predict([container_im])[0]
        final_boxes = {}
        for box, score, cls in zip(boxes, scores, classes):
            if cls not in final_boxes or score > final_boxes[cls][1]:
                final_boxes[cls] = (box, score)

        # ocr lp
        for label in ['rear_license_plate', 'front_license_plate']:
            if label in final_boxes and label in extract_labels:
                box, box_score = final_boxes[label]
                lp_im = container_im[box[1]:box[3], box[0]:box[2]]
                blur_score = compute_image_blurriness(lp_im)
                if blur_score < self.blur_threshold:
                    lp_text, prob = self.license_plate_ocr.predict([lp_im])[0]
                    extracted_info[label] = (lp_text, prob)
        
        # ocr field
        batch_images, batch_labels = [], []
        for label in final_boxes.keys():
            if label in ['rear_license_plate', 'front_license_plate'] or label not in extract_labels:
                continue
            box, _ = final_boxes[label]
            field_im = container_im[box[1]:box[3], box[0]:box[2]]
            blur_score = compute_image_blurriness(field_im)
            if blur_score < self.blur_threshold:
                batch_images.append(field_im)
                batch_labels.append(label)
        results = self.general_ocr.predict(batch_images)
        for res, label in zip(results, batch_labels):
            text, prob = res
            extracted_info[label] = (text, prob)
        
        extracted_info = self._postprocess_info(extracted_info)
        return extracted_info
    

    def _postprocess_info(self, extracted_info):
        for label, (value, prob) in extracted_info.items():
            if value is None:
                continue

            if label in ['container_type']:
                if len(value) == 4 and value[2] == '6':
                    value = value[:2] + 'G' + value[3:]

            extracted_info[label] = (value, prob)
        return extracted_info        


    def _process_detections(self, frame, timestamp, boxes, scores):
        """
            track -> update database -> extract info -> set container detected event -> push to queue
        """
        # update tracker
        tracked_ids = []
        class_ids = [0] * len(boxes)
        xywh_boxes = [xyxy2xywh(box) for box in boxes]
        dets = {'conf': np.array(scores), 'xywh': np.array(xywh_boxes), 'cls': np.array(class_ids), 'xyxy': np.array(boxes)}
        self.tracker.update(results=dets, img=frame)

        # update each id
        for track in self.tracker.tracked_stracks:
            obj_id = track.track_id
            tracked_ids.append(obj_id)
            bbox = clip_bbox(np.array(track.xyxy).astype(int), self.im_w, self.im_h).tolist()
            container_info = self._update_or_create_container(obj_id, timestamp, bbox)
            
            if not container_info.is_done:
                if self._is_frame_candidate(bbox, container_info):
                    container_im = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                    extracted_info = self._extract_container_info(container_im, extract_labels=container_info.get_incomplete_labels())
                    container_info.update_info(extracted_info)
                else:
                    container_info.update_info({}) # if container detected but cannot extract info, still increase info_time_since_update
            # set container detected event
            if container_info.num_appear >= self.min_appearance_to_count_as_detected and not self.container_detected_event[self.cam_id].is_set():
                self.container_detected_event[self.cam_id].set()
            # push result to queue if needed
            if container_info.is_done and (not container_info.is_pushed) and container_info.is_valid_container:
                if self.is_last_valid_container_pushed(container_info):
                    self._push_info(container_info)
                    # do not remove id here, otherwise the tracker will create new id for the container we've just removed
        
        return tracked_ids


    def _update_or_create_container(self, obj_id, timestamp, bbox):
        if obj_id not in self.database:
            container_info = ContainerOCRInfo(self.cam_id, obj_id, self.fps, (self.im_w, self.im_h), self.skip_frame)
            container_info.start_time = timestamp
            self.database[obj_id] = container_info
        else:
            container_info = self.database[obj_id]
        container_info.update_history(timestamp, bbox)
        return container_info
    

    def _push_info(self, container_info: ContainerOCRInfo):
        result = {
            'type': 'rear_info' if container_info.camera_direction == 'out' else 'front_info',
            'camera_id': self.cam_id,
            'start_time': container_info.start_time,
            'push_time': container_info.history[-1][0],  # current timestamp of this frame
            'info': container_info.info,
            'is_matched': False
        }
        self.result_queue.append(result)
        with open(self.log_path, 'a') as f:
            f.write(f'--------------- Container {container_info.id} ---------------\n')
            f.write(f'OCR Info: {container_info.info}\n\n')
        container_info.is_pushed = True


    def _process_inactive_tracks(self, inactive_ids):
        """
        Removes inactive tracks from the tracker and the database.
        """
        for id in inactive_ids:
            container_info: ContainerOCRInfo = self.database[id]
            container_info.update_info({})  # info update
            container_info.time_since_update += 1  # history update
            if container_info.time_since_update <= self.max_frame_lost:
                continue
            # process for container that does not appear for a long time
            if container_info.is_valid_container and not container_info.is_pushed:
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
            self.logger.debug(
                f'CONTAINER {container_id}: appear: {container_info.num_appear}, '
                f'moving_direction: {container_info.moving_direction}, '
                f'camera_direction: {container_info.camera_direction}, '
                f'is_valid: {container_info.is_valid_container}, '
                f'is_done: {container_info.is_done}, '
                f'info: {container_info.info}'
            )
        self.logger.debug('\n\n')
    
    
    def _is_frame_candidate(self, bbox, container_info: ContainerOCRInfo):
        """
            heuristic to check if the current frame is a candidate for extracting info
        """
        camera_direction = container_info.camera_direction
        if camera_direction is None:
            return False
        xmin, ymin, xmax, ymax = bbox
        cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2

        if self.cam_id in ['htt-ocr']:            
            if camera_direction == 'out':
                return ymax < 0.95*self.im_h or xmax < 0.6 * self.im_w  # container is moving out from right to left
            elif camera_direction == 'in':
                return xmax < self.im_w // 2 and ymax < 0.95 * self.im_h  # container is moving in from left to right

        elif self.cam_id in ['hts-ocr']:
            if camera_direction == 'in':
                return 0.8 * self.im_w > xmin > self.im_w // 2
            elif camera_direction == 'out':
                return ymax < 0.95 * self.im_h or xmin > 0.5
        
        elif self.cam_id in ['bst-ocr']:
            if camera_direction == 'in':
                return xmax > 0.5 * self.im_w
            elif camera_direction == 'out':
                return xmax < 0.7 * self.im_w and ymax < 0.8 * self.im_h
        
        elif self.cam_id in ['bss-ocr']:
            if camera_direction == 'in':
                return xmin < 0.6 * self.im_w and xmin > 0.2 * self.im_w
            elif camera_direction == 'out':
                return xmin > 0.2 * self.im_w

        else:
            raise NotImplementedError(f"Camera ID {self.cam_id} not supported")
        

    def run(self):
        self.process()