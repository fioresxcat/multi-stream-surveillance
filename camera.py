import numpy as np
import cv2
import os
import time
import pdb
import threading
from line_profiler import profile
from queue import Queue
from collections import deque
from trackers import BYTETracker, BOTSORT
from easydict import EasyDict
from utils.utils import sort_box_by_score, xyxy2xywh, compute_image_blurriness
from methods import ContainerDetector, ContainerInfoDetector, LicensePlateOCR, PPOCRv4Rec
from container_info import ContainerInfo


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



class CameraProcessor:
    def __init__(self, cam_id, fps, frame_size: tuple, 
                 frame_queue: Queue, result_queue: deque, container_detected_event: threading.Event, 
                 config_env: dict, config_model: dict):
        self.cam_id = cam_id
        self.fps = fps
        self.frame_queue = frame_queue
        self.result_queue = result_queue
        self.container_detected_event = container_detected_event
        self.im_w, self.im_h = frame_size
        self.database = {}

        self.tracker = BOTSORT(args=TRACKER_ARGS, frame_rate=fps)
        self.tracker.reset()
        self.container_detector = ContainerDetector.get_instance(config_env, config_model['container_detection'])
        self.container_info_detector = ContainerInfoDetector.get_instance(config_env, config_model['container_info_detection'])
        self.license_plate_ocr = LicensePlateOCR.get_instance(config_env, config_model['license_plate_ocr'])
        self.general_ocr = PPOCRv4Rec.get_instance(config_env, config_model['ppocrv4_rec'])

        self.is_running = False
        self.blur_threshold = 1
        self.frame_cnt = 0
        self.max_time_lost = 2 # seconds


    def is_container_valid(self, bbox, container_info):
        raise NotImplementedError("is_container_valid method should be implemented for each specific camera")
    

    def extract_container_info(self, container_im, extract_labels=[]):
        extracted_info = {lb: (None, 0) for lb in extract_labels}
        if len(extract_labels) == 0:
            return extracted_info

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
        for cl_name in final_boxes.keys():
            if cl_name in ['rear_license_plate', 'front_license_plate'] or cl_name not in extract_labels:
                continue
            box, _ = final_boxes[cl_name]
            field_im = container_im[box[1]:box[3], box[0]:box[2]]
            blur_score = compute_image_blurriness(field_im)
            if blur_score < self.blur_threshold:
                batch_images.append(field_im)
                batch_labels.append(cl_name)
        results = self.general_ocr.predict(batch_images)
        for res, label in zip(results, batch_labels):
            text, prob = res
            extracted_info[label] = (text, prob)
        
        return extracted_info
    

    def print(self, str):
        print(f'{self.cam_id.upper()}: {str}')


    @property
    def current_time(self):
        return self.frame_cnt / self.fps
    

    @profile
    def run(self):
        self.is_running = True
        while self.is_running:
            try:
                frame_info = self.frame_queue.get(block=True, timeout=0.5)
            except Exception as e:
                continue
            self.frame_cnt += 1
            # print(f'------- CAMERA {self.cam_id} - FRAME {self.frame_cnt} - TIME {self.current_time} ---------')
            s = time.perf_counter()
            
            timestamp, frame = frame_info['timestamp'], frame_info['frame']
            boxes, scores, cl_names = self.container_detector.predict([frame])[0]
            tracked_ids = []
            if len(boxes) > 0:
                class_ids = [0] * len(boxes)
                # track
                xywh_boxes = [xyxy2xywh(box) for box in boxes]
                dets = {'conf': np.array(scores), 'xywh': np.array(xywh_boxes), 
                        'cls': np.array(class_ids), 'xyxy': np.array(boxes)}
                self.tracker.update(results=dets, img=frame)

                # extract info for ids
                for track in self.tracker.tracked_stracks:  # for all activated tracks in this frame
                    obj_id = track.track_id
                    tracked_ids.append(obj_id)
                    cls = track.cls
                    # bbox = np.array(track.bb_history[-1]).astype(int)  # cái box này đúng rồi, nhưng ko hiểu sao ocr lại ra sai
                    bbox = np.array(track.xyxy).astype(int)
                    bbox[::2] = np.clip(bbox[::2], 0, self.im_w)
                    bbox[1::2] = np.clip(bbox[1::2], 0, self.im_h)
                    bbox = bbox.tolist()

                    # update container_info in database
                    if obj_id not in self.database:
                        container_info = ContainerInfo(self.cam_id, obj_id, self.fps, (self.im_w, self.im_h))
                        container_info.start_time = timestamp
                        container_info.update_history(timestamp, bbox)
                        self.database[obj_id] = container_info
                    else:
                        container_info: ContainerInfo = self.database[obj_id]
                        container_info.update_history(timestamp, bbox)

                    # if valid, extract info
                    if self.is_container_valid(bbox, container_info) and not container_info.is_full:
                        container_im = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                        incomplete_labels = container_info.get_incomplete_labels()
                        extracted_info = self.extract_container_info(container_im, extract_labels=incomplete_labels)
                    else:
                        extracted_info = {}
                    container_info.update_info(extracted_info)

                    # set event
                    if container_info.direction is not None and not self.container_detected_event[self.cam_id].is_set():
                        # print(f'container detected from {self.cam_id}!')
                        self.container_detected_event[self.cam_id].set()
                    
                    if container_info.is_full and not container_info.pushed_to_queue:
                        # check if the last valid container is pushed or not
                        will_push = True
                        keys = list(self.database.keys())
                        index = keys.index(obj_id)
                        for i in range(index-1, -1, -1):
                            prev_container_info = self.database[keys[i]]
                            if prev_container_info.is_valid_container and not prev_container_info.pushed_to_queue:
                                will_push = False
                                break
                        if will_push:
                            self.result_queue.append({
                                'type': 'rear_info' if container_info.direction == 'out' else 'front_info',
                                'start_time': container_info.start_time,
                                'push_time': self.current_time,
                                'info': container_info.info,
                                'is_done': False
                            })
                            container_info.pushed_to_queue = True
                            with open(f'logs/{self.cam_id}_queue.txt', 'a') as f:
                                f.write(f'time: {time.time()} - {self.result_queue[-1]}\n')
            else:
                # theo dung logic la can phai update tracker de update frame_id chu nhi
                # nhung yolo11 cung ko lam the ma chi track khi detect duoc object
                # tam thoi de the nay vay
                # theo logic trong byte_tracker thi 1 track được coi là removed nếu tracker.frame_id - strack.last_frame_id > max_time_lost
                # do đó frame_id của tracker phải cần được cập nhật nếu ko detect được chứ ??
                # self.tracker.frame_id += 1
                pass

            # remove linhtinh tracks
            for id in list(self.database.keys()):
                if id in tracked_ids:
                    continue
                container_info = self.database[id]
                # update info
                container_info.update_info({})
                # check to remove non tracked containers
                container_info.time_since_update += 1
                if container_info.time_since_update > self.max_time_lost * self.fps:
                    if container_info.is_valid_container and not container_info.pushed_to_queue:
                        result = {
                            'type': 'rear_info' if container_info.direction == 'out' else 'front_info',
                            'start_time': container_info.start_time,
                            'push_time': self.current_time,
                            'info': container_info.info,
                            'is_done': False
                        }
                        self.result_queue.append(result)
                        with open(f'logs/{self.cam_id}_queue.txt', 'a') as f:
                            f.write(f'time: {time.time()} - {self.result_queue[-1]}\n')
                    self.database.pop(id)
            
            # clear event if no container detected
            if len(list(self.database.keys())) == 0 and self.container_detected_event[self.cam_id].is_set():
                print(f'{self.cam_id}: clear container detected event because nothing in database')
                self.container_detected_event[self.cam_id].clear()

            print(f'------- FRAME {self.frame_cnt} - TIME {timestamp} - {self.cam_id.upper()} DATABASE -------')  
            for container_id, container_info in self.database.items():
                print(f'CONTAINER {container_id}: {container_info}')
            print()

            # print(f'{self.cam_id} time elapsed: {time.perf_counter() - s:.2f}s')



class HTTCameraProcessor(CameraProcessor):
    def is_container_valid(self, bbox, container_info):
        direction = container_info.direction
        if direction is None:
            return False
        
        xmin, ymin, xmax, ymax = bbox
        cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
        if direction == 'out':
            return ymax < int(0.95*self.im_h)
        elif direction == 'in':
            return cx < self.im_w // 2
        

class HTSCameraProcessor(CameraProcessor):
    def is_container_valid(self, bbox, container_info):
        direction = container_info.direction
        if direction is None:
            return False
        
        xmin, ymin, xmax, ymax = bbox
        cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
        if direction == 'in':
            return xmin > self.im_w // 2
        elif direction == 'out':
            return ymax < self.im_h - 20