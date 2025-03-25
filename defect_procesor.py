import pdb
import numpy as np
import cv2
import time
from typing_extensions import List, Dict, Tuple, Union, Any, Literal
import threading
from collections import deque
from queue import Queue

from utils.utils import sort_box_by_score
from methods.defect_detector import DefectDetector
from methods.container_detector import ContainerDetector


class DefectProcessor:
    def __init__(self, frame_queues: dict, frame_sizes: dict, result_queue: deque, fps: int, 
                 container_detected_event: dict, required_cameras: List[str],
                 config_env: dict, config_model: dict):
        self.frame_queues = frame_queues
        self.camera_names = list(self.frame_queues.keys())
        self.frame_sizes = frame_sizes
        self.result_queue = result_queue
        self.fps = fps
        self.container_detected_event = container_detected_event
        self.required_cameras = required_cameras

        self.defect_detector = DefectDetector.get_instance(config_env, config_model['container_defect_detection'])
        self.container_detector = ContainerDetector.get_instance(config_env, config_model['container_detection'])
        
        self.is_running = False


    def wait(self):
        """
            wait until all cameras have container detected
        """
        for cam_id in self.required_cameras:
            self.container_detected_event[cam_id].wait()


    def is_having_container(self):
        """
            all cameras must have container detected
        """
        for cam_id in self.required_cameras:
            if not self.container_detected_event[cam_id].is_set():
                return False
        return True
    

    def is_candidate(self, cam_id, container_bb):
        frame_w, frame_h = self.frame_sizes[cam_id]
        xmin, ymin, xmax, ymax = container_bb
        if cam_id == 'hps':  # noccongtruoc
            return 0.6 * frame_w < xmax < 0.9 * frame_w
        else:
            raise NotImplementedError(f'Camera {cam_id} is not supported yet')
        


    def find_frame_candidates(self, cam_id: str, frame_queue: Queue, result_queue: Queue):
        """
            read frames from queue and process them, find the potential frames to do defect detection, then push them to result queue
        """
        print(f'find_frame_candidates: cam_id: {cam_id}')
        # pdb.set_trace()
        cands = []
        while True:  # max 3 cands
            if not self.is_having_container():
                print(f'break because container_detected_event is not set')
        
            if frame_queue.empty():
                print(f'break because frame_queue is empty')
                break

            frame_info = frame_queue.get()
            time_stamp, frame = frame_info['timestamp'], frame_info['frame']
            if frame is None:
                print(f'break because frame is None')
                break
            boxes, scores, cl_names = self.container_detector.predict([frame])[0]
            print(f'Found {len(boxes)} container at cam_id: {cam_id}, time_stamp: {time_stamp}')
            if len(boxes) == 0:
                continue
            boxes, scores, cl_names = sort_box_by_score(boxes, scores, cl_names)
            container_bb = boxes[0]
            # if self.is_candidate(cam_id, container_bb):
            if time_stamp >= 18.5:
                cands.append((cam_id, frame, container_bb))
                print(f'CONTAINER DETECTED: cam_id: {cam_id}, time_stamp: {time_stamp}, is_valid: TRUE')
            else:
                print(f'CONTAINER DETECTED: cam_id: {cam_id}, time_stamp: {time_stamp}, is_valid: FALSE')
            frame_queue.task_done()
            if len(cands) == 3:
                print(f'found 3 candidates, stop searching')
                # pdb.set_trace()
                break
            
        # choose best cand
        if len(cands) > 0:
            cand = cands[-1]
            result_queue.put(cand)


    def test_sleep(self):
        time.sleep(2)


    def run(self):
        self.is_running = True
        while self.is_running:
            self.wait()
            
            print(f'container detected, finding frames ...')
            
            # find candidate frames
            all_cands = Queue()
            processing_threads = []
            for cam_id in self.camera_names:
                frame_queue = self.frame_queues[cam_id]
                if frame_queue.empty():
                    continue
                
                thread = threading.Thread(target=self.find_frame_candidates, args=(cam_id, frame_queue, all_cands))
                # thread = threading.Thread(target=self.test_sleep)

                processing_threads.append(thread)
                thread.start()
            for thread in processing_threads:
                thread.join()

            # # find candidate frames
            # all_cands = Queue()
            # for cam_id in self.camera_names:
            #     frame_queue = self.frame_queues[cam_id]
            #     if frame_queue.empty():
            #         continue
            #     self.find_frame_candidates(cam_id, frame_queue, all_cands)

            # detect defects
            frame_infos = list(all_cands.queue)
            print(f'num cands: {len(frame_infos)}')
            # pdb.set_trace()
            if len(frame_infos) > 0:
                images, cam_ids = [], []
                for cam_id, frame, container_bb in frame_infos:
                    crop = frame[container_bb[1]:container_bb[3], container_bb[0]:container_bb[2]]
                    images.append(crop)
                    cam_ids.append(cam_id)
                results = self.defect_detector.predict(images)
                defect_infos = {}
                for (boxes, scores, cl_names), cam_id in zip(results, cam_ids):
                    defect_infos[cam_id] = cl_names
                self.result_queue.append({
                    'type': 'defect_info',
                    'info': defect_infos,
                })
                pdb.set_trace()