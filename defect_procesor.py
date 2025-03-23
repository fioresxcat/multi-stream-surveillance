import pdb
import numpy as np
import cv2
from typing_extensions import List, Dict, Tuple, Union, Any, Literal
import threading
from collections import deque
from queue import Queue

from utils.utils import sort_box_by_score
from methods.defect_detector import DefectDetector
from methods.container_detector import ContainerDetector


class DefectProcessor:
    def __init__(self, frame_queues: dict, frame_sizes: dict, result_queue: deque, fps: int, 
                 container_detected_event: threading.Event,
                 config_env: dict, config_model: dict):
        self.frame_queues = frame_queues
        self.camera_names = list(self.frame_queues.keys())
        self.frame_sizes = frame_sizes
        self.result_queue = result_queue
        self.fps = fps
        self.container_detected_event = container_detected_event

        self.defect_detector = DefectDetector.get_instance(config_env, config_model['container_defect_detection'])
        self.container_detector = ContainerDetector.get_instance(config_env, config_model['container_detection'])
        
        self.is_running = False



    def is_candidate(self, cam_id, container_bb):
        frame_w, frame_h = self.frame_sizes[cam_id]
        xmin, ymin, xmax, ymax = container_bb
        if cam_id == 'nct':  # noccongtruoc
            return ymin >= 1/3 * frame_h
        else:
            raise NotImplementedError(f'Camera {cam_id} is not supported yet')
        


    def find_frame_candidates(self, cam_id: str, frame_queue: Queue, result_queue: Queue):
        """
            read frames from queue and process them, find the potential frames to do defect detection, then push them to result queue
        """
        if frame_queue.empty():
            return
        
        cands = []
        while self.container_detected_event.is_set() and len(cands) < 3:  # max 3 cands
            if frame_queue.empty():
                break
            frame_info = frame_queue.get()
            time_stamp, frame = frame_info['time_stamp'], frame_info['frame']
            if frame is None:
                break
            boxes, scores, cl_names = self.container_detector.detect([frame])[0]
            if len(boxes) == 0:
                continue
            boxes, scores, cl_names = sort_box_by_score(boxes, scores, cl_names)
            container_bb = boxes[0]
            if self.is_candidate(cam_id, container_bb):
                cands.append((cam_id, frame, container_bb))
            
            # release the frame
            frame_queue.task_done()
        
        # choose best cand
        if len(cands) > 0:
            cand = cands[0]
            result_queue.append(cand)



    def run(self):
        self.is_running = True
        while self.is_running:
            self.container_detected_event.wait()
            
            print(f'container detected, finding frames ...')
            # find candidate frames
            all_cands = Queue()
            processing_threads = []
            for cam_id in self.camera_names:
                frame_queue = self.frame_queues[cam_id]
                if frame_queue.empty():
                    continue
                thread = threading.Thread(target=self.find_frame_candidates, args=(cam_id, frame_queue, all_cands))
                processing_threads.append(thread)
                thread.start()
            for thread in processing_threads:
                thread.join()

            # detect defects
            frame_infos = list(all_cands.queue)
            if len(frame_infos) > 0:
                images, cam_ids = [], []
                for cam_id, frame, container_bb in frame_infos:
                    crop = frame[container_bb[1]:container_bb[3], container_bb[0]:container_bb[2]]
                    images.append(crop)
                    cam_ids.append(cam_id)
                list_boxes, list_scores, list_cl_names = self.defect_detector.predict(frame_infos)
                defect_infos = {}
                for boxes, scores, cl_names, cam_id in zip(list_boxes, list_scores, list_cl_names, cam_ids):
                    defect_infos[cam_id] = cl_names
                self.result_queue.append({
                    'type': 'defect_info',
                    'info': defect_infos,
                })