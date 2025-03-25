import os
import pdb
import cv2
import numpy as np
import torch
from pathlib import Path
from easydict import EasyDict
import time
import yaml
from queue import Queue
from collections import deque
from typing_extensions import List, Dict, Tuple, Optional, Literal, Any
from utils.utils import load_yaml
from info_camera import *
from defect_camera import *


config_env = load_yaml('configs/config_env.yaml')['inference_server']
config_model = load_yaml('configs/config_models.yaml')


class ContainerProcessor:
    def __init__(self, video_sources: dict, fps: int):
        self.cam1, self.cam2 = 'htt', 'hts'
        self.ocr_cams = [self.cam1, self.cam2]
        # self.defect_cams = ['hps']
        self.defect_cams = []
        self.container_detected_event = {cam_id: threading.Event() for cam_id in video_sources.keys()}

        # setup capture
        self.caps = {cam_id: cv2.VideoCapture(cam_source) for cam_id, cam_source in video_sources.items()}
        self.frame_sizes = {cam_id: (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))) for cam_id, cap in self.caps.items()}
        # setup frames queue
        self.frame_queues = {cam_id: Queue(maxsize=10) for cam_id in video_sources.keys()}
        # setup results queue
        self.ocr_results_queues = {cam_id: deque() for cam_id in [self.cam1, self.cam2]}
        self.defect_results_queue = {cam_id: deque() for cam_id in self.defect_cams}
        # setup info processor
        self.ocr_camera_processors = {}
        for cam_id, cam_source in video_sources.items():
            if cam_id not in [self.cam1, self.cam2]:
                continue
            if cam_id == 'htt':
                camera_class = HTTCameraProcessor
            elif cam_id == 'hts':
                camera_class = HTSCameraProcessor
            else:
                raise NotImplementedError(f'Camera {cam_id} is not supported yet')
            self.ocr_camera_processors[cam_id] = camera_class(cam_id, fps, self.frame_sizes[cam_id], self.frame_queues[cam_id], 
                                                              self.ocr_results_queues[cam_id], self.container_detected_event, 
                                                              config_env, config_model)
        # setup defect processor
        self.defect_camera_processors = {}
        for cam_id, cam_source in video_sources.items():
            if cam_id not in self.defect_cams:
                continue
            if cam_id == 'hps':
                camera_class = HPSCameraProcessor
            else:
                raise NotImplementedError(f'Camera {cam_id} is not supported yet')
            self.defect_camera_processors[cam_id] = camera_class(cam_id, fps, self.frame_sizes[cam_id], self.frame_queues[cam_id], 
                                                                 self.defect_results_queue[cam_id], self.container_detected_event, self.ocr_cams, 
                                                                 config_env, config_model)

        self.is_running = False
        self.fps = fps
        self.final_results = []
        self.max_time_diff = 1e9
        self.output_path = 'logs/results.log'
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        with open(self.output_path, 'w') as f:
            f.write('CONTAINER DETECTION RESULTS\n')


    def get_frames(self):
        is_stopped = {cam_id: False for cam_id in self.caps.keys()}

        # fourcc = cv2.VideoWriter_fourcc(*'XVID')
        # out = cv2.VideoWriter('output.avi', fourcc, self.fps, (640, 480))

        while self.is_running:
            for cam_id, cap in self.caps.items():
                cam_queue: Queue = self.frame_queues[cam_id]
                if is_stopped[cam_id]:
                    continue
                if cam_queue.full():  # comment this to always get most recent frames
                    continue
                ret, frame = cap.read()
                if not ret:
                    print(f"Failed to read frame from {cam_id}")
                    is_stopped[cam_id] = True
                    continue
                frame_index = cap.get(cv2.CAP_PROP_POS_FRAMES)
                timestamp = round(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0, 2)  # Convert to seconds
                if timestamp < 5:  # start from second n
                    continue
                cam_queue.put({'timestamp': timestamp, 'frame': frame, 'frame_index': frame_index})  # notice this

            if all(is_stopped.values()):
                print("All cameras stopped. Exiting...")
                break



    def match_results(self):
        def combine_info(ocr_infos: List, defect_infos: List):
            final_ocr_info = {}
            for ocr_info in ocr_infos:
                for label, (value, score) in ocr_info.items():
                    if label not in final_ocr_info:
                        final_ocr_info[label] = (value, score)
                    else:
                        if score > final_ocr_info[label][1]:
                            final_ocr_info[label] = (value, score)
            
            final_defect_info = []
            for defect_info in defect_infos:
                for im_index, im_result in enumerate(defect_info):
                    final_defect_info.append(im_result)

            info = {'ocr_info': final_ocr_info, 'defect_info': final_defect_info}
            return info
        

        container_cnt = 0
        while self.is_running:
            time.sleep(0.1)
            
            # print('------------ RESULTS QUEUE ------------')
            # print(f'{self.cam1.upper()}: {list(res_queue_1)}')
            # print(f'{self.cam2.upper()}: {list(res_queue_2)}')
            # print()

            if any(len(queue) == 0 for queue in self.ocr_results_queues.values()) or any(len(queue) == 0 for queue in self.defect_results_queue.values()):
                continue
            
            ocr_infos = []
            for cam_id, ocr_queue in self.ocr_results_queues.items():
                ocr_infos.append(ocr_queue.popleft())
            defect_infos = []
            for cam_id, defect_queue in self.defect_results_queue.items():
                defect_infos.append(defect_queue.popleft())

            info = combine_info(ocr_infos, defect_infos)

            # write result
            start_time = min(el['start_time'] for el in ocr_infos)
            container_cnt += 1
            with open(self.output_path, 'a') as f:
                f.write(f'time: {time.time()} Container {container_cnt}: start_time: {start_time}, info: {info}\n')


    def stop(self):
        self.is_running = False
        for cam_id, cap in self.caps.items():
            cap.release()


    def run(self):
        self.is_running = True
        get_frame_thread = threading.Thread(target=self.get_frames, daemon=True)
        processing_threads = []
        for cam_id, processor in self.ocr_camera_processors.items():
            processing_threads.append(threading.Thread(target=processor.run, daemon=True))
        # for cam_id, processor in self.defect_camera_processors.items():
        #     processing_threads.append(threading.Thread(target=processor.run, daemon=True))
        matching_thread = threading.Thread(target=self.match_results, daemon=True)
        
        # Start threads
        get_frame_thread.start()
        time.sleep(0.01)
        for thread in processing_threads:
            thread.start()
        matching_thread.start()
        
        # Wait for threads to finish
        get_frame_thread.join()
        for thread in processing_threads:
            thread.join()
        matching_thread.join()

        self.stop()


def main():
    fps = 25
    video_sources = {
        'htt': 'test_files/hongtraitruoc-cut6.mp4',
        'hts': 'test_files/hongtraisau-cut6.mp4',
        'hps': 'test_files/hongphaisau-cut6.mp4'
        # 'bst': 'test_files/biensotruoc-part2.mkv',
        # 'bss': 'test_files/biensosau-part2.mkv',
        # 'htt': 'test_files/hongtraitruoc-part2.mp4',
        # 'hts': 'test_files/hongtraisau-part2.mp4',
    }
    processor = ContainerProcessor(video_sources, fps)
    processor.run()
    processor.stop()



if __name__ == "__main__":
    main()