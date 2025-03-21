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
from camera import *


config_env = load_yaml('configs/config_env.yaml')['inference_server']
config_model = load_yaml('configs/config_models.yaml')

container_detected_event = threading.Event()


class ContainerProcessor:
    def __init__(self, video_sources: dict, fps: int):
        # setup capture
        self.caps = {cam_id: cv2.VideoCapture(cam_source) for cam_id, cam_source in video_sources.items()}
        self.frame_sizes = {cam_id: (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))) for cam_id, cap in self.caps.items()}
        # setup frames queue
        self.frame_queues = {cam_id: Queue(maxsize=10) for cam_id in video_sources.keys()}
        # setup results queue
        self.results_queues = {cam_id: deque() for cam_id in video_sources.keys()}
        # setup processor
        self.camera_processors = {}
        for cam_id, cam_source in video_sources.items():
            if cam_id == 'htt':
                camera_class = HTTCameraProcessor
            elif cam_id == 'hts':
                camera_class = HTSCameraProcessor
            else:
                raise NotImplementedError(f'Camera {cam_id} is not supported yet')
            
            self.camera_processors[cam_id] = camera_class(cam_id, fps, self.frame_sizes[cam_id], self.frame_queues[cam_id], self.results_queues[cam_id], container_detected_event, config_env, config_model)

        self.is_running = False
        self.fps = fps
        self.final_results = []
        self.cam1, self.cam2 = 'htt', 'hts'
        # self.max_time_diff = int(3 * self.fps)  # 3 seconds
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
                if is_stopped[cam_id]:
                    continue
                ret, frame = cap.read()
                if not ret:
                    print(f"Failed to read frame from {cam_id}")
                    is_stopped[cam_id] = True
                    continue
                frame_index = cap.get(cv2.CAP_PROP_POS_FRAMES)
                timestamp = round(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0, 2)  # Convert to seconds
                self.frame_queues[cam_id].put({'timestamp': timestamp, 'frame': frame, 'frame_index': frame_index})



    def match_results(self):
        res_queue_1 = self.results_queues[self.cam1]
        res_queue_2 = self.results_queues[self.cam2]
        container_cnt = 0
        while self.is_running:
            time.sleep(0.1)
            
            # print('------------ RESULTS QUEUE ------------')
            # print(f'{self.cam1.upper()}: {list(res_queue_1)}')
            # print(f'{self.cam2.upper()}: {list(res_queue_2)}')
            # print()

            if len(res_queue_1) == 0 or len(res_queue_2) == 0:
                continue
            
            res1 = res_queue_1.popleft()
            res2 = res_queue_2.popleft()

            if abs(res1['start_time'] - res2['start_time']) < self.max_time_diff: # Match found
                info = {}
                for label, (value, score) in res1['info'].items():
                    if label not in info:
                        info[label] = (value, score)
                    else:
                        if score > info[label][1]:
                            info[label] = (value, score)
                for label, (value, score) in res2['info'].items():
                    if label not in info:
                        info[label] = (value, score)
                    else:
                        if score > info[label][1]:
                            info[label] = (value, score)
                self.final_results.append(info)
                print(f'Find a container: {info}')

                # write result
                start_time = res1['start_time']
                container_cnt += 1
                with open(self.output_path, 'a') as f:
                    f.write(f'Container {container_cnt}: start_time: {start_time}, info: {info}\n')


    def stop(self):
        self.is_running = False
        for cam_id, cap in self.caps.items():
            cap.release()


    def process(self):
        self.is_running = True
        get_frame_thread = threading.Thread(target=self.get_frames, daemon=True)
        processing_threads = []
        for cam_id, processor in self.camera_processors.items():
            processing_threads.append(threading.Thread(target=processor.run, daemon=True))
        matching_thread = threading.Thread(target=self.match_results, daemon=True)
        
        # Start threads
        get_frame_thread.start()
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
        'htt': 'test_files/hongtraitruoc-part2_cut1-duplicate.mp4',
        'hts': 'test_files/hongtraisau-part2_cut1-duplicate.mp4',
        # 'bst': 'test_files/biensotruoc-part2.mkv',
        # 'bss': 'test_files/biensosau-part2.mkv',
        # 'htt': 'test_files/hongtraitruoc-part2.mp4',
        # 'hts': 'test_files/hongtraisau-part2.mp4',
    }
    processor = ContainerProcessor(video_sources, fps)
    processor.process()
    processor.stop()



if __name__ == "__main__":
    main()