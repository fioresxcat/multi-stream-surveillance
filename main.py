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
from defect_procesor import DefectProcessor


config_env = load_yaml('configs/config_env.yaml')['inference_server']
config_model = load_yaml('configs/config_models.yaml')


class ContainerProcessor:
    def __init__(self, video_sources: dict, fps: int):
        self.cam1, self.cam2 = 'htt', 'hts'
        self.defect_cams = ['hps']
        self.container_detected_event = {cam_id: threading.Event() for cam_id in video_sources.keys()}

        # setup capture
        self.caps = {cam_id: cv2.VideoCapture(cam_source) for cam_id, cam_source in video_sources.items()}
        self.frame_sizes = {cam_id: (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))) for cam_id, cap in self.caps.items()}
        # setup frames queue
        self.frame_queues = {cam_id: Queue(maxsize=10) for cam_id in video_sources.keys()}
        # setup results queue
        self.info_results_queues = {cam_id: deque() for cam_id in [self.cam1, self.cam2]}
        self.defect_results_queue = deque()
        # setup info processor
        self.camera_processors = {}
        for cam_id, cam_source in video_sources.items():
            if cam_id not in [self.cam1, self.cam2]:
                continue
            if cam_id == 'htt':
                camera_class = HTTCameraProcessor
            elif cam_id == 'hts':
                camera_class = HTSCameraProcessor
            else:
                raise NotImplementedError(f'Camera {cam_id} is not supported yet')
            self.camera_processors[cam_id] = camera_class(cam_id, fps, self.frame_sizes[cam_id], self.frame_queues[cam_id], self.info_results_queues[cam_id], self.container_detected_event, config_env, config_model)
        # setup defect processor
        self.defect_processor = DefectProcessor(
            frame_queues={cam_id: self.frame_queues[cam_id] for cam_id in self.defect_cams},
            frame_sizes={cam_id: self.frame_sizes[cam_id] for cam_id in self.defect_cams},
            result_queue=self.defect_results_queue,
            fps=fps,
            container_detected_event=self.container_detected_event,
            required_cameras=[self.cam1, self.cam2],
            config_env=config_env,
            config_model=config_model,
        )

        self.is_running = False
        self.fps = fps
        self.final_results = []
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
                cam_queue: Queue = self.frame_queues[cam_id]
                if is_stopped[cam_id] or cam_queue.full():
                    continue
                ret, frame = cap.read()
                if not ret:
                    print(f"Failed to read frame from {cam_id}")
                    is_stopped[cam_id] = True
                    continue
                frame_index = cap.get(cv2.CAP_PROP_POS_FRAMES)
                timestamp = round(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0, 2)  # Convert to seconds
                if timestamp < 16:  # start from second 15
                    continue
                cam_queue.put({'timestamp': timestamp, 'frame': frame, 'frame_index': frame_index}, block=False)  # notice this

            if all(is_stopped.values()):
                print("All cameras stopped. Exiting...")
                break


    def match_results(self):
        def combine_info(info1, info2):
            info = {}
            for label, (value1, score1) in info1.items():
                if label not in info:
                    info[label] = (value1, score1)
                else:
                    if score1 > info[label][1]:
                        info[label] = (value1, score1)
            for label, (value2, score2) in info2.items():
                if label not in info:
                    info[label] = (value2, score2)
                else:
                    if score2 > info[label][1]:
                        info[label] = (value2, score2)
            return info
        

        res_queue_1 = self.info_results_queues[self.cam1]
        res_queue_2 = self.info_results_queues[self.cam2]
        container_cnt = 0
        while self.is_running:
            time.sleep(0.1)
            
            # print('------------ RESULTS QUEUE ------------')
            # print(f'{self.cam1.upper()}: {list(res_queue_1)}')
            # print(f'{self.cam2.upper()}: {list(res_queue_2)}')
            # print()

            if len(res_queue_1) == 0 or len(res_queue_2) == 0:
                continue
            # pdb.set_trace()
            
            res1 = res_queue_1.popleft()
            res2 = res_queue_2.popleft()
            time1, type1, info1 = res1['start_time'], res1['type'], res1['info']
            time2, type2, info2 = res2['start_time'], res2['type'], res2['info']
            info = None

            # if type1 == 'front_info' and type2 == 'rear_info':
            #     if time1 < time2:  # front_info arrives before rear_info, valid combination
            #         info = combine_info(info1, info2)
            #     else:  # front_info arrives behind rear_info
            #         res_queue_1.appendleft(res1)  # reappend front_info and only pop out rear_info for old container
            #         info = info2
            # elif type1 == 'rear_info' and type2 == 'front_info':
            #     if time2 < time1: # front_info arrives before rear_info, valid combination
            #         info = combine_info(info1, info2)
            #     else:  # front_info arrives behind rear_info
            #         res_queue_2.appendleft(res2)  # reappend front_info and only pop out rear_info for old container
            #         info = info1

            if type1 != type2:
                info = combine_info(info1, info2)
            elif type1 == type2:
                # pop cái đến trước và giữ lại cái đến sau
                if time1 < time2:
                    res_queue_2.appendleft(res2)
                    info = info1
                else:
                    res_queue_1.appendleft(res1)
                    info = info2
                    

            if info is not None:
                # write result
                start_time = res1['start_time']
                container_cnt += 1
                with open(self.output_path, 'a') as f:
                    f.write(f'time: {time.time()} Container {container_cnt}: start_time: {start_time}, info: {info}\n')
                # pdb.set_trace()


    
    def match_results(self):
        def combine_info(info1, info2, info3):
            info = {}
            for label, (value1, score1) in info1.items():
                if label not in info:
                    info[label] = (value1, score1)
                else:
                    if score1 > info[label][1]:
                        info[label] = (value1, score1)
            for label, (value2, score2) in info2.items():
                if label not in info:
                    info[label] = (value2, score2)
                else:
                    if score2 > info[label][1]:
                        info[label] = (value2, score2)
            for label, (value3, score3) in info3.items():
                if label not in info:
                    info[label] = (value3, score3)
                else:
                    if score3 > info[label][1]:
                        info[label] = (value3, score3)
            return info
        

        res_queue_1 = self.info_results_queues[self.cam1]
        res_queue_2 = self.info_results_queues[self.cam2]
        res_queue_3 = self.defect_results_queue
        container_cnt = 0
        while self.is_running:
            time.sleep(0.1)
            
            # print('------------ RESULTS QUEUE ------------')
            # print(f'{self.cam1.upper()}: {list(res_queue_1)}')
            # print(f'{self.cam2.upper()}: {list(res_queue_2)}')
            # print()

            if len(res_queue_1) == 0 or len(res_queue_2) == 0 or len(res_queue_3) == 0:
                continue
            # pdb.set_trace()
            
            res1 = res_queue_1.popleft()
            res2 = res_queue_2.popleft()
            res3 = res_queue_3.popleft()
            time1, type1, info1 = res1['start_time'], res1['type'], res1['info']
            time2, type2, info2 = res2['start_time'], res2['type'], res2['info']
            type3, info3 = res3['type'], res3['info']
            info = None

            # if type1 == 'front_info' and type2 == 'rear_info':
            #     if time1 < time2:  # front_info arrives before rear_info, valid combination
            #         info = combine_info(info1, info2)
            #     else:  # front_info arrives behind rear_info
            #         res_queue_1.appendleft(res1)  # reappend front_info and only pop out rear_info for old container
            #         info = info2
            # elif type1 == 'rear_info' and type2 == 'front_info':
            #     if time2 < time1: # front_info arrives before rear_info, valid combination
            #         info = combine_info(info1, info2)
            #     else:  # front_info arrives behind rear_info
            #         res_queue_2.appendleft(res2)  # reappend front_info and only pop out rear_info for old container
            #         info = info1

            if type1 != type2:
                info = combine_info(info1, info2, info3)
            elif type1 == type2:
                # pop cái đến trước và giữ lại cái đến sau
                if time1 < time2:
                    res_queue_2.appendleft(res2)
                    info = combine_info(info1, info3)
                else:
                    res_queue_1.appendleft(res1)
                    info = combine_info(info2, info3)
                    
            if info is not None:
                # write result
                start_time = res1['start_time']
                container_cnt += 1
                with open(self.output_path, 'a') as f:
                    f.write(f'time: {time.time()} Container {container_cnt}: start_time: {start_time}, info: {info}\n')
                # pdb.set_trace()


    def stop(self):
        self.is_running = False
        for cam_id, cap in self.caps.items():
            cap.release()


    def run(self):
        self.is_running = True
        get_frame_thread = threading.Thread(target=self.get_frames, daemon=True)
        processing_threads = []
        for cam_id, processor in self.camera_processors.items():
            processing_threads.append(threading.Thread(target=processor.run, daemon=True))
        defect_thread = threading.Thread(target=self.defect_processor.run, daemon=True)
        processing_threads.append(defect_thread)
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