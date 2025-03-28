import os
import pdb
import cv2
import numpy as np
import torch
from pathlib import Path
from easydict import EasyDict
import time
import logging
import yaml
from queue import Queue
from collections import deque
from typing_extensions import List, Dict, Tuple, Optional, Literal, Any

from utils.utils import load_yaml
from utils.logging_config import setup_logging
from ocr_camera import *
from defect_camera import *

# load config
config_env = load_yaml('configs/config_env.yaml')
config_inference_server = config_env['inference_server']
config_model = load_yaml('configs/config_models.yaml')

# setup logging
log_dir = 'logs'
setup_logging(log_dir, log_file='app.log', level=logging.DEBUG, enabled_cameras=['hps'])
logger = logging.getLogger('main')

# some constants
CAMERA_MODE = 'video' # 'video' or 'stream'


class ContainerProcessor:
    def __init__(self, video_sources: dict, fps: int, skip_frame: int, ocr_cams: List[str], defect_cams: List[str]):
        self.fps = fps
        self.skip_frame = skip_frame
        self.ocr_cams = ocr_cams
        self.defect_cams = defect_cams
        self.container_detected_event = {cam_id: threading.Event() for cam_id in video_sources.keys()}
        self.is_running = False
        self.final_results = []
        self.container_count = 0

        # Modularized setup
        self._setup_captures(video_sources)
        self._setup_queues(video_sources)
        self._setup_processors(video_sources)


    def _setup_captures(self, video_sources):
        self.caps = {cam_id: cv2.VideoCapture(cam_source) for cam_id, cam_source in video_sources.items()}
        self.frame_sizes = {
            cam_id: (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            for cam_id, cap in self.caps.items()
        }


    def _setup_queues(self, video_sources):
        self.frame_queues = {cam_id: Queue(maxsize=10) for cam_id in video_sources.keys()}
        self.ocr_results_queues = {cam_id: deque() for cam_id in self.ocr_cams}
        self.defect_results_queue = {cam_id: deque() for cam_id in self.defect_cams}


    def _setup_processors(self, video_sources):
        self.ocr_camera_processors = {}
        for cam_id in self.ocr_cams:
            self.ocr_camera_processors[cam_id] = OCRCameraProcessor(
                cam_id, self.fps, self.frame_sizes[cam_id], self.skip_frame, 
                self.frame_queues[cam_id], self.ocr_results_queues[cam_id], 
                self.container_detected_event, config_inference_server, config_model
            )

        self.defect_camera_processors = {}
        for cam_id in self.defect_cams:
            self.defect_camera_processors[cam_id] = DefectCameraProcessor(
                cam_id, self.fps, self.frame_sizes[cam_id], self.skip_frame, 
                self.frame_queues[cam_id], self.defect_results_queue[cam_id], 
                self.container_detected_event, self.ocr_cams, config_inference_server, config_model
            )


    
    def get_frames(self):
        is_stopped = {cam_id: False for cam_id in self.caps.keys()}
        
        # # Set the initial position of the video capture
        # for cam_id, cap in self.caps.items():
        #     cap.set(cv2.CAP_PROP_POS_MSEC, 15)

        while self.is_running:
            for cam_id, cap in self.caps.items():
                cam_queue: Queue = self.frame_queues[cam_id]
                if CAMERA_MODE == 'video' and cam_queue.full():  # if mode is video, process all frames
                    continue
                if is_stopped[cam_id]:
                    frame = np.full((self.frame_sizes[cam_id][1], self.frame_sizes[cam_id][0], 3), 255, dtype=np.uint8)
                else:
                    ret, frame = cap.read()
                    if not ret:
                        logger.info(f"Failed to read frame from {cam_id}")
                        is_stopped[cam_id] = True
                        continue
                frame_index = cap.get(cv2.CAP_PROP_POS_FRAMES)
                if frame_index % self.skip_frame != 0:
                    continue
                if not is_stopped[cam_id]:
                    timestamp = round(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0, 2)  # Convert to seconds
                else:
                    timestamp += 1 / self.fps  # Increment timestamp to avoid duplicates
                cam_queue.put({'timestamp': timestamp, 'frame': frame, 'frame_index': frame_index})  # notice this



    def match_results(self):
        def _queues_are_empty():
            return any(len(queue) == 0 for queue in self.ocr_results_queues.values()) or \
                any(len(queue) == 0 for queue in self.defect_results_queue.values())
        
        def _collect_results(results_queues):
            results = []
            for queue in results_queues.values():
                results.append(queue.popleft())
            return results

        def _combine_info(ocr_results, defect_results):
            final_ocr_info = {}
            for ocr_result in ocr_results:
                info = ocr_result['info']
                for label, (value, score) in info.items():
                    if label not in final_ocr_info or score > final_ocr_info[label][1]:
                        final_ocr_info[label] = (value, score)

            final_defect_info = {}
            for defect_result in defect_results:
                info: List[dict] = defect_result['info']
                cam_id = defect_result['camera_id']
                final_defect_info[cam_id] = info

            return {'ocr_info': final_ocr_info, 'defect_info': final_defect_info}

        
        def _write_log(info):
            ocr_info: Dict = info['ocr_info']
            defect_info: Dict = info['defect_info']
            for cam_id, defects in defect_info.items():
                for defect in defects:
                    defect.pop('image', None)  # remove image from defect info
                    defect.pop('boxes', None)  # remove boxes from defect info
                    defect.pop('scores', None)  # remove scores from defect info
            with open(log_path, 'a') as f:
                f.write(f'--------------- Container {self.container_count} ---------------\n')
                f.write(f'OCR Info: {ocr_info}\n')
                f.write(f'Defect Info: {defect_info}\n\n')


        log_path = os.path.join(log_dir, 'result.log')
        with open(log_path, 'w') as f:
            f.write('')
        while self.is_running:
            time.sleep(0.1)
            if _queues_are_empty():
                continue
            ocr_results = _collect_results(self.ocr_results_queues)
            defect_results = _collect_results(self.defect_results_queue)
            info = _combine_info(ocr_results, defect_results)
            self.container_count += 1
            _write_log(info)


    def stop(self):
        self.is_running = False
        for cam_id, cap in self.caps.items():
            cap.release()


    def run(self):
        self.is_running = True

        threads = []
        threads.append(threading.Thread(target=self.get_frames, daemon=True))
        threads.extend(threading.Thread(target=processor.run, daemon=True) for processor in self.ocr_camera_processors.values())
        threads.extend(threading.Thread(target=processor.run, daemon=True) for processor in self.defect_camera_processors.values())
        threads.append(threading.Thread(target=self.match_results, daemon=True))

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        self.stop()


def main():
    fps = 25
    video_sources = {
        # 'htt': 'test_files/hongtraitruoc-cut610_longer.mp4',
        # 'hts': 'test_files/hongtraisau-cut610_longer.mp4',
        # 'hps': 'test_files/hongphaisau-cut610_longer.mp4',

        'hps': 'test_files/hongphaisau-21032025-cut1.mp4',
        'bst': 'test_files/biensotruoc-21032025-cut1.mp4',
        'bss': 'test_files/biensosau-21032025-cut1.mp4',
    }
    ocr_cams = ['bst', 'bss']
    defect_cams = ['hps']
    skip_frame = int(0.15*fps) # num frames
    processor = ContainerProcessor(video_sources, fps, skip_frame, ocr_cams, defect_cams)
    processor.run()
    processor.stop()



if __name__ == "__main__":
    main()