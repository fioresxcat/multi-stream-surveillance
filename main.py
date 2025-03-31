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
from collections import deque
from typing_extensions import List, Dict, Tuple, Optional, Literal, Any
import multiprocessing
from multiprocessing import Process, Queue, Event, Manager

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
setup_logging(log_dir, log_file='app.log', level=logging.DEBUG, enabled_cameras=['htt'])
logger = logging.getLogger('main')

# some constants
CAMERA_MODE = 'video' # 'video' or 'stream'

class ContainerProcessor:
    def __init__(self, video_sources: dict, skip_time: float, ocr_cams: List[str], defect_cams: List[str]):
        self.skip_time = skip_time
        self.ocr_cams = ocr_cams
        self.defect_cams = defect_cams
        self.is_running = False
        self.final_results = []
        self.container_count = 0

        self.manager = multiprocessing.Manager()
        self.container_detected_event = self.manager.dict({cam_id: False for cam_id in video_sources.keys()})

        # Modularized setup
        self._setup_queues(video_sources)
        self._setup_processors(video_sources)


    def _setup_queues(self, video_sources):
        self.ocr_results_queues = self.manager.dict({cam_id: self.manager.Queue() for cam_id in self.ocr_cams})
        self.defect_results_queue = self.manager.dict({cam_id: self.manager.Queue() for cam_id in self.defect_cams})



    def _setup_processors(self, video_sources):
        self.ocr_camera_processors = {}
        for cam_id in self.ocr_cams:
            cam_src = video_sources[cam_id]
            self.ocr_camera_processors[cam_id] = OCRCameraProcessor(
                cam_id, cam_src, self.skip_time, 
                self.ocr_results_queues[cam_id], self.container_detected_event, 
                config_inference_server, config_model
            )

        self.defect_camera_processors = {}
        for cam_id in self.defect_cams:
            cam_src = video_sources[cam_id]
            self.defect_camera_processors[cam_id] = DefectCameraProcessor(
                cam_id, cam_src, self.skip_time, 
                self.defect_results_queue[cam_id], self.container_detected_event, 
                self.ocr_cams, config_inference_server, config_model
            )


    def match_results(self):
        def _queues_are_empty():
            return any(queue.empty() for queue in self.ocr_results_queues.values()) or \
                any(queue.empty() for queue in self.defect_results_queue.values())
        
        def _collect_results(results_queues: List[Queue]):
            results = []
            for queue in results_queues.values():
                results.append(queue.get())
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


    def run(self):
        self.is_running = True
        # multiprocessing.set_start_method("spawn", force=True)
        
        processes = []
        processes.extend(Process(target=processor.run) for processor in self.ocr_camera_processors.values())
        processes.extend(Process(target=processor.run) for processor in self.defect_camera_processors.values())

        threads = [threading.Thread(target=self.match_results, daemon=True)]

        for process in processes:
            process.start()
        for thread in threads:
            thread.start()

        for process in processes:
            process.join()
        for thread in threads:
            thread.join()

        self.stop()


def main():
    multiprocessing.set_start_method("spawn", force=True)
    fps = 25
    video_sources = {
        'htt-ocr': 'test_files/hongtraitruoc-cut611.mp4',
        'hts-ocr': 'test_files/hongtraisau-cut611.mp4',
        'hps-defect': 'test_files/hongphaisau-cut611.mp4',
        # 'htt-defect': 'test_files/hongtraitruoc-cut610_longer.mp4',
        # 'hts-defect': 'test_files/hongtraisau-cut610_longer.mp4',

        # 'hps-defect': 'test_files/hongphaisau-21032025-cut1.mp4',
        # 'bst-ocr': 'test_files/biensotruoc-21032025-cut1.mp4',
        # 'bss-ocr': 'test_files/biensosau-21032025-cut1.mp4',
    }
    ocr_cams = [
        'htt-ocr', 
        'hts-ocr',
        # 'bst-ocr',
        # 'bss-ocr'
    ]
    defect_cams = [
        # 'hps-defect', 
        # 'htt-defect', 
        # 'hts-defect',
    ]
    skip_time = 0.15 # seconds
    processor = ContainerProcessor(video_sources, skip_time, ocr_cams, defect_cams)
    processor.run()
    processor.stop()



if __name__ == "__main__":
    main()