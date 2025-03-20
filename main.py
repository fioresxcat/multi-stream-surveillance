import os
import pdb
import cv2
import numpy as np
import torch
from pathlib import Path
from easydict import EasyDict
import time
import yaml
from typing_extensions import List, Dict, Tuple, Optional, Literal, Any
from trackers import BYTETracker, BOTSORT
from methods import ContainerDetector, ContainerInfoDetector, LicensePlateOCR, PPOCRv4Rec
from utils.utils import *


config_env = load_yaml('configs/config_env.yaml')['inference_server']
config_model = load_yaml('configs/config_models.yaml')
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


container_detector = ContainerDetector.get_instance(config_env, config_model['container_detection'])
container_info_detector = ContainerInfoDetector.get_instance(config_env, config_model['container_info_detection'])
license_plate_ocr = LicensePlateOCR.get_instance(config_env, config_model['license_plate_ocr'])
general_ocr = PPOCRv4Rec.get_instance(config_env, config_model['ppocrv4_rec'])


class ContainerInfo:
    def __init__(self, id):
        self.id = id
        self.start_time = None
        self.end_time = None
        self.info = {
            'owner_code': (None, 0),  # value, score
            'container_number': (None, 0),
            'check_digit': (None, 0),
            'container_type': (None, 0),
            'rear_license_plate': (None, 0),
            'front_license_plate': (None, 0),
        }
        self.history = []
        self.direction = {}  # camid: direction



    def get_direction(self, cam_id, frame_w, frame_h):
        if cam_id in self.direction and self.direction[cam_id] is not None:
            return self.direction[cam_id]
        if len(self.history) < 5:
            return None
        
        direction = None
        if cam_id in ['htt', 'hts', 'bst', 'bss']:
            first_bboxes = [el[1] for el in self.history[:5]]
            centers = [(bb[0] + bb[2]) / 2 for bb in first_bboxes]
            num_near_left = sum([1 if center - frame_w//2 < 0 else 0 for center in centers])
            num_near_right = sum([1 if center - frame_w//2 > 0 else 0 for center in centers])
            if cam_id in ['htt']:
                direction = 'out' if num_near_right >= num_near_left else 'in'

        self.direction[cam_id] = direction
        return self.direction[cam_id]


    def update_info(self, extracted_info):
        for label in extracted_info:
            if label not in self.info:
                continue
            value, score = extracted_info[label]
            if score > self.info[label][1]:
            # if True:
                self.info[label] = (value, score)
    

    def get_complete_labels(self, score_threshold):
        complete_labels = []
        for label, (value, score) in self.info.items():
            if value is not None and score > score_threshold:
                complete_labels.append(label)
        return complete_labels


    def get_incomplete_labels(self, score_threshold=0.85, direction=None):
        incomplete_labels = [label for label, (value, score) in self.info.items() if value is None or score < score_threshold]
        
        if direction == 'out':
            incomplete_labels = [label for label in incomplete_labels if label != 'front_license_plate']
        elif direction == 'in':
            exclude_labels = {'owner_code', 'container_number', 'check_digit', 'container_type', 'rear_license_plate'}
            incomplete_labels = [label for label in incomplete_labels if label not in exclude_labels]
        
        return incomplete_labels


class ContainerProcessor:
    def __init__(self, video_sources: dict, fps: int):
        self.cameras = {cam_id: cv2.VideoCapture(cam_source) for cam_id, cam_source in video_sources.items()}
        self.trackers = {cam_id: BOTSORT(args=TRACKER_ARGS, frame_rate=fps) for cam_id in video_sources.keys()}

        self.running = False
        self.im_w = self.cameras['htt'].get(cv2.CAP_PROP_FRAME_WIDTH)
        self.im_h = self.cameras['htt'].get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.fps = fps
        self.database = {}  # container_id: ContainerInfo


    def get_frame(self, cam_id):
        ret, frame = self.cameras[cam_id].read()
        if not ret:
            return None, None
        timestamp = self.cameras[cam_id].get(cv2.CAP_PROP_POS_FRAMES)
        return timestamp, frame
    

    def stop(self):
        self.running = False
        for cam_id in self.cameras:
            self.cameras[cam_id].release()


    def is_container_valid(self, bb, cam_id, container_info: ContainerInfo):
        container_direction = container_info.get_direction(cam_id, self.im_w, self.im_h)
        if container_direction is None:
            return False
        
        xmin, ymin, xmax, ymax = bb
        cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
        if cam_id == 'htt':
            if container_direction == 'out':
                return ymax < self.im_h - 20
            else:
                return cx < self.im_w // 2
        
        raise NotImplementedError(f'Camera {cam_id} is not supported yet')
    

    def extract_container_info(self, container_im, extract_labels=[]):
        extracted_info = {lb: (None, 0) for lb in extract_labels}
        if len(extract_labels) == 0:
            return extracted_info

        BLUR_THRESHOLD = 0.5
        boxes, scores, classes = container_info_detector.predict([container_im])[0]
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
                if blur_score < BLUR_THRESHOLD:
                    lp_text, prob = license_plate_ocr.predict([lp_im])[0]
                    extracted_info[label] = (lp_text, prob)
        
        # ocr field
        batch_images, batch_labels = [], []
        for cl_name in final_boxes.keys():
            if cl_name in ['rear_license_plate', 'front_license_plate']:
                continue
            if cl_name not in extract_labels:
                continue
            box, _ = final_boxes[cl_name]
            field_im = container_im[box[1]:box[3], box[0]:box[2]]
            blur_score = compute_image_blurriness(field_im)
            if blur_score < BLUR_THRESHOLD:
                batch_images.append(field_im)
                batch_labels.append(cl_name)
        results = general_ocr.predict(batch_images)
        for res, label in zip(results, batch_labels):
            text, prob = res
            extracted_info[label] = (text, prob)
        
        return extracted_info



    def process(self):
        self.running = True
        # fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
        # writer = cv2.VideoWriter('test.mp4', fourcc, 25, (850, 480)) # keep aspect ratio
        self.cameras['htt'].set(cv2.CAP_PROP_POS_FRAMES, 0)

        while self.running:
            timestamp, frame = self.get_frame('htt')
            print(f'-------- processing frame {timestamp} ---------'.upper())
            # if timestamp < 2580: 
            #     continue

            boxes, scores, cl_names = container_detector.predict([frame])[0]
            if len(boxes) == 0:
                continue
            boxes, scores, cl_names = sort_box_by_score(boxes, scores, cl_names)
            class_ids = [container_detector.labels.index(cl_name) for cl_name in cl_names]
            
            # track
            xywh_boxes = [xyxy2xywh(box) for box in boxes]
            dets = {'conf': np.array(scores), 'xywh': np.array(xywh_boxes), 
                    'cls': np.array(class_ids), 'xyxy': np.array(boxes)}
            self.trackers['htt'].update(results=dets, img=frame)

            # extract info for ids
            for track in self.trackers['htt'].tracked_stracks:  # for all activated tracks in this frame
                id = track.track_id
                bb = track.xyxy.astype(int)
                bb[::2] = np.clip(bb[::2], 0, self.im_w)
                bb[1::2] = np.clip(bb[1::2], 0, self.im_h)
                bb = bb.tolist()

                if id not in self.database:
                    container_info = ContainerInfo(id)
                    container_info.start_time = timestamp
                    container_info.history.append((timestamp, bb))
                    self.database[id] = container_info
                else:
                    container_info = self.database[id]
                    container_info.history.append((timestamp, bb))

                if self.is_container_valid(bb, 'htt', container_info):
                    container_im = frame[bb[1]:bb[3], bb[0]:bb[2]]
                    direction = container_info.get_direction('htt', self.im_w, self.im_h)
                    incomplete_labels = container_info.get_incomplete_labels(direction=direction)
                    print('label to extract:', incomplete_labels)
                    extracted_info = self.extract_container_info(container_im, extract_labels=incomplete_labels)
                    container_info.update_info(extracted_info)

            for container_id, container_info in self.database.items():
                print(f'container {container_id}: direction: {container_info.direction}, info: {container_info.info}')

            # check keyboard interupt
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # # draw
            # draw_frame = frame.copy()
            # for bb, id in zip(tracked_bboxes, tracked_ids):
            #     cv2.rectangle(draw_frame, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 0), 2)
            #     cv2.putText(draw_frame, str(id), (bb[0], bb[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            # draw_frame = cv2.resize(draw_frame, (850, 480))
            # writer.write(draw_frame)
            # if timestamp == 500:
            #     break

        # writer.release()
        self.stop()


def main():
    fps = 25
    video_sources = {
        'htt': 'test_files/hongtraitruoc-part2.mp4'
    }
    processor = ContainerProcessor(video_sources, fps)
    processor.process()
    processor.stop()


def test():
    from fast_alpr.default_ocr import DefaultOCR

    im = cv2.imread('test1.jpg')

    # fast_alpr_ocr = DefaultOCR('global-plates-mobile-vit-v2-model')
    # res = fast_alpr_ocr.predict(im)
    # print(res)

    # res = license_plate_ocr.predict([im])
    # print(res)

    # boxes, scores, classes = container_info_detector.predict([im])[0]
    # for bb in boxes:
    #     xmin, ymin, xmax, ymax = bb
    #     crop = im[ymin:ymax, xmin:xmax]
    # cv2.imwrite('test1.jpg', crop) 



if __name__ == "__main__":
    # main()
    test()