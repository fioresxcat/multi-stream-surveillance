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
    def __init__(self, cam_id, id):
        self.cam_id = cam_id
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
        self.direction = None
        self.is_done = False
        self.score_threshold = 0.5
        self.frame_w, self.frame_h = 2688, 1520


    def update_info(self, new_info):
        for label in new_info:
            if label not in self.info:
                continue
            value, score = new_info[label]
            if score > self.info[label][1]:
            # if True:
                self.info[label] = (value, score)
    

    def update_history(self, time_stamp, bb):
        self.history.append((time_stamp, bb))
        if len(self.history) > 5:
            if self.cam_id in ['cam1', 'cam2']:
                first_bboxes = [el[1] for el in self.history[:5]]
                centers = [(bb[0] + bb[2]) / 2 for bb in first_bboxes]
                num_near_left = sum([1 if center - self.frame_w//2 < 0 else 0 for center in centers])
                num_near_right = sum([1 if center - self.frame_w//2 > 0 else 0 for center in centers])
                if self.cam_id in ['cam1']:
                    self.direction = 'out' if num_near_right >= num_near_left else 'in'
                elif self.cam_id in ['cam2']:
                    self.direction = 'in' if num_near_right >= num_near_left else 'out'
            else:
                raise NotImplementedError(f'Camera {self.cam_id} is not supported yet')


    def get_complete_labels(self):
        complete_labels = []
        for label, (value, score) in self.info.items():
            if value is not None and score > self.score_threshold:
                complete_labels.append(label)
        return complete_labels


    def get_incomplete_labels(self):
        incomplete_labels = [label for label, (value, score) in self.info.items() if value is None or score < self.score_threshold]
    
        if self.direction == 'out':
            incomplete_labels = [label for label in incomplete_labels if label != 'front_license_plate']
        elif self.direction == 'in':
            exclude_labels = {'owner_code', 'container_number', 'check_digit', 'container_type', 'rear_license_plate'}
            incomplete_labels = [label for label in incomplete_labels if label not in exclude_labels]
        
        return incomplete_labels
    

    def is_full(self):
        if self.direction is None:
            return False
        required_labels = ['front_license_plate'] if self.direction == 'in' else ['owner_code', 'container_number', 'check_digit', 'container_type', 'rear_license_plate']
        return set(self.get_complete_labels()) == set(required_labels)



class ContainerProcessor:
    def __init__(self, video_sources: dict, fps: int):
        self.cameras = {cam_id: cv2.VideoCapture(cam_source) for cam_id, cam_source in video_sources.items()}
        self.trackers = {cam_id: BOTSORT(args=TRACKER_ARGS, frame_rate=fps) for cam_id in video_sources.keys()}

        self.running = False
        self.im_w = self.cameras['cam1'].get(cv2.CAP_PROP_FRAME_WIDTH)
        self.im_h = self.cameras['cam1'].get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.fps = fps
        self.database = {cam_id: {} for cam_id in self.cameras.keys()}  # for each cam: container_id: ContainerInfo
        self.final_results = []


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
        direction = container_info.direction
        if direction is None:
            return False
        
        xmin, ymin, xmax, ymax = bb
        cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
        if cam_id == 'cam1':
            if direction == 'out':
                return ymax < self.im_h - 20
            elif direction == 'in':
                return cx < self.im_w // 2
        elif cam_id == 'cam2':
            if direction == 'in':
                return xmin > self.im_w // 2
            elif direction == 'out':
                return ymax < self.im_h - 20
        
        raise NotImplementedError(f'Camera {cam_id} is not supported yet')
    

    def extract_container_info(self, container_im, extract_labels=[]):
        extracted_info = {lb: (None, 0) for lb in extract_labels}
        if len(extract_labels) == 0:
            return extracted_info

        BLUR_THRESHOLD = 1
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
        self.cameras['cam1'].set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.cameras['cam2'].set(cv2.CAP_PROP_POS_FRAMES, 0)


        while self.running:
            timestamp1, frame1 = self.get_frame('cam1')
            timestamp2, frame2 = self.get_frame('cam2')
            print(f'-------- processing frame {timestamp1} ---------'.upper())

            boxes, scores, cl_names = container_detector.predict([frame1])[0]
            if len(boxes) > 0:
                # ---------- cam 1 process ---------------
                boxes, scores, cl_names = sort_box_by_score(boxes, scores, cl_names)
                class_ids = [container_detector.labels.index(cl_name) for cl_name in cl_names]
                
                # track
                xywh_boxes = [xyxy2xywh(box) for box in boxes]
                dets = {'conf': np.array(scores), 'xywh': np.array(xywh_boxes), 
                        'cls': np.array(class_ids), 'xyxy': np.array(boxes)}
                self.trackers['cam1'].update(results=dets, img=frame1)

                # extract info for ids
                for track in self.trackers['cam1'].tracked_stracks:  # for all activated tracks in this frame
                    id = track.track_id
                    bb = track.xyxy.astype(int)
                    bb[::2] = np.clip(bb[::2], 0, self.im_w)
                    bb[1::2] = np.clip(bb[1::2], 0, self.im_h)
                    bb = bb.tolist()

                    if id not in self.database['cam1']:
                        container_info = ContainerInfo('cam1', id)
                        container_info.start_time = timestamp1
                        container_info.update_history(timestamp1, bb)
                        self.database['cam1'][id] = container_info
                    else:
                        container_info = self.database['cam1'][id]
                        container_info.update_history(timestamp1, bb)

                    if self.is_container_valid(bb, 'cam1', container_info) and not container_info.is_done:
                        container_im = frame1[bb[1]:bb[3], bb[0]:bb[2]]
                        incomplete_labels = container_info.get_incomplete_labels()
                        print('label to extract:', incomplete_labels)
                        extracted_info = self.extract_container_info(container_im, extract_labels=incomplete_labels)
                        container_info.update_info(extracted_info)

            # --------------- cam 2 process ---------------
            boxes, scores, cl_names = container_detector.predict([frame2])[0]
            if len(boxes) > 0:
                boxes, scores, cl_names = sort_box_by_score(boxes, scores, cl_names)
                class_ids = [container_detector.labels.index(cl_name) for cl_name in cl_names]
                
                # track
                xywh_boxes = [xyxy2xywh(box) for box in boxes]
                dets = {'conf': np.array(scores), 'xywh': np.array(xywh_boxes), 
                        'cls': np.array(class_ids), 'xyxy': np.array(boxes)}
                self.trackers['cam2'].update(results=dets, img=frame1)

                # extract info for ids
                for track in self.trackers['cam2'].tracked_stracks:  # for all activated tracks in this frame
                    id = track.track_id
                    bb = track.xyxy.astype(int)
                    bb[::2] = np.clip(bb[::2], 0, self.im_w)
                    bb[1::2] = np.clip(bb[1::2], 0, self.im_h)
                    bb = bb.tolist()

                    if id not in self.database['cam2']:
                        container_info = ContainerInfo('cam2', id)
                        container_info.start_time = timestamp2
                        container_info.update_history(timestamp2, bb)
                        self.database['cam2'][id] = container_info
                    else:
                        container_info = self.database['cam2'][id]
                        container_info.update_history(timestamp2, bb)

                    if self.is_container_valid(bb, 'cam2', container_info) and not container_info.is_done:
                        container_im = frame2[bb[1]:bb[3], bb[0]:bb[2]]
                        incomplete_labels = container_info.get_incomplete_labels()
                        print('label to extract:', incomplete_labels)
                        extracted_info = self.extract_container_info(container_im, extract_labels=incomplete_labels)
                        container_info.update_info(extracted_info)


            # --------------- merge from cam1 and cam2 ---------------
            for id1, container_info_1 in self.database['cam1'].items():
                start_time1 = container_info_1.start_time
                if container_info_1.is_done:
                    continue
                for id2, container_info_2 in self.database['cam2'].items():
                    start_time2 = container_info_2.start_time
                    if container_info_2.is_done:
                        continue
                    if abs(start_time1 - start_time2) < int(3*self.fps):
                        if container_info_1.is_full() and container_info_2.is_full():
                            info = {}
                            for field, (field_value, field_score) in container_info_1.info.items():
                                if field_value is not None:
                                    info[field] = field_value
                            for field, (field_value, field_score) in container_info_2.info.items():
                                if field_value is not None:
                                    info[field] = field_value
                            self.final_results.append(info)
                            container_info_1.is_done = True
                            container_info_2.is_done = True
                            # self.database['cam1'].pop(id1)
                            # self.database['cam2'].pop(id2)
                            pdb.set_trace()
                        else:
                            print(f'Find matching container {id1} and {id2} but not full')

            for camid in self.database:
                print('CAMERA:', camid)
                for container_id, container_info in self.database[camid].items():
                    print(f'container {container_id}: start frame: {container_info.start_time} direction: {container_info.direction}, info: {container_info.info}')
            

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
        'cam1': 'test_files/hongtraitruoc-part2_cut1.mp4',
        'cam2': 'test_files/hongtraisau-part2_cut1.mp4',
    }
    processor = ContainerProcessor(video_sources, fps)
    processor.process()
    processor.stop()


def test():
    from fast_alpr.default_ocr import DefaultOCR

    im = cv2.imread('test.jpg')

    # fast_alpr_ocr = DefaultOCR('global-plates-mobile-vit-v2-model')
    # res = fast_alpr_ocr.predict(im)
    # print(res)

    res = license_plate_ocr.predict([im,im])
    print(res)    

if __name__ == "__main__":
    main()
    # test()