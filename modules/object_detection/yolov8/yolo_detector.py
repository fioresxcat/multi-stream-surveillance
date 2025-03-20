import pdb
import cv2
import numpy as np

from utils.utils import total_time, sort_bbs
from .base_object_detection import BaseObjectDetection, LetterBox


class YOLODetector(BaseObjectDetection):
    instance = None
    
    def __init__(self, common_config, model_config):
        super(YOLODetector, self).__init__(common_config, model_config)
        input_shape = model_config['input_shape']
        self.resizer = LetterBox((input_shape[0], input_shape[1]), auto=False)
        self.labels = []
        
        
    @staticmethod
    def get_instance(common_config, model_config):
        if YOLODetector.instance is None:
            YOLODetector.instance = YOLODetector(common_config, model_config)
        return YOLODetector.instance
    
    
    def preprocess(self, image):
        image = self.resizer(image)
        image = image[..., ::-1].transpose((2, 0, 1))
        image = np.ascontiguousarray(image, dtype='float32')  # contiguous
        image /= 255  # 0 - 255 to 0.0 - 1.0
        return image

   
    
    def predict(self, images, metadata={}):
        outputs, batch_images, batch_shapes = [], [], []
        for im_index, im in enumerate(images):
            batch_images.append(self.preprocess(im))
            batch_shapes.append(im.shape[:2])
            if len(batch_images) == self.model_config['max_batch_size'] or (im_index == len(images) - 1):
                batch_images = np.array(batch_images).astype(np.float32)
                output_dict = self.request(batch_images)
                outputs.extend(output_dict.as_numpy(self.model_config['output_name']))
                metadata = self.add_metadata(metadata, 1, len(batch_images))
                batch_images = []
        assert len(batch_images) == 0

        results = [([], [], []) for _ in range(len(images))]  # boxes, scores, classes
        if len(outputs) > 0:
            outputs = np.array(outputs)
            detections = self.non_max_suppression(
                outputs,
                conf_thres=self.model_config['conf_threshold'],
                iou=self.model_config['iou_threshold'],
                max_nms=30000
            )
            del outputs
            for det_index, detection in enumerate(detections):  # detection result for each image
                boxes, scores, class_names = [], [], []
                if len(detection) != 0:
                    boxes, scores, class_ids = detection[:, :4], detection[:, 4], detection[:, 5]
                    input_shape = self.model_config['input_shape'][:2]
                    orig_shape = batch_shapes[det_index]
                    boxes = self.scale_boxes(input_shape, boxes, orig_shape)
                    boxes = boxes.astype(np.int32)
                    class_names = [self.labels[int(class_id)] for class_id in class_ids]
                results[det_index] = (boxes, scores, class_names)
                    
        return results