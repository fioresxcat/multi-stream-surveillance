import pdb
import cv2
import numpy as np

from utils.utils import total_time, sort_bbs
from modules.base import BaseModule
from modules.ocr_parseq.base_ocr import BaseOCR


class ParseqLicensePlateOCR(BaseOCR):
    instance = None
    
    def __init__(self, common_config, model_config):
        super().__init__(common_config, model_config)


    @staticmethod
    def get_instance(common_config, model_config):
        if ParseqLicensePlateOCR.instance is None:
            ParseqLicensePlateOCR.instance = ParseqLicensePlateOCR(common_config, model_config)
        return ParseqLicensePlateOCR.instance
    

    def predict(self, images, metadata={}):
        batch_images = []
        for j, image in enumerate(images):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            resized_image = cv2.resize(image, (self.input_shape[1], self.input_shape[0]), interpolation=cv2.INTER_CUBIC)
            processed_image = np.transpose(resized_image/255., (2, 0, 1)).astype(np.float32)
            normalized_image = (processed_image - 0.5) / 0.5
            batch_images.append(normalized_image)

        batch_images = np.array(batch_images)
        text_output = []
        if len(batch_images) != 0:
            index = 0
            while index < len(batch_images):
                text_output += self.request_batch(batch_images[index:index+self.model_config['max_batch_size']])
                index += self.model_config['max_batch_size']
        
        results = []
        for cand in text_output:
            word, prob = self.index_to_word(cand)
            results.append((word, prob))

        return results