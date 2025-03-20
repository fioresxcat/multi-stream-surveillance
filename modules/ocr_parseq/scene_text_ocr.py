import os
import pdb
import cv2
import numpy as np
from PIL import Image

from .base_ocr import BaseOCR
from utils.utils import total_time



class SceneTextOCR(BaseOCR):
    instance = None
    
    def __init__(self, common_config, model_config):
        super(SceneTextOCR, self).__init__(common_config, model_config)
        
        
    @staticmethod
    def get_instance(common_config, model_config):
        if SceneTextOCR.instance is None:
            SceneTextOCR.instance = SceneTextOCR(common_config, model_config)
        return SceneTextOCR.instance
    
    
    def predict_batch(self, list_list_boxes, metadata):
        batch_images = []
        page_lengths = []
        list_raw_words = []
        list_raw_cands = []
        for i in range(len(list_list_boxes)):
            list_raw_words.append([])
            list_raw_cands.append([])
            page_lengths.append(len(list_list_boxes[i]))
            for j, image in enumerate(list_list_boxes[i]):
                resized_image = cv2.resize(image, (self.input_shape[1], self.input_shape[0]), interpolation=cv2.INTER_CUBIC)
                processed_image = np.transpose(resized_image/255., (2, 0, 1)).astype(np.float32)
                normalized_image = (processed_image - 0.5) / 0.5
                batch_images.append(normalized_image)

        batch_images_length = len(batch_images)
        #while len(batch_images) % self.model_config['max_batch_size'] != 0:
        #    batch_images.append(batch_images[0])

        batch_images = np.array(batch_images)
        text_output = []
        if len(batch_images) != 0:
            index = 0
            while index < len(batch_images):
                #print(len(batch_images[index:index+self.model_config['max_batch_size']]))
                text_output += self.request_batch(batch_images[index:index+self.model_config['max_batch_size']])
                metadata = self.add_metadata(metadata, 1, self.model_config['max_batch_size'])
                index += self.model_config['max_batch_size']
        text_output = text_output[:batch_images_length]
        
        cnt_index = 0
        for i, page_length in enumerate(page_lengths):
            list_raw_cands[i] = text_output[cnt_index:cnt_index+page_length]
            for j in range(page_length):
                list_raw_words[i].append(self.index_to_word(text_output[cnt_index+j]))
            cnt_index += page_length
        return list_raw_words, list_raw_cands

