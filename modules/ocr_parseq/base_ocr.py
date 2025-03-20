import os
import pdb
import cv2
import numpy as np
from PIL import Image

from modules.base import BaseModule



class BaseOCR(BaseModule):
    def __init__(self, common_config, model_config):
        super(BaseOCR, self).__init__(common_config, model_config)
        self.input_shape = self.model_config['input_shape']
        self.input_type = self.model_config['input_type']
        self.max_sequence_length = self.model_config['max_sequence_length']
        charset = self.model_config['charset']
        self.charset_list = ['[E]'] + list(tuple(charset))
        
    
    def resize(self, im):
        height, width = self.input_shape[:2]
        h, w, d = im.shape
        unpad_im = cv2.resize(im, (int(height*w/h), height), interpolation=cv2.INTER_AREA)
        if unpad_im.shape[1] > width:
            im = cv2.resize(im, (width, height), interpolation=cv2.INTER_AREA)
        else:
            im = cv2.copyMakeBorder(unpad_im, 0, 0, 0, width-int(height*w/h), cv2.BORDER_CONSTANT, value=[0, 0, 0])
        return im
    
    
    def decode(self, p):
        cands = []
        for cand in p:
            if np.argmax(cand) == 0:
                break
            cands.append(cand)
        return cands

    
    def request_batch(self, images):
        result = []
        if len(images) == 0:
            return result
        output = self.request(images)
        output = np.array(output.as_numpy(self.model_config['output_name']))
        ps = np.exp(output)/np.expand_dims(np.sum(np.exp(output), axis=-1), axis=-1)
        for p in ps:
            result.append(self.decode(p))
        return result

    
    def index_to_word(self, output):
        res = ''
        for probs in output:
            if np.argmax(probs) == 0:
                break
            else:
                res += self.charset_list[np.argmax(probs)]
        return res
    
    
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
                resized_image = self.resize(image)
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
