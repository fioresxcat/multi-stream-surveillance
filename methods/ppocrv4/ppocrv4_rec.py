import pdb
import cv2
import numpy as np
import os
import math
import pkgutil

from utils.utils import total_time, sort_bbs
from modules.base import BaseModule


class BaseRecLabelDecode:
    def __init__(self, character_dict):
        self.character = self.add_special_char(character_dict + list('  '))
        self.dict = dict(enumerate(self.character))

    def add_special_char(self, dict_character):
        return dict_character

    def decode(self, text_index, text_prob=None, remove_duplicate=False):
        """ convert text-index into text-label. """
        result_list = []
        ignored_tokens = self.get_ignored_tokens()
        for batch_idx in range(len(text_index)):
            char_list = []
            conf_list = []
            for idx in range(len(text_index[batch_idx])):
                if text_index[batch_idx][idx] in ignored_tokens:
                    continue
                if remove_duplicate:
                    if idx > 0 and text_index[batch_idx][idx - 1] == text_index[batch_idx][idx]:
                        continue
                char_list.append(self.character[int(text_index[batch_idx][idx])])
                if text_prob is not None:
                    conf_list.append(text_prob[batch_idx][idx])
                else:
                    conf_list.append(1)
            text = ''.join(char_list)
            result_list.append((text, np.mean(conf_list) if conf_list else np.nan))
        return result_list

    def get_ignored_tokens(self):
        return [0]  # for ctc blank


class CTCLabelDecode(BaseRecLabelDecode):
    """ Convert between text-label and text-index """

    def __init__(self,character_dict):
        super().__init__(character_dict)
        self.char_mask = None

    def __call__(self, preds, label=None, *args, **kwargs):
        if self.char_mask is not None:
            preds[:, :, ~self.char_mask] = 0
        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)
        text = self.decode(preds_idx, preds_prob, remove_duplicate=True)
        if label is None:
            return text
        label = self.decode(label)
        return text, label

    def add_special_char(self, dict_character):
        dict_character = ['blank'] + dict_character
        return dict_character
    

class PPOCRv4Rec(BaseModule):
    instance = None
    
    def __init__(self, common_config, model_config):
        super(PPOCRv4Rec, self).__init__(common_config, model_config)
        self.rec_image_shape = [3, 48, 320]
        self.postprocess_op = CTCLabelDecode(character_dict=self.get_character_dict('char_dict.txt'))


    def get_instance(common_config, model_config):
        if PPOCRv4Rec.instance is None:
            PPOCRv4Rec.instance = PPOCRv4Rec(common_config, model_config)
        return PPOCRv4Rec.instance
    

    def get_character_dict(self, dict_path):
        return pkgutil.get_data(__name__, dict_path).decode('utf-8').splitlines()
    

    def resize_norm_img(self, img, max_wh_ratio):
        imgC, imgH, imgW = self.rec_image_shape
        imgW = int(imgH * max_wh_ratio)
        resized_w = min(imgW, int(math.ceil(imgH * img.shape[1] / img.shape[0])))
        resized_image = cv2.resize(
            img,
            (resized_w, imgH),
            interpolation=cv2.INTER_CUBIC
        ).astype('float32').transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        return padding_im


    def predict(self, images, metadata={}):
        img_num = len(images)
        # Sorting can speed up the recognition process
        indices = np.argsort([img.shape[1] / img.shape[0] for img in images])

        rec_res = [['', 0.0]] * img_num
        batch_num = self.model_config['max_batch_size']
        for beg_img_no in range(0, img_num, batch_num):
            end_img_no = min(img_num, beg_img_no + batch_num)
            h, w = images[indices[end_img_no - 1]].shape[:2]
            norm_img_batch = np.concatenate([
                self.resize_norm_img(images[indices[ino]], w / h)[None]
                for ino in range(beg_img_no, end_img_no)
            ]).copy()

            # if norm_img_batch.shape[3] < 320:
            #     pad_width = 320 - norm_img_batch.shape[3]
            #     norm_img_batch = np.pad(norm_img_batch, ((0, 0), (0, 0), (0, 0), (0, pad_width)), mode='constant')

            output_dict = self.request(norm_img_batch)
            output = output_dict.as_numpy(self.model_config['output_name'])
            for rno, res in enumerate(self.postprocess_op(output)):
                rec_res[indices[beg_img_no + rno]] = res
        return rec_res
    