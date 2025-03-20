import pdb
import cv2
import numpy as np

from utils.utils import total_time, sort_bbs
from modules.base import BaseModule


class LicensePlateOCR(BaseModule):
    instance = None
    
    def __init__(self, common_config, model_config):
        super(LicensePlateOCR, self).__init__(common_config, model_config)
        self.im_h, self.im_w = model_config['input_shape'][0], model_config['input_shape'][1]
        self.max_plate_slots = 9
        self.alphabet = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_'


    @staticmethod
    def get_instance(common_config, model_config):
        if LicensePlateOCR.instance is None:
            LicensePlateOCR.instance = LicensePlateOCR(common_config, model_config)
        return LicensePlateOCR.instance
    

    def preprocess_image(self, image, img_height, img_width):
        im = cv2.resize(image, (img_width, img_height), interpolation=cv2.INTER_LINEAR)
        im = np.expand_dims(im, axis=-1)
        return im


    def postprocess_output(self, model_output: np.ndarray, max_plate_slots: int,
                            model_alphabet: str, return_confidence: bool = False):
        predictions = model_output.reshape((-1, max_plate_slots, len(model_alphabet)))
        prediction_indices = np.argmax(predictions, axis=-1)
        alphabet_array = np.array(list(model_alphabet))
        plate_chars = alphabet_array[prediction_indices]
        plates: list[str] = np.apply_along_axis("".join, 1, plate_chars).tolist()
        if return_confidence:
            probs = np.max(predictions, axis=-1)
            return plates, probs
        return plates


    def predict(self, images, metadata={}):
        batch_images = []
        outputs = []
        for i, im in enumerate(images):
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            im = self.preprocess_image(im, self.im_h, self.im_w)
            batch_images.append(im)
            if len(batch_images) == self.model_config['max_batch_size'] or i == len(images) - 1:
                batch_images = np.array(batch_images)
                output_dict = self.request(batch_images)
                output = output_dict.as_numpy(self.model_config['output_name'])
                outputs.extend(output)
                batch_images = []
        assert len(batch_images) == 0
        assert len(outputs) == len(images)
        outputs = np.array(outputs)
        texts, probs = self.postprocess_output(outputs, self.max_plate_slots, self.alphabet, return_confidence=True)
        results = []
        for text, prob in zip(texts, probs):
            text = text.replace("_", "")
            results.append((text, np.min(prob)))
        return results