import cv2
import numpy as np
import time
from ultralytics import YOLO


def nothing():
    im = cv2.imread('IMG_20250321_134050-crop.jpg')
    print(im.shape)
    im = cv2.resize(im, (1200, 1800))
    cv2.imwrite('test.jpg', im)


if __name__ == '__main__':
    nothing()