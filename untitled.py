import cv2
import numpy as np
import time

def nothing():
    cap = cv2.VideoCapture('test_files/biensosau-part2.mkv')
    cnt = 0
    while True:
        ret, frame = cap.read()
        start = time.perf_counter()
        print(cap.get(cv2.CAP_PROP_POS_MSEC))
        # print(cnt)
        print('time:', time.perf_counter() - start)
        if not ret:
            break
        cnt += 1


if __name__ == '__main__':
    nothing()