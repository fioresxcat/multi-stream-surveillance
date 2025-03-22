import cv2
import numpy as np
import time
import pdb



def nothing():
    cap = cv2.VideoCapture('rtsp://admin:Abcd7890!@noccongtruoc.cameraddns.net:8556/Streaming/Channels/101')
    if not cap.isOpened():
        print("Failed to open the RTSP stream.")
    else:
        print("RTSP stream opened successfully.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        print(frame.shape)
        cv2.imwrite('test.jpg', frame)
        pdb.set_trace()


if __name__ == '__main__':
    nothing()