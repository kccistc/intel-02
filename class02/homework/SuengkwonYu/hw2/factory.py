#!/usr/bin/env python3

import os
import threading
from queue import Empty, Queue
from time import sleep

import cv2
import numpy as np

FORCE_STOP = False

def thread_cam1(q):
    file_path = './resources/conveyor.mp4'
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        print("Error: Could not open the video")
        exit()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if frame is None:
            break
        
        q.put(('Cam1 live', frame))
        
    cap.release()
    q.put(('DONE', None))


def thread_cam2(q):
    file_path = './resources/conveyor.mp4'
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        print("Error: Could not open the video")
        exit()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if frame is None:
            break
        
        q.put(('Cam2 live', frame))
        
    cap.release()
    q.put(('DONE', None))

def imshow(title, frame, pos=None):
    cv2.namedWindow(title)
    if pos:
        cv2.moveWindow(title, pos[0], pos[1])
    cv2.imshow(title, frame)

def main():
    global FORCE_STOP

    q = Queue()
    

    thread1 = threading.Thread(target=thread_cam1, args=(q,))
    thread2 = threading.Thread(target=thread_cam2, args=(q,))
    thread1.start()
    thread2.start()
    
    while not FORCE_STOP:
        name, frame = q.get(timeout=1)
        if name:
            imshow(name, frame)
            q.task_done()

        if cv2.waitKey(10) & 0xff == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()