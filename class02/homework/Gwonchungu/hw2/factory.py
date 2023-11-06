#!/usr/bin/env python3

import os
import threading
from argparse import ArgumentParser
from queue import Empty, Queue
from time import sleep

import cv2
import numpy as np
from openvino.inference_engine import IECore

from iotdemo import FactoryController

FORCE_STOP = False


def thread_cam1(q):
    cap = cv2.VideoCapture('conveyor.mp4')

    while not FORCE_STOP:
        ret, frame = cap.read()
        if not ret:
            break

        q.put(('Cam1 live', frame))

    cap.release()
    q.put(('DONE', None))


def thread_cam2(q):
    cap = cv2.VideoCapture('conveyor.mp4')

    while not FORCE_STOP:
        ret, frame = cap.read()
        if not ret:
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

    parser = ArgumentParser(prog='python3 factory.py', description="Factory tool")
    parser.add_argument("-d", "--device", default=None, type=str, help="Arduino port")
    args = parser.parse_args()

    q = Queue()

    thread1 = threading.Thread(target=thread_cam1, args=(q,))
    thread2 = threading.Thread(target=thread_cam2, args=(q,))
    thread1.start()
    thread2.start()

    with FactoryController(args.device) as ctrl:
        while not FORCE_STOP:
            if cv2.waitKey(10) & 0xff == ord('q'):
                break

            try:
                name, frame = q.get(timeout=1)
                if name:
                    imshow(name, frame)
                    q.task_done()

                if name == 'DONE':
                    FORCE_STOP = True
            except Empty:
                pass

    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        main()
    except Exception:
        os._exit()

