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
from iotdemo import MotionDetector
from iotdemo import ColorDetector

import openvino as ov
import logging as log
import sys

FORCE_STOP = False


def thread_cam1(q):
     
    det = MotionDetector()
    det.load_preset('resources/motion.cfg', 'default')


    # TODO: Load and initialize OpenVINO

    core = ov.Core()
    model_path = 'resources/openvino.xml'
    model = core.read_model(model_path)
    ppp = ov.preprocess.PrePostProcessor(model)
    
    cap = cv2.VideoCapture('resources/conveyor.mp4')
        
    flag = True

    while not FORCE_STOP:
        sleep(0.03)
        _, frame = cap.read()
        if frame is None:
            break
        detected = det.detect(frame)
        if detected is None:
            continue
        q.put(('Cam1 detected', detected))
        
        input_tensor = np.expand_dims(detected, 0)
        # TODO: Inference OpenVINO
        if flag is True:
            _, h, w, _ = input_tensor.shape
            ppp.input().tensor() \
                    .set_shape(input_tensor.shape) \
                    .set_element_type(ov.Type.u8) \
                    .set_layout(ov.Layout('NHWC'))
            ppp.input().preprocess().resize(ov.preprocess.ResizeAlgorithm.RESIZE_LINEAR)
            ppp.input().model().set_layout(ov.Layout('NCHW'))
            ppp.output().tensor().set_element_type(ov.Type.f32)
            model = ppp.build()
            device_name = 'CPU'
            compiled_model = core.compile_model(model, device_name)
            flag = False

        results = compiled_model.infer_new_request({0: input_tensor})
        predictions = next(iter(results.values()))
        probs = predictions.reshape(-1)
        x_ratio = probs[0]*100
        circle_ratio = probs[1]*100
        
        print(f"X = {x_ratio:.2f}%, Circle = {circle_ratio:.2f}%")
        
        if x_ratio > 80:
            q.put(('PUSH', 1)) 
    cap.release()
    q.put(('DONE', None))
    exit()


def thread_cam2(q):
    det = MotionDetector()

    det.load_preset('resources/motion.cfg', 'default')

    color = ColorDetector()
    color.load_preset('resources/color.cfg', 'default')

    cap = cv2.VideoCapture("resources/conveyor.mp4")

    while not FORCE_STOP:
        sleep(0.03)
        _, frame = cap.read()
        if frame is None:
            break

        detected = det.detect(frame)
        if detected is None:
            continue
        q.put(('Cam2 detected', detected))
        
        predict = color.detect(detected)
        name, ratio = predict[0]
        ratio = ratio*100
        print(f"{name}: {ratio:.2f}%")

        if name == 'blue':
            q.put(('PUSH', 2))
    cap.release()
    q.put(('DONE', None))
    exit()


def imshow(title, frame, pos=None):
    cv2.namedWindow(title)
    if pos:
        cv2.moveWindow(title, pos[0], pos[1])
    cv2.imshow(title, frame)


def main():
    global FORCE_STOP


    parser = ArgumentParser(prog='python3 factory.py',
                            description="Factory tool")

    parser.add_argument("-d",
                        "--device",
                        default=None,
                        type=str,
                        help="Arduino port")
    args = parser.parse_args()

    
    q = Queue() 


    t1 = threading.Thread(target=thread_cam1, args=(q,))
    t2 = threading.Thread(target=thread_cam2, args=(q,))
    t1.start()
    t2.start()
    

    with FactoryController(args.device) as ctrl:
        ctrl.system_start()
        while not FORCE_STOP:
            if cv2.waitKey(10) & 0xff == ord('q'):
                break
            try:
                name, frame = q.get(timeout=1)
                if name == 'PUSH':
                    ctrl.push_actuator(frame)
                elif name:
                    imshow(name, frame)
                q.task_done()
                if name == 'DONE':
                    FORCE_STOP = True
            except Empty:
                pass
    t1.join()
    t2.join()
    cv2.destroyAllWindows()
    ctrl.system_stop()
    ctrl.close()

if __name__ == '__main__':
    try:
        main()
    except Exception:
        os._exit(0)
