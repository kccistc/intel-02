#!/usr/bin/env python3

import os
import threading
from argparse import ArgumentParser
from queue import Empty, Queue
from time import sleep

import cv2
import logging as log
import sys
import numpy as np
import openvino as ov
from openvino.inference_engine import IECore

from iotdemo import FactoryController
from iotdemo import MotionDetector
from iotdemo import ColorDetector

FORCE_STOP = False

def thread_cam1(q):
    # TODO: MotionDetector
    flag = False

    det = MotionDetector()
    det.load_preset("resources/motion.cfg", 'default')
    
    # TODO: Load and initialize OpenVINO

    core = ov.Core()
    model_path = 'resources/openvino.xml'
    device_name = "CPU"
    
    model = core.read_model(model_path)
    ppp = ov.preprocess.PrePostProcessor(model)
    
    # TODO: HW2 Open video clip resources/conveyor.mp4 instead of camera device.
    cap = cv2.VideoCapture('resources/conveyor.mp4')

    while not FORCE_STOP:
        sleep(0.03)
        _, frame = cap.read()
        if frame is None:
            break

        # TODO: HW2 Enqueue "VIDEO:Cam1 live", frame info
        q.put(("Cam1 live",frame))

        # TODO: Motion detect
        
        detected_frame = det.detect(frame)
        if detected_frame is None:
            continue
        
        # TODO: Enqueue "VIDEO:Cam1 detected", detected info.

        q.put(("Cam1 detected",detected_frame))
        
        # abnormal detect
        
        input_tensor = np.expand_dims(detected_frame, 0)
        if flag == False:
            flag = True
            _, h, w, _ = input_tensor.shape
            
            ppp.input().tensor() \
                .set_shape(input_tensor.shape) \
                .set_element_type(ov.Type.u8) \
                .set_layout(ov.Layout('NHWC'))
            
            ppp.input().preprocess().resize(ov.preprocess.ResizeAlgorithm.RESIZE_LINEAR)
            ppp.input().model().set_layout(ov.Layout('NCHW'))
            ppp.output().tensor().set_element_type(ov.Type.f32)
            model = ppp.build()
            
            compiled_model = core.compile_model(model, device_name)
        
        # TODO: Inference OpenVINO

        results = compiled_model.infer_new_request({0: input_tensor})
        predictions = next(iter(results.values()))
        
        probs = predictions.reshape(-1)
        x_ratio = probs[0] * 100
        circle_ratio = probs[1] * 100
        
        # TODO: Calculate ratios
        print(f"X = {x_ratio:.2f}%, Circle = {circle_ratio:.2f}%")

        # TODO: in queue for moving the actuator 1
        
        if x_ratio > 90:
            q.put(("PUSH", 1))
        
        

    cap.release()
    q.put(('DONE', None))
    exit()


def thread_cam2(q):
    # TODO: MotionDetector

    det = MotionDetector()
    det.load_preset("resources/motion.cfg", 'default')

    # TODO: ColorDetector
    
    color = ColorDetector()
    color.load_preset("resources/color.cfg", 'default')
    
    # TODO: HW2 Open "resources/conveyor.mp4" video clip
    cap = cv2.VideoCapture('resources/conveyor.mp4')

    while not FORCE_STOP:
        sleep(0.03)
        _, frame = cap.read()
        if frame is None:
            break

        # TODO: HW2 Enqueue "VIDEO:Cam2 live", frame info
        q.put(("Cam2 live",frame))

        # TODO: Detect motion

        detected_frame = det.detect(frame)
        if detected_frame is None:
            continue
    
        # TODO: Enqueue "VIDEO:Cam2 detected", detected info.

        q.put(("Cam2 detected",detected_frame))

        # TODO: Detect color

        predict = color.detect(detected_frame)
        name, sum = predict[0]
        _, ratio2 = predict[1]

        ratio = sum/(sum + ratio2) * 100
        
        # TODO: Compute ratio
        print(f"{name}: {ratio:.2f}%")

        # TODO: Enqueue to handle actuator 2
        
        if name == 'blue':
            q.put(("PUSH", 2))
        

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

    # TODO: HW2 Create a Queue
    cam_queue = Queue()
    
    # TODO: HW2 Create thread_cam1 and thread_cam2 threads and start them.
    t1 = threading.Thread(target=thread_cam1, args=(cam_queue,))
    t2 = threading.Thread(target=thread_cam2, args=(cam_queue,))
    t1.start()
    t2.start()
    
    
    with FactoryController(args.device) as ctrl:
        while not FORCE_STOP:
            if cv2.waitKey(10) & 0xff == ord('q'):
                break

            # TODO: HW2 get an item from the queue. You might need to properly handle exceptions.
            # de-queue name and data
            name,cam1_frame = cam_queue.get()
            cv2.imshow(name, cam1_frame)
            
            # TODO: HW2 show videos with titles of 'Cam1 live' and 'Cam2 live' respectively.

            # TODO: Control actuator, name == 'PUSH'
            if name == 'PUSH':
                ctrl.push_actuator(cam1_frame)


            if name == 'DONE':
                FORCE_STOP = True

            cam_queue.task_done()

    cv2.destroyAllWindows()

    t1.join()
    t2.join()

if __name__ == '__main__':
    try:
        main()
    except Exception:
        os._exit()
