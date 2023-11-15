#!/usr/bin/env python3


import os
import threading
from argparse import ArgumentParser
from queue import Empty, Queue
from time import sleep

import cv2
import numpy as np
from openvino.inference_engine import IECore

from iotdemo import MotionDetector
from iotdemo import ColorDetector
from iotdemo import FactoryController

import openvino as ov
import logging as log

FORCE_STOP = False
stack = []

def thread_cam1(q):
    # TODO: MotionDetector
    det = MotionDetector()
    det.load_preset('resources/motion.cfg', 'default')
    # TODO: Load and initialize OpenVINO
    flag = False
    model_path = 'resources/openvino.xml'
    device = 'CPU'
    
    core = ov.Core()
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
        q.put(('Cam1 live', frame))

        # TODO: Motion detect
        detected = det.detect(frame)
        
        if detected is None:
            continue 
        # TODO: Enqueue "VIDEO:Cam1 detected", detected info.
        q.put(('VIDEO: Cam1 detected', detected))
        
        input_tensor = np.expand_dims(detected, 0)
        
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
            compiled_model = core.compile_model(model, device)
        
        results = compiled_model.infer_new_request({0: input_tensor})
        predictions = next(iter(results.values())) 
        # print(predictions)

        # TODO: in queue for moving the actuator 1
        if predictions[0][0] > 0.5:
            print('X')
            stack.append(1)
        
        else:
            print('O')


    cap.release()
    q.put(('DONE', None))
    exit()


def thread_cam2(q):
    # TODO: MotionDetector
    det = MotionDetector()
    det.load_preset('resources/motion.cfg', 'default')
    # TODO: ColorDetector
    c_det = ColorDetector()
    c_det.load_preset('resources/color.cfg', 'default')

    # TODO: HW2 Open "resources/conveyor.mp4" video clip
    cap = cv2.VideoCapture('resources/conveyor.mp4')

    while not FORCE_STOP:
        sleep(0.03)
        _, frame = cap.read()
        if frame is None:
            break

        # TODO: HW2 Enqueue "VIDEO:Cam2 live", frame info
        q.put(('Cam2 live', frame))

        # TODO: Detect motion
        detected = det.detect(frame)
        if detected is None:
            continue
        
        # TODO: Detect color
        predict = c_det.detect(detected)
        print(predict)
        # TODO: Enqueue "VIDEO:Cam2 detected", detected info.
        q.put(('VIDEO: Cam2 detected', detected))
        
        # TODO: Enqueue to handle actuator 2
        if predict[1][0] == 'blue':
            stack.append(2)

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
    q = Queue()

    # TODO: HW2 Create thread_cam1 and thread_cam2 threads and start them.
    t1 = threading.Thread(target=thread_cam1, args=(q,))
    t2 = threading.Thread(target=thread_cam2, args=(q,))
    t1.start()
    t2.start()


    with FactoryController(args.device) as ctrl:
        while not FORCE_STOP:
            if cv2.waitKey(10) & 0xff == ord('q'):
                break

            try:
            # TODO: HW2 get an item from the queue. You might need to properly handle exceptions.
            # de-queue name and data
                name, frame = q.get(timeout=1)
                if name:
                    # TODO: HW2 show videos with titles of 'Cam1 live' and 'Cam2 live' respectively.
                    imshow(name, frame)
                    q.task_done()

                if name == 'DONE':
                    FORCE_STOP = True
            except Empty:
                pass

            # TODO: Control actuator, name == 'PUSH's
            if len(stack) != 1:
                ctrl.push_actuator(stack.pop())

    t1.join()
    t2.join()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        main()
    except Exception:
        os._exit()