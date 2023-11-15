'''!/usr/bin/env python3'''

import os
import threading
from argparse import ArgumentParser
from queue import Queue
from time import sleep

import sys
import cv2
import numpy as np
import openvino as ov

from iotdemo import FactoryController, MotionDetector, ColorDetector

FORCE_STOP = False

def thread_cam1(q):
    '''detect motion'''
    flag = False
    # MotionDetector
    det = MotionDetector()
    det.load_preset("resources/motion.cfg", 'default')
    # Load and initialize OpenVINO
    core = ov.Core()
    model_path = 'resources/openvino.xml'
    model = core.read_model(model_path)
    ppp = ov.preprocess.PrePostProcessor(model)
    # HW2 Open video clip resources/conveyor.mp4 instead of camera device.
    cap = cv2.VideoCapture('resources/conveyor.mp4')

    while not FORCE_STOP:
        sleep(0.03)
        _, frame = cap.read()
        if frame is None:
            break
        # HW2 Enqueue "VIDEO:Cam1 live", frame info
        q.put(("VIDEO:Cam1 live",frame))
        # Motion detect
        detected_frame = det.detect(frame)
        if detected_frame is None:
            continue
        # Enqueue "VIDEO:Cam1 detected", detected info.
        q.put(("VIDEO:Cam1 detected",detected_frame))
        # abnormal detect
        input_tensor = np.expand_dims(detected_frame, 0)
        if flag is False:
            flag = True
            _, _, _, _ = input_tensor.shape
            ppp.input().tensor() \
                .set_shape(input_tensor.shape) \
                .set_element_type(ov.Type.u8) \
                .set_layout(ov.Layout('NHWC'))
            ppp.input().preprocess().resize(ov.preprocess.ResizeAlgorithm.RESIZE_LINEAR)
            ppp.input().model().set_layout(ov.Layout('NCHW'))
            ppp.output().tensor().set_element_type(ov.Type.f32)
            model = ppp.build()
            compiled_model = core.compile_model(model, "CPU")
        # Inference OpenVINO
        results = compiled_model.infer_new_request({0: input_tensor})
        predictions = next(iter(results.values()))
        probs = predictions.reshape(-1)
        # Calculate ratios
        x_ratio = probs[0] * 100
        circle_ratio = probs[1] * 100
        print(f"X = {x_ratio:.2f}%, O = {circle_ratio:.2f}%")
        # in queue for moving the actuator 1
        if x_ratio > 90:
            q.put(("PUSH", 1))
    cap.release()
    q.put(('DONE', None))
    sys.exit()

def thread_cam2(q):
    '''detect color'''
    # MotionDetector
    det = MotionDetector()
    det.load_preset("resources/motion.cfg", 'default')
    # ColorDetector
    color = ColorDetector()
    color.load_preset("resources/color.cfg", 'default')
    # HW2 Open "resources/conveyor.mp4" video clip
    cap = cv2.VideoCapture('resources/conveyor.mp4')
    while not FORCE_STOP:
        sleep(0.03)
        _, frame = cap.read()
        if frame is None:
            break
        # HW2 Enqueue "VIDEO:Cam2 live", frame info
        q.put(("VIDEO:Cam2 live",frame))
        # Detect motion
        detected_frame = det.detect(frame)
        if detected_frame is None:
            continue
        # Enqueue "VIDEO:Cam2 detected", detected info.
        q.put(("VIDEO:Cam2 detected",detected_frame))
        # Detect color
        predict = color.detect(detected_frame)
        # Compute ratio
        name, ratio1 = predict[0]
        _, ratio2 = predict[1]
        ratio = ratio1/(ratio1 + ratio2) * 100
        print(f"{name}: {ratio:.2f}%")
        # Enqueue to handle actuator 2
        if name == 'blue':
            q.put(("PUSH", 2))
    cap.release()
    q.put(('DONE', None))
    sys.exit()

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

    # HW2 Create a Queue
    cam_queue = Queue()
    # HW2 Create thread_cam1 and thread_cam2 threads and start them.
    t1 = threading.Thread(target=thread_cam1, args=(cam_queue,))
    t2 = threading.Thread(target=thread_cam2, args=(cam_queue,))
    t1.start()
    t2.start()

    with FactoryController(args.device) as ctrl:
        while not FORCE_STOP:
            if cv2.waitKey(10) & 0xff == ord('q'):
                break
            # HW2 get an item from the queue. You might need to properly handle exceptions.
            # de-queue name and data
            try:
                event = cam_queue.get_nowait()
            except Exception:
                continue
            name,frame = event
            # HW2 show videos with titles of 'Cam1 live' and 'Cam2 live' respectively.
            if name.startswith("VIDEO:"):
                cv2.imshow(name[5:], frame)

            # Control actuator, name == 'PUSH'
            if name == 'PUSH':
                ctrl.push_actuator(frame)

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
