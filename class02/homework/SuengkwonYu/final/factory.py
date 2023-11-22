#!/usr/bin/env python3

import os
import threading
from argparse import ArgumentParser
from queue import Empty, Queue
from time import sleep

import cv2
import numpy as np
import openvino as ov

from iotdemo import FactoryController, MotionDetector, ColorDetector

FORCE_STOP = False

def thread_cam1(q):
    # MotionDetector
    det = MotionDetector()
    det.load_preset('motion.cfg', 'default')

    # Load and initialize OpenVINO
    core = ov.Core()

    # Load the model
    model_xml = "/home/suengkwon/workspace/classification-DeiT-Tiny/outputs/20231107_214309_export/openvino/openvino.xml"
    model = core.read_model(model_xml)

    # Open video clip resources/conveyor.mp4 instead of camera device.
    # pylint: disable=E1101
    cap = cv2.VideoCapture("resources/conveyor.mp4")

    # Preprocessing
    ppp = ov.preprocess.PrePostProcessor(model)
    ppp.input().tensor() \
        .set_shape((1, 224, 224, 3)) \
	.set_element_type(ov.Type.u8) \
	.set_layout(ov.Layout('NHWC'))  # noqa: ECE001, N400
    ppp.input().preprocess().resize(ov.preprocess.ResizeAlgorithm.RESIZE_LINEAR)
    ppp.input().model().set_layout(ov.Layout('NCHW'))
    ppp.output().tensor().set_element_type(ov.Type.f32)
    model = ppp.build()
    compiled_model = core.compile_model(model = model, device_name = 'CPU')

    while not FORCE_STOP:
        sleep(0.03)
        _, frame = cap.read()
        if frame is None:
            break

        # Enqueue "VIDEO:Cam1 live", frame info
        q.put(("VIDEO:Cam1 live", frame))

        # Motion detect
        detected = det.detect(frame)
        if detected is None:
            continue

        # Enqueue "VIDEO:Cam1 detected", detected info.
        q.put(('VIDEO:Cam1 detected', detected))

        # abnormal detect
        input_tensor = np.expand_dims(detected, 0)

        # Inference OpenVINO
        results = compiled_model.infer_new_request({0: input_tensor})

        # Calculate ratios
        #predictions = next(iter(results.values()))
        probs = next(iter(results.values())).reshape(-1)
        x_ratio = probs[0] * 100
        circle_ratio = probs[1] * 100
        print(f"X = {x_ratio:.2f}%, Circle = {circle_ratio:.2f}%")

        # in queue for moving the actuator 1
        if probs[1] > 0.7:
            q.put(('PUSH', 1))

    cap.release()
    q.put(('DONE', None))

def thread_cam2(q):
    # MotionDetector
    det = MotionDetector()
    det.load_preset('motion.cfg', 'default')

    # ColorDetector
    color = ColorDetector()
    color.load_preset('color.cfg', 'default')

    # Open "resources/conveyor.mp4" video clip
    # pylint: disable=E1101
    cap = cv2.VideoCapture("resources/conveyor.mp4")
    if cap is None:
        return

    while not FORCE_STOP:
        sleep(0.03)
        _, frame = cap.read()
        if frame is None:
            break

        # Enqueue "VIDEO:Cam2 live", frame info
        q.put(("VIDEO:Cam2 live", frame))

        # Detect motion
        detected = det.detect(frame)
        if detected is None:
            continue

        # Enqueue "VIDEO:Cam2 detected", detected info.
        q.put(('VIDEO:Cam2 detected', detected))

        # Detect color
        predict = color.detect(detected)
        if not predict:
            continue

        # Compute ratio
        name, ratio = predict[0]
        ratio = ratio * 100
        print(f"{name}: {ratio:.2f}%")

        # Enqueue to handle actuator 2
        if name == 'blue' and ratio > 30:
            q.put(('PUSH', 2))

    cap.release()
    q.put(('DONE', None))


def imshow(title, frame, pos=None):
    # pylint: disable=E1101
    cv2.namedWindow(title)
    if pos:
        # pylint: disable=E1101
        cv2.moveWindow(title, pos[0], pos[1])
    # pylint: disable=E1101
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

    # Create a Queue
    q = Queue()

    # Create thread_cam1 and thread_cam2 threads and start them.
    thread1 = threading.Thread(target = thread_cam1, args = (q, ))
    thread2 = threading.Thread(target = thread_cam2, args = (q, ))

    thread1.start()
    thread2.start()

    with FactoryController(args.device) as ctrl:
        while not FORCE_STOP:
            # pylint: disable=E1101
            if cv2.waitKey(10) & 0xff == ord('q'):
                break

            # get an item from the queue. You might need to properly handle exceptions.
            # de-queue name and data
            try:
                event = q.get_nowait()
            except Empty:
                continue

            name, data = event
            # show videos with titles of 'Cam1 live' and 'Cam2 live' respectively.
            if name.startswith('VIDEO:'):
                imshow(name[6:], data)
            # Control actuator, name == 'PUSH'
            elif name == 'PUSH':
                ctrl.push_actuator(data)
            elif name == 'DONE':
                FORCE_STOP = True

    thread1.join()
    thread2.join()
    # pylint: disable=E1101
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        main()
    except Exception:
        os._exit()
