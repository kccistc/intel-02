'''
module run camera and send signal to arduino
'''
# import os
import logging
import threading
import sys
from argparse import ArgumentParser
from queue import Empty, Queue
from time import sleep

import cv2
import numpy as np
# from openvino.inference_engine import IECore

import openvino as ov
from iotdemo import ColorDetector, FactoryController, MotionDetector

FORCE_STOP = False
stack = []


# func of camera 1
def thread_cam1(q) -> None:
    '''func of camera 1'''
    flag = True
    # MotionDetector
    det = MotionDetector()
    det.load_preset("resources/motion.cfg", "default")

    # Load and initialize OpenVINO
    core = ov.Core()
    # model = core.read_model("resources/DeiT-Tiny.xml")
    model = core.read_model("resources/Mobile.xml")

    # Open video clip resources/conveyor.mp4 instead of camera device.
    cap = cv2.VideoCapture("resources/conveyor.mp4")

    while not FORCE_STOP:
        sleep(0.03)
        _, frame = cap.read()
        # _ is meta data unneccesary
        if frame is None:
            break

        # Enqueue "VIDEO:Cam1 live", frame info
        q.put(("VIDEO:CAM1 live", frame))

        # Motion detect
        detected = det.detect(frame)
        if detected is None:
            continue

        # Enqueue "VIDEO:Cam1 detected", detected info.
        q.put(("VIDEO:CAM1 detected", detected))

        input_tensor = np.expand_dims(detected, 0)

        if flag is True:
            ppp = ov.preprocess.PrePostProcessor(model)
            ppp.input().tensor() \
                .set_shape(input_tensor.shape) \
                .set_element_type(ov.Type.u8) \
                .set_layout(ov.Layout('NHWC'))  # noqa: ECE001, N400
            ppp.input().preprocess().resize(ov.preprocess.ResizeAlgorithm.RESIZE_LINEAR)
            ppp.input().model().set_layout(ov.Layout('NCHW'))
            ppp.output().tensor().set_element_type(ov.Type.f32)
            model = ppp.build()
            compiled_model = core.compile_model(model, "CPU")

            flag = False

        # Inference OpenVINO
        results = compiled_model.infer_new_request({0: input_tensor})
        predictions = next(iter(results.values()))
        probs = predictions.reshape(-1)
        print(f"{probs}")

        # in queue for moving the actuator 1
        if probs[0] > 0.0:
            print("Not Good Item.")
            q.put(("PUSH", 1))
        else:
            print("Good Item.")

    cap.release()
    q.put(('DONE', None))
    sys.exit()


# func of camera 2
def thread_cam2(q) -> None:
    '''func of cameara 2'''
    # MotionDetector
    det = MotionDetector()
    det.load_preset("resources/motion.cfg", "default")

    # ColorDetector
    color = ColorDetector()
    color.load_preset("resources/color.cfg", "default")

    # Open "resources/conveyor.mp4" video clip
    cap = cv2.VideoCapture("resources/conveyor.mp4")

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
        q.put(("VIDEO:Cam2 detected", detected))

        # Detect color
        predict = color.detect(detected)
        if not predict:
            continue

        # Compute ratio
        name, ratio = predict[0]
        ratio = ratio * 100
        print(f"{name}: {ratio:.2f}%")

        # Enqueue to handle actuator 2
        if name == "blue":
            q.put(("PUSH", 2))

    cap.release()
    q.put(('DONE', None))
    sys.exit()


def imshow(title, frame, pos=None) -> None:
    '''function to pop up video'''
    cv2.namedWindow(title)
    if pos:
        cv2.moveWindow(title, pos[0], pos[1])
    cv2.imshow(title, frame)


def main():
    '''main func which starts threads and sending signal to arduino'''
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
    queue = Queue()

    # Create thread_cam1 and thread_cam2 threads and start them.
    thread1 = threading.Thread(target=thread_cam1, args=(queue, ))
    thread2 = threading.Thread(target=thread_cam2, args=(queue, ))

    thread1.start()
    thread2.start()

    with FactoryController(args.device) as ctrl:
        while not FORCE_STOP:
            if cv2.waitKey(10) & 0xff == ord('q'):
                break

            # get an item from the queue. You might need to properly handle exceptions.
            # de-queue name and data
            try:
                event = queue.get_nowait()
            except Empty:
                continue

            # show videos with titles of 'Cam1 live' and 'Cam2 live' respectively.
            # Control actuator, name == 'PUSH'
            name, data = event
            if name.startswith('VIDEO:'):
                imshow(name[6:], data)
            elif name == "PUSH":
                ctrl.push_actuator(data)
            elif name == "DONE":
                FORCE_STOP = True

            queue.task_done()
        # end while not FORCE_STOP
    # end FactoryController

    thread1.join()
    thread2.join()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        main()
    except (Exception,) as e:
        logging.exception(e)
