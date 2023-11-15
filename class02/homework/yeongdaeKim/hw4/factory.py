#!/usr/bin/env python3

import os
import threading
from argparse import ArgumentParser
from queue import Empty, Queue
from time import sleep

import cv2
import numpy as np
import openvino as ov
from openvino.inference_engine import IECore

from iotdemo import FactoryController
from iotdemo.motion.motion_detector import MotionDetector
from iotdemo.color.color_detector import ColorDetector
from iotdemo.common.preset import load_preset, save_preset

FORCE_STOP = False


def thread_cam1(q):
    # TODO: MotionDetector 
    det = MotionDetector()
    det.load_preset("motion.cfg", "default")

    # TODO: Load and initialize OpenVINO
    model_path = "openvino.xml"
    device_name = "GPU"

    core = ov.Core()
    
    model = core.read_model(model_path)
   
    if len(model.inputs) != 1: return -1
    if len(model.outputs) != 1: return -1
    
    # TODO: HW2 Open video clip resources/conveyor.mp4 instead of camera device.
    cap = cv2.VideoCapture("resources/conveyor.mp4")
    if cap is None: return

    flag = False

    while not FORCE_STOP:
        sleep(0.03)
        _, frame = cap.read()
        if frame is None:
            break

        # TODO: HW2 Enqueue "VIDEO:Cam1 live", frame info
        q.put(("VIDEO:Cam1 live", frame))

        # TODO: Motion detect
        detected = det.detect(frame)
        if detected is None: continue

        input_tensor = np.expand_dims(detected, 0)
        
        # TODO: Enqueue "VIDEO:Cam1 detected", detected info.
        q.put(('VIDEO:Cam1 detected', detected))

        # abnormal detect
        #detected = cv2.cvtColor(detected, cv2.COLOR_BGR2RGB)
        #reshaped = detected[:, :, [2, 1, 0]]
        #np_data = np.moveaxis(reshaped, -1, 0)
        #preprocessed_numpy = [((np_data / 255.0) - 0.5) * 2]
        #batch_tensor = np.stack(preprocessed_numpy, axis=0)
        
        # TODO: Inference OpenVINO
        if not flag:
            ppp = ov.preprocess.PrePostProcessor(model)

            ppp.input().tensor() \
                .set_shape(input_tensor.shape) \
                .set_element_type(ov.Type.u8) \
                .set_layout(ov.Layout("NHWC"))

            ppp.input().preprocess().resize(ov.preprocess.ResizeAlgorithm.RESIZE_LINEAR)

            ppp.input().model().set_layout(ov.Layout("NCHW"))

            ppp.output().tensor().set_element_type(ov.Type.f32)

            builded_model = ppp.build()

            compiled_model = core.compile_model(builded_model, device_name)
            
            flag = True

        results = compiled_model.infer_new_request({0: input_tensor})

        predictions = next(iter(results.values()))

        probs = predictions.reshape(-1)

        #x_ratio, circle_ratio 
        norm_probs = np.argsort(probs)[-2:][::-1]

        x_ratio = norm_probs[1] * 100
        circle_ratio = norm_probs[0] * 100

        # TODO: Calculate ratios
        print(f"X = {x_ratio:.2f}%, Circle = {circle_ratio:.2f}%")

        # TODO: in queue for moving the actuator 1
        if x_ratio > circle_ratio: 
            q.put(('PUSH', 1))


    cap.release()
    q.put(('DONE', None))
    exit()


def thread_cam2(q):
    # TODO: MotionDetector
    det = MotionDetector()
    det.load_preset("motion.cfg", "default")

    # TODO: ColorDetector
    color = ColorDetector()
    color.load_preset("color.cfg", "default")

    # TODO: HW2 Open "resources/conveyor.mp4" video clip
    cap = cv2.VideoCapture("resources/conveyor.mp4")
    if cap is None: return
    
    while not FORCE_STOP:
        sleep(0.03)
        _, frame = cap.read()
        if frame is None:
            break
        
        # TODO: HW2 Enqueue "VIDEO:Cam2 live", frame info
        q.put(("VIDEO:Cam2 live", frame))

        # TODO: Detect motion
        detected = det.detect(frame)
        if detected is None: continue

        # TODO: Enqueue "VIDEO:Cam2 detected", detected info. 
        q.put(('VIDEO:Cam2 detected', detected))

        # TODO: Detect color
        predict = color.detect(detected)
                
        name, ratio1 = predict[0]
        _, ratio2 = predict[1]
        
        # TODO: Compute ratio --> Get normal ratio 
        norm_ratio = ratio1 / (ratio1 + ratio2) * 100

        # TODO: Enqueue to handle actuator 2
        if name == 'blue':
            q.put(('PUSH', 2))
        
        print(f"{name}: {norm_ratio:.2f}%") 

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
    thread1 = threading.Thread(target = thread_cam1, args = (q, ))
    
    thread2 = threading.Thread(target = thread_cam2, args = (q, ))
    
    thread1.start()

    thread2.start()

    #lock = threading.Lock()

    with FactoryController(args.device) as ctrl:
        ctrl.system_start()
        while not FORCE_STOP:
            if cv2.waitKey(10) & 0xff == ord('q'):
                break

            # TODO: HW2 get an item from the queue. You might need to properly handle exceptions.
            # de-queue name and data
            try:
                name, frame = q.get()

            # TODO: HW2 show videos with titles of 'Cam1 live' and 'Cam2 live' respectively.

            # TODO: Control actuator, name == 'PUSH'
                if name == 'PUSH':
                    ctrl.push_actuator(int(frame))
                elif name == 'DONE':
                    FORCE_STOP = True
                else:
                    if name == 'VIDEO:Cam1 live': 
                        imshow(name, frame, (50, 50))
                    elif name == 'VIDEO:Cam2 live': 
                        imshow(name, frame, (720, 50))
                    elif name == 'VIDEO:Cam1 detected': 
                        imshow(name, frame, (50, 630))
                    elif name == 'VIDEO:Cam2 detected':
                        imshow(name, frame, (720, 630))
            except 1: pass

            q.task_done()
    
    thread1.join()
    thread2.join()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        main()
    except Exception:
        os._exit()
