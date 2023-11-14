#!/usr/bin/env python3

import os
import threading
from argparse import ArgumentParser
from queue import Empty, Queue
from time import sleep

import cv2
import numpy as np
from openvino.inference_engine import IECore
import openvino as ov
import logging as log
from iotdemo import FactoryController
from iotdemo import MotionDetector
from iotdemo import ColorDetector

FORCE_STOP = False


def thread_cam1(q):
    motion1 = MotionDetector()
    flag = False

    # TODO: MotionDetector
    motion1.load_preset('/home/ubuntu/intel-02/class02/smart-factory/resources/motion.cfg', 'default')
    # TODO: HW1 Open "resources/conveyor.mp4" video clip
    path = '/home/ubuntu/intel-02/class02/smart-factory/resources/factory/conveyor.mp4'
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print("can't open videofile\r\n")
    else:
        print("can open video\r\n")
    # TODO: Load and initialize OpenVINO
    core = ov.Core()
    model_path = "/home/ubuntu/intel-02/class02/smart-factory/resources/openvino.xml"
    model = core.read_model(model_path)
    if len(model.inputs) != 1:
        core.error('Sample supports only single input topologies')
        return -1

    if len(model.outputs) != 1:
        core.error('Sample supports only single output topologies')
        return -1

    while not FORCE_STOP:
        sleep(0.03)
        _, frame = cap.read()
        if frame is None:
            break

        # TODO: HW2 Enqueue "VIDEO:Cam1 live", frame info
        
        q.put(('live Cam 1', frame))
        # TODO: Motion detect
        dector = motion1.detect(frame)
        if dector is None:
            #print(1)
            continue
        q.put(('dectect', dector))
            
        # TODO: Enqueue "VIDEO:Cam1 detected", detected info.
        # abnormal detect
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #reshaped = detected[:, :, [2, 1, 0]]
        #np_data = np.moveaxis(reshaped, -1, 0)
        #preprocessed_numpy = [((np_data / 255.0) - 0.5) * 2]
        #batch_tensor = np.stack(preprocessed_numpy, axis=0)
        
        
        # TODO: Inference OpenVINO
        # image = cv2.imread(dector)
        input_tensor = np.expand_dims(dector, 0)
        
        ppp = ov.preprocess.PrePostProcessor(model)
        
        if(flag == False):
            _ , h, w, _ = input_tensor.shape

            ppp.input().tensor() \
                .set_shape(input_tensor.shape) \
                .set_element_type(ov.Type.u8) \
                .set_layout(ov.Layout('NHWC'))  # noqa: ECE001, N400
            ppp.input().preprocess().resize(ov.preprocess.ResizeAlgorithm.RESIZE_LINEAR)
            ppp.input().model().set_layout(ov.Layout('NCHW'))
            ppp.output().tensor().set_element_type(ov.Type.f32)
            model = ppp.build()
    
            compiled_model = core.compile_model(model, "CPU")
            flag = True
        results = compiled_model.infer_new_request({0: input_tensor})
        predictions = next(iter(results.values()))
        print(predictions[0][0])
    
        # TODO: Calculate ratios
        #print(f"X = {x_ratio:.2f}%, Circle = {circle_ratio:.2f}%")

        # TODO: in queue for moving the actuator 1
        if(predictions[0][0] > 0):
            q.put(('push', 1))

    cap.release()
    q.put(('DONE', None))
    exit()


def thread_cam2(q):
    color2 = ColorDetector()
    motion2 = MotionDetector()

    # TODO: MotionDetector
    motion2.load_preset('/home/ubuntu/intel-02/class02/smart-factory/resources/motion.cfg', 'default')
    # TODO: ColorDetector
    color2.load_preset('/home/ubuntu/intel-02/class02/smart-factory/resources/color.cfg', 'default')

    # TODO: HW2 Open "resources/conveyor.mp4" video clip
    path = '/home/ubuntu/intel-02/class02/smart-factory/resources/factory/conveyor.mp4'
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print("can't open videofile\r\n")
    else:
        print("can open video\r\n")



    while not FORCE_STOP:
        sleep(0.03)
        _, frame = cap.read()
        if frame is None:
            break

        # TODO: HW2 Enqueue "VIDEO:Cam2 live", frame info
        q.put(('live Cam 2', frame))
        
        # TODO: Detect motion
        det2 = motion2.detect(frame)
        if det2 is None:
            continue
        q.put(('dectect2', det2))
        # TODO: Enqueue "VIDEO:Cam2 detected", detected info.

        # TODO: Detect color
        color_det = color2.detect(det2)
        print(color_det)

        # TODO: Compute ratio
        #print(f"{name}: {ratio:.2f}%")

        # TODO: Enqueue to handle actuator 2
        if(color_det[0][1] > 0.5 ):
            q.put(('push', 2))
            
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
    t2 = threading.Thread(target=thread_cam2, args = (q,))
    
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
            if name == 'push':
                ctrl.push_actuator(int(frame))
    
    t1.join()
    t2.join()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        main()
    except Exception:
        os._exit()