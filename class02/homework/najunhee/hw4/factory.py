'''
start
'''
import os
import sys
import threading
from queue import Empty, Queue
from argparse import ArgumentParser
from time import sleep
import numpy as np
import openvino as ov
import cv2
from iotdemo import FactoryController
from iotdemo import MotionDetector
from iotdemo import ColorDetector

FORCE_STOP = False

def thread_cam1(q_):
    det = MotionDetector()
    det.load_preset("resources/motion.cfg","default")
    model_path = "openvino.xml"

    core = ov.Core()

    model = core.read_model(model_path)

    ppp = ov.preprocess.PrePostProcessor(model)
    cap = cv2.VideoCapture("resources/conveyor.mp4")

    flag = True

    while not FORCE_STOP:
        sleep(0.03)
        _, frame = cap.read()
        if frame is None:
            break


        q_.put(("VIDEO:Cam1 live", frame))

        detected= det.detect(frame)
        if detected is None:
            continue
        input_tensor = np.expand_dims(detected, 0)

        q_.put(("VIDEO:Cam1 detected", detected))
        if flag:
            ppp.input().tensor() \
                .set_shape(input_tensor.shape) \
                .set_element_type(ov.Type.u8) \
                .set_layout(ov.Layout('NHWC'))
            ppp.input().preprocess().resize(ov.preprocess.ResizeAlgorithm.RESIZE_LINEAR)
            ppp.input().model().set_layout(ov.Layout('NCHW'))
            ppp.output().tensor().set_element_type(ov.Type.f32)
            model = ppp.build()
            flag = False
        compiled_model = core.compile_model(model, "CPU")

        results = compiled_model.infer_new_request({0: input_tensor})
        predictions = next(iter(results.values()))
        probs = predictions.reshape(-1)
        print(f"Probabillity = {probs}")
        if probs[0]>0:
            print("Defected item")
            q_.put(("PUSH",1))
        else:
            print("Nice item")

    cap.release()
    q_.put(('DONE', None))
    sys.exit()


def thread_cam2(q_):
    det = MotionDetector()
    det.load_preset("resources/motion.cfg", "default")
    color = ColorDetector()
    color.load_preset("resources/color_info.cfg","default")
    cap = cv2.VideoCapture("resources/conveyor.mp4")
    while not FORCE_STOP:
        sleep(0.03)
        _, frame = cap.read()
        if frame is None:
            break
        q_.put(("VIDEO:Cam2 live",frame))
        detected= det.detect(frame)
        if detected is None:
            continue
        q_.put(("VIDEO:Cam2 detected",detected))
        predict = color.detect(detected)
        name, sun = predict[0]
        _, ratio2 = predict[1]
        ratio = sun/(sun+ratio2)*100
        print(f"{name}: {ratio:.2f}%")
        if name=="blue":
            q_.put(("PUSH",2))

    cap.release()
    q_.put(('DONE', None))
    sys.exit()


def imshow(title, frame, pos=None):
    cv2.namedWindow(title)
    if pos:
        cv2.moveWindow(title, pos[0], pos[1])
    cv2.imshow(title, frame)


def main():
    '''
    main start
    '''
    global FORCE_STOP
    parser = ArgumentParser(prog='python3 factory.py',
                            description="Factory tool")

    parser.add_argument("-d",
                        "--device",
                        default=None,
                        type=str,
                        help="Arduino port")
    args = parser.parse_args()
    shared_queue = Queue()
    t_1 = threading.Thread(target = thread_cam1, args = (shared_queue,))
    t_2 = threading.Thread(target = thread_cam2, args = (shared_queue,))
    t_1.start()
    t_2.start()

    with FactoryController(args.device) as ctrl:
        while not FORCE_STOP:
            if cv2.waitKey(10) & 0xff == ord('q'):
                break
            try:
                event = shared_queue.get_nowait()
            except Empty:
                continue
            name, data = event
            if name.startswith("VIDEO:"):
                cv2.imshow(name[6:], data)
            elif name == 'DONE':
                FORCE_STOP = True
            elif name == "PUSH":
                ctrl.push_actuator(data)
            shared_queue.task_done()

    cv2.destroyAllWindows()
    t_1.join()
    t_2.join()

if __name__ == '__main__':
    try:
        main()
    except FileNotFoundError:
        os._exit()
