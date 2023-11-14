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

FORCE_STOP = False

path = 'resources/conveyor.mp4'


def thread_cam1(q):
    
    # TODO: MotionDetector
    det = MotionDetector()
    det.load_preset('motion.cfg', 'default')
    
    # TODO: Load and initialize OpenVINO
    core = ov.Core()
    model_path = 'openvino.xml'
    model = core.read_model(model_path)
    ppp = ov.preprocess.PrePostProcessor(model)

    # TODO: HW2 Open video clip resources/conveyor.mp4 instead of camera device.
    cap = cv2.VideoCapture(path)
    
    flag = True

    while not FORCE_STOP:
        sleep(0.03)
        ret, frame = cap.read()
        if frame is None:
            break
        
        # cap.read() 메서드는 두 가지 값을 반환합니다. ret은 프레임을 성공적으로 읽었는지를 나타내는 부울이고, frame은 프레임의 실제 이미지 데이터

        # TODO: HW2 Enqueue "VIDEO:Cam1 live", frame info
        q.put(('1', frame))


        # TODO: Motion detect
        detected1 = det.detect(frame)
        if detected1 is None:
            continue
        
        # TODO: Enqueue "VIDEO:Cam1 detected", detected info.
        q.put(('VIDEO: Cam1 Detected', detected1))
        
        input_tensor = np.expand_dims(detected1, 0)
        # np.expand_dims(detected1, 0): detected1의 차원을 확장, np.expand_dims 함수를 사용하여 이 이미지를 배치 차원을 추가한 형태로 만듭
        # 이렇게 하는 이유는 모델이 배치(batch) 단위로 데이터를 처리, 딥러닝 모델은 여러 이미지를 동시에 처리하기 위해 배치 차원을 가지게 됩니다.
        
        if flag is True:
            
            # 입력 텐서 모양 및 레이아웃 설정
            _, h, w, _ = input_tensor.shape
            # input_tensor.shape: 입력 텐서의 모양을 나타냅니다. 이는 (size, 높이, 너비, 채널)의 형태, _: 사용하지 않는 첫 번째 값입니다. 높이(h), 너비(w), 채널(_) 값을 언패킹합니다
            ppp.input().tensor() \
                .set_shape(input_tensor.shape) \
                .set_element_type(ov.Type.u8) \
                .set_layout(ov.Layout('NHWC'))
                # 원소 유형을 8비트 부호 없는 정수(u8)로 설정, 대부분의 이미지는 8비트 정수로 표현

                # 입력 데이터의 레이아웃을 'NHWC'로 설정, 이는 (높이, 너비, 채널) 순서를 나타냅니다.
            
            # 크기 조정 및 출력 텐서 레이아웃 설정
            ppp.input().preprocess().resize(ov.preprocess.ResizeAlgorithm.RESIZE_LINEAR)
            ppp.input().model().set_layout(ov.Layout('NCHW'))
            ppp.output().tensor().set_element_type(ov.Type.f32)
            
            # 전처리 모델 빌드 및 컴파일
            model = ppp.build()
            device_name = 'CPU'
            compiled_model = core.compile_model(model, device_name) #model compile
            flag = False
            
        # 모델 추론
        results = compiled_model.infer_new_request({0: input_tensor})
        # infer_new_request 메서드는 새로운 요청에 대해 추론을 수행
        predictions = next(iter(results.values()))
        # values() 메서드는 출력 blob의 목록을 가져오며, iter()는 반복자를 만들기 위해 사용, next()는 해당 반복자에서 첫 번째(일반적으로 유일한) 요소를 가져옵
        probs = predictions.reshape(-1)
        # 배열을 1차원 배열로 펼치는 작업,특히 분류 작업에서는 출력을 1차원 배열로 변형하는 것이 일반적
        x_ratio = probs[0]*100
        o_ratio = probs[1]*100

        print(f"X = {x_ratio:.2f}%, O = {o_ratio:.2f}%")
        
        # TODO : in queue for moving the actuator 1
        if x_ratio > 80:
            q.put(('PUSH', 1))
        
    cap.release()
    q.put(('DONE', None))
    exit()

def thread_cam2(q):
    # TODO: MotionDetector
    det = MotionDetector()
    det.load_preset('motion.cfg', 'default')
    
    # TODO: ColorDetector
    colorDet = ColorDetector()
    colorDet.load_preset('color.cfg', 'default')

    # TODO: HW2 Open "resources/conveyor.mp4" video clip
    cap = cv2.VideoCapture(path)

    while not FORCE_STOP:
        sleep(0.03)
        ret, frame = cap.read()
        if frame is None:
            break

        # TODO: HW2 Enqueue "VIDEO:Cam2 live", frame info
        q.put(('2', frame))
        

        # TODO: Detect motion
        detected2 = det.detect(frame)
        if detected2 is None:
            continue
        
        # TODO: Detect Color
        predict = colorDet.detect(detected2)
        #print(predict)
        name, ratio = predict[0]
        ratio = ratio * 100
        
        # TODO : Compute ratio
        print(f"{name}: {ratio:.2f}")
        
        # TODO: Enqueue "VIDEO:Cam2 detected", detected info.
        q.put(('VIDEO: Cam2 Detected', detected2))
        
        # TODO: Enqueue to handle actuator 2
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
                
                if name == 'PUSH':
                    # TODO: HW2 show videos with titles of 'Cam1 live' and 'Cam2 live' respectively.
                    ctrl.push_actuator(frame)
                elif name:
                    imshow(name, frame)
                q.task_done()

                if name == 'DONE':
                    FORCE_STOP = True
            except Empty:
                pass

            # TODO: Control actuator, name == 'PUSH's

    t1.join()
    t2.join()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        main()
    except Exception:
        os._exit() 
