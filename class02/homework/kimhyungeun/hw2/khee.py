import os
import cv2
import threading
from queue import Queue, Empty
from iotdemo import FactoryController

# import 순서 재배치
FORCE_STOP = False

def thread_cam1(q):
    # HW2: 여기에 Cam1에 대한 처리 로직을 추가하세요
    # OpenCV를 사용하여 프레임 처리 코드를 추가하세요
    # 예를 들어, 카메라 캡처 및 프레임 처리 코드
    cap = cv2.VideoCapture(0)
    while not FORCE_STOP:
        ret, frame = cap.read()
        if not ret:
            break
        q.put(('Cam1 live', frame))
    cap.release()

def thread_cam2(q):
    # HW2: 여기에 Cam2에 대한 처리 로직을 추가하세요
    pass

def imshow(title, frame, pos=None):
    cv2.namedWindow(title)
    if pos:
        cv2.moveWindow(title, pos[0], pos[1])
    cv2.imshow(title, frame)

def main():
    global FORCE_STOP

    # HW2: Queue를 생성하세요
    q = Queue()

    # HW2: 두 개의 쓰레드 (thread_cam1, thread_cam2)를 생성하고 시작하세요
    t_cam1 = threading.Thread(target=thread_cam1, args=(q,))
    t_cam2 = threading.Thread(target=thread_cam2, args=(q,))
    t_cam1.start()
    t_cam2.start()

    with FactoryController() as ctrl:
        while not FORCE_STOP:
            if cv2.waitKey(10) & 0xff == ord('q'):
                break

            try:
                # HW2: Queue에서 아이템을 가져오고 적절히 예외를 처리하세요
                name, data = q.get(timeout=0.1)
            except Empty:
                continue

            # HW2: 'Cam1 live' 및 'Cam2 live' 제목의 비디오를 표시하세요
            imshow(name, data)

            # HW2: 액추에이터를 제어하세요
            if name == 'PUSH':
                # data에 따라 액추에이터 제어를 추가하세요
                pass

            if name == 'DONE':
                FORCE_STOP = True

            q.task_done()

    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        main()
    except Exception:
        os._exit()
