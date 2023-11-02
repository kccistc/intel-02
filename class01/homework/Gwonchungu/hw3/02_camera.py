import numpy as np
import cv2

# Read from the first camera device
cap = cv2.VideoCapture(0)

topLeft = (50, 50)
bottomRight = (300, 300)

fx=0
fy=0

# 마우스 클릭 이벤트 처리를 위한 함수
def draw_circle(event, x, y, flags, param):
    global frame  # frame 변수를 전역 변수로 선언
    if event == cv2.EVENT_LBUTTONDOWN:
        global fx
        global fy
        fx=x
        fy=y

# 성공적으로 video device 가 열렸으면 while 문 반복
while(cap.isOpened()):
    # 한 프레임을 읽어옴
    ret, frame = cap.read()
    if ret is False:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Line
    cv2.line(frame, topLeft, bottomRight, (0, 255, 0), 5)

    # Rectangle
    cv2.rectangle(frame,
        [pt+40 for pt in topLeft], [pt-40 for pt in bottomRight], (0, 0, 255),5)

    # Text 
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, 'I am Good Boy', [pt+30 for pt in topLeft], font, 2, (0, 255, 255), 5)

    # Display
    cv2.imshow("Camera",frame)

    # 마우스 이벤트 콜백 함수 등록
    cv2.setMouseCallback("Camera", draw_circle)
    cv2.circle(frame, (fx, fy), 30, (255, 0, 255), 5)
    cv2.imshow("Camera",frame)

    # 1 ms 동안 대기하며 키 입력을 받고 'q' 입력 시 종료
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
