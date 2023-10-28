import numpy as np
import cv2
# Read from the first camera device
cap = cv2.VideoCapture(0)

def on_mouse(event, x, y, flags, param):
    global drawing, center

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing=True
        center = (x,y)

drawing = False
center = (-50,-50)

cv2.namedWindow("Circle")


# 성공적으로 video device 가 열렸으면 while 문 반복
while(cap.isOpened()):
    # 한 프레임을 읽어옴
    ret, frame = cap.read()
    if ret is False:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Text 
    font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
    # Display
    cv2.setMouseCallback("Circle", on_mouse, frame)
    cv2.circle(frame, center, 30, (0,0,255), 3)
    cv2.putText(frame, 'Circle', center, font, 3, (255, 255, 255), 5)
    cv2.imshow("Circle",frame)
    

    # 1 ms 동안 대기하며 키 입력을 받고 'q' 입력 시 종료
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
