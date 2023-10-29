'''
- HW3 : OpenCV (Homework 1)
- Author : Yurim Lee
- Date : 2023.10.29.
- TODO
    1. Text 문구 / Font / 색상 / 크기 / 굵기 / 출력 위치 변경
    2. 마우스 왼쪽 버튼 클릭 시, 동그라미 그리기
'''

import numpy as np
import cv2

cap = cv2.VideoCapture(0)

topLeft = (100, 50)
bottomRight = (350, 300)
center = None 

# Callback function for Mouse Click Event
def on_mouse(event, x, y, flags, params):
    global center
    if event == cv2.EVENT_LBUTTONDOWN:
        center = (x, y)

windowTitle = "Camera"
cv2.namedWindow(windowTitle)

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret is False:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    # Draw line, rectangle
    cv2.line(frame, topLeft, bottomRight, (0, 255, 0), 5)
    cv2.rectangle(frame, 
        [pt+30 for pt in topLeft], [pt-30 for pt in bottomRight], (0, 0, 255), 5) 


    # [TODO 1] Text : 문구, 출력 위치, 폰트, 크기, 색상, 굵기 변경
    text = "<-Yurim"
    org = (bottomRight[0], bottomRight[1]-100)
    font = cv2.FONT_ITALIC
    size = 1 
    color = (255, 0, 255)
    thickness = 3
    cv2.putText(frame, text, org, font, size, color, thickness)

    
    # [TODO 2] 마우스 왼쪽 버튼 클릭 시, 원 그리기
    cv2.setMouseCallback(windowTitle, on_mouse)

    if center != None:
        cv2.circle(frame, center, 10, (255, 255, 0), 2)


    # Display
    cv2.imshow(windowTitle, frame)

    # 1 ms 동안 대기하며 키 입력을 받고 'q' 입력 시 종료
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
