import numpy as np
import cv2

def mouse_event(event, x, y, flags, param):
    global mx, my

    if event == cv2.EVENT_LBUTTONDOWN:
        mx, my = x, y        

# Read from the first camera device
cap = cv2.VideoCapture(0)

topLeft = (100, 50)
bottomRight = (450, 300)

cv2.namedWindow('Camera')
cv2.setMouseCallback('Camera', mouse_event)

# 성공적으로 video device 가 열렸으면 while 문 반복
while(cap.isOpened()):
    # 한 프레임을 읽어옴
    ret, frame = cap.read()
    if ret is False:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Line
    cv2.line(frame, topLeft, bottomRight, (128, 255, 35), 1)

    # Rectangle
    cv2.rectangle(frame, 
        [pt+210 for pt in topLeft], [pt+10 for pt in bottomRight], (255, 0, 0), 3) 
    
    # Text 
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(frame, 'IMG', [pt+50 for pt in topLeft], font, 5, (128, 0, 255), 5)
   
    cv2.circle(frame, (350, 250), 15, (0, 0, 0), 5)
    if 'mx' in globals() and 'my' in globals():
        cv2.circle(frame, (mx, my), 30, (255, 255, 255), 3)

    # Display
    cv2.imshow('Camera',frame)

    # 1 ms 동안 대기하며 키 입력을 받고 'q' 입력 시 종료
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
