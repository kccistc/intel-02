import numpy as np
import cv2

fx=0
fy=0

def onMouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        #print(f'Left mouse button clicked at ({x}, {y})')
        global fx
        global fy
        fx=x
        fy=y

# Read from the first camera device
cap = cv2.VideoCapture(0)

topLeft = (00, 00)
bottomRight = (00, 300)

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
        [pt+30 for pt in topLeft], [pt-30 for pt in bottomRight], (0, 0, 255), 5) 

    # Display
    cv2.imshow("Camera",frame)

    #circle
    cv2.setMouseCallback("Camera", onMouse)
    cv2.circle(frame, (fx,fy), 100, (0,0,255), 3)
    
    # Text 
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, 'me?', (fx-100,fy-100), font, 2, (255, 0, 0), 10)
    
    # Display
    cv2.imshow("Camera",frame)
    
    # 1 ms 동안 대기하며 키 입력을 받고 'q' 입력 시 종료
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
