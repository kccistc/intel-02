import numpy as np
import cv2
# Read from the first camera device
cap = cv2.VideoCapture(0)

topLeft = (50, 50)
bottomRight = (300, 300)

pos = []

def onMouseClicked(event, x, y, flags, param):
    global frame
    if event == cv2.EVENT_LBUTTONDOWN:
        pos.append((x, y))
    
cv2.namedWindow('Camera')

# Event : activated when the left mouse clicked    
cv2.setMouseCallback('Camera', onMouseClicked)

# 성공적으로 video device 가 열렸으면 while 문 반복
while(cap.isOpened()):
    # 한 프레임을 읽어옴
    global frame
    ret, frame = cap.read()
    if ret is False:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Line
    cv2.line(frame, topLeft, bottomRight, (0, 255, 0), 5)

    # Rectangle
    cv2.rectangle(frame, 
        [pt+30 for pt in topLeft], [pt-30 for pt in bottomRight], (0, 0, 255), 5) 

    # Text 
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, 'me', [pt+80 for pt in topLeft], font, 2, (0, 255, 255), 10)
    
    # Circle
    for p in pos:
        cv2.circle(frame, p, 60, (255, 255, 255), 3)

    # 1 ms 동안 대기하며 키 입력을 받고 'q' 입력 시 종료
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break


    # Display
    cv2.imshow("Camera", frame)


cap.release()
cv2.destroyAllWindows()
