import numpy as np
import cv2
# Read from the first camera device
cap = cv2.VideoCapture(0)

topLeft = (50, 50)
bottomRight = (300, 300)


# clicked Mouse
def on_mouse(event, x,y,flags, param):

    global center, draw

    if event == cv2.EVENT_LBUTTONDOWN: 
        center = (x,y)
        draw = True

cv2.namedWindow("camera")
cv2.setMouseCallback("camera", on_mouse)

center = (0,0)
draw = False

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

    # Text 
    font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
    cv2.putText(frame, 'Hello!',(0,479), font, 3, (255, 205, 255), 3)

    # Circle 
    if draw:
        cv2.circle(frame, center,30, (255,205,255), -1)  

    # Display
    cv2.imshow("camera",frame)

    # 1 ms 동안 대기하며 키 입력을 받고 'q' 입력 시 종료
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
