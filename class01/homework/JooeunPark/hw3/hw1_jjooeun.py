import numpy as np
import cv2
# Read from the first camera device
cap = cv2.VideoCapture(0)

topLeft = (50, 50)
bottomRight = (300, 300)

def on_mouse(event, x, y, flag, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # circle
        cv2.circle(frame, (x,y), 15, (255,0,0), -1)
        # text
        font = cv2.FONT_HERSHEY_PLAIN
        cv2.putText(frame, 'click', (x+20,y), font, 3, (255,0,0), 5)

        cv2.imshow("Camera", frame)
    
# 성공적으로 video device 가 열렸으면 while 문 반복
while(cap.isOpened()):
    # 한 프레임을 읽어옴
    ret, frame = cap.read()
    if ret is False:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Display
    cv2.imshow("Camera",frame)

    # 1 ms 동안 대기하며 키 입력을 받고 'q' 입력 시 종료
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break

    cv2.setMouseCallback("Camera", on_mouse, frame)

cap.release()
cv2.destroyAllWindows()
