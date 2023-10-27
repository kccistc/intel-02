import numpy as np
import cv2


def mouseCallback(event, x, y, flags, userdata):
    if (event == cv2.EVENT_LBUTTONDOWN) :
        # Add Circle pos
        circle_list.append((x,y))

# Read from the first camera device
cap = cv2.VideoCapture(0)

title = "camera"

circle_list = []


topLeft = (50, 50)
bottomRight = (300, 300)


# 성공적으로 video device 가 열렸으면 while 문 반복
while(cap.isOpened()):
    # 한 프레임을 읽어옴
    ret, frame = cap.read()
    if ret is False:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Circle
    for tuple in circle_list :
        cv2.circle(frame,(tuple[0],tuple[1]), 30, (0,255,255), 15)

    # Line
    cv2.line(frame, topLeft, bottomRight, (0, 255, 0), 5)


    # Rectangle
    cv2.rectangle(frame,
        [pt+30 for pt in topLeft], [pt-30 for pt in bottomRight], (0, 0, 255), 5)


    # Text
    FONT = cv2.FONT_HERSHEY_TRIPLEX
    cv2.putText(frame, 'sample of cv video capture ', [pt+80 for pt in topLeft], FONT, 4, (255, 128, 255), 6)


    # Display
    cv2.imshow(title, frame)


    # Set Mouse Left Btn Callback
    cv2.setMouseCallback(title, mouseCallback)


    # 1 ms 동안 대기하며 키 입력을 받고 'q' 입력 시 종료
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
