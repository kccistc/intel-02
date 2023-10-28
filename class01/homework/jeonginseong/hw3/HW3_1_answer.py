import numpy as np
import cv2

# Read from the first camera device
cap = cv2.VideoCapture(0)

cv2.namedWindow("Camera")

topLeft = (100, 100)
bottomRight = (350, 350)

c_point = None

# mouse callback function
def click_mouse(event, x, y, flags, param):
    global c_point
    if event == cv2.EVENT_LBUTTONDOWN:
        c_point = (x, y)

while cap.isOpened():
    # 한 프레임을 읽어옴
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    # mouse event
    cv2.setMouseCallback('Camera', click_mouse)

    if c_point is not None:
        cv2.circle(frame, c_point, 10, (0, 255, 0), 1)

    # Line
    cv2.line(frame, topLeft, bottomRight, (255, 255, 0), 3)

    # Rectangle
    cv2.rectangle(frame, topLeft, bottomRight, (0, 255, 255), 3) 

    # Text 
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, 'good', (topLeft[0] + 40, topLeft[1]), font, 3, (0, 255, 0), 10)

    # Display
    cv2.imshow("Camera", frame)

    # 1 ms 동안 대기하며 키 입력을 받고 'q' 입력 시 종료
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

