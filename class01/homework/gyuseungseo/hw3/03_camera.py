import numpy as np
import cv2
# Read from the first camera device
cap = cv2.VideoCapture(0)

topLeft = (50, 150)
bold = 0
font_size = 1

# Callback function for the trackbar
def on_bold_trackbar(value):
    #print("Trackbar value:", value)
    global bold
    bold = value

def on_font_size(size):
    global font_size
    font_size = size

def nothing(x):
    pass

cv2.namedWindow("Camera")
cv2.createTrackbar("bold", "Camera", bold, 10, on_bold_trackbar)
cv2.createTrackbar("fontsize", "Camera", font_size, 10, on_font_size)

cv2.createTrackbar("R", "Camera", 0, 255, nothing)
cv2.createTrackbar("G", "Camera", 0, 255, nothing)
cv2.createTrackbar("B", "Camera", 0, 255, nothing)

# 성공적으로 video device 가 열렸으면 while 문 반복
while(cap.isOpened()):
    # 한 프레임을 읽어옴
    ret, frame = cap.read()
    if ret is False:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    r = cv2.getTrackbarPos("R", "Camera")
    g = cv2.getTrackbarPos("G", "Camera")
    b = cv2.getTrackbarPos("B", "Camera")

    # Text 
    cv2.putText(frame, "TEXT",
        topLeft, cv2.FONT_HERSHEY_SIMPLEX, font_size, (b, g, r), 1 + bold)


    # Display
    cv2.imshow("Camera",frame)

    # 1 ms 동안 대기하며 키 입력을 받고 'q' 입력 시 종료
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
