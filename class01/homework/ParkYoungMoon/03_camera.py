import numpy as np
import cv2
# Read from the first camera device
cap = cv2.VideoCapture(0)

topLeft = (150, 150)
bold = 0
size = 0
red = 0
green = 0
blue = 0
# Callback function for the trackbar
def on_bold_trackbar(value):
    #print("Trackbar value:", value)
    global bold
    bold = value
def on_size_trackbar(value):
    global size
    size = value
def on_red_trackbar(value):
    global red
    red = value
def on_green_trackbar(value):    
    global green
    green = value
def on_blue_trackbar(value):
    global blue
    blue = value

cv2.namedWindow("Camera")
cv2.createTrackbar("bold", "Camera", bold, 10, on_bold_trackbar)
cv2.createTrackbar("size", "Camera", size, 15, on_size_trackbar)
cv2.createTrackbar("R", "Camera", red, 255, on_red_trackbar)
cv2.createTrackbar("G", "Camera", green, 255, on_green_trackbar)
cv2.createTrackbar("B", "Camera", blue, 255, on_blue_trackbar)
# 성공적으로 video device 가 열렸으면 while 문 반복
while(cap.isOpened()):
    # 한 프레임을 읽어옴
    ret, frame = cap.read()
    if ret is False:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Text 
    cv2.putText(frame, "TEXT",
        topLeft, cv2.FONT_HERSHEY_SIMPLEX, 2 + size, (blue, green, red), 1 + bold)


    # Display
    cv2.imshow("Camera",frame)

    # 1 ms 동안 대기하며 키 입력을 받고 'q' 입력 시 종료
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
