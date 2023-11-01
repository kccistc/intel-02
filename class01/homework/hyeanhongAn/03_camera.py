import numpy as np
import cv2

#font default value
font_size = 2
font_color = (0, 255, 255)

# Read from the first camera device
cap = cv2.VideoCapture(0)

topLeft = (150, 150)
bold = 0

# Callback function for the trackbar
def on_bold_trackbar(value):
    #print("Trackbar value:", value)
    global font_size
    font_size = value
def on_blue_trackbar(value):
    #print("Trackbar value:", value)
    global font_color
    font_color = (value, font_color[1], font_color[2])
def on_green_trackbar(value):
    #print("Trackbar value:", value)
    global font_color
    font_color = (font_color[0], value, font_color[2])
def on_red_trackbar(value):
    #print("Trackbar value:", value)
    global font_color
    font_color = (font_color[0], font_color[1], value)
cv2.namedWindow("Camera")

# trackbar add (B/G/R)
cv2.createTrackbar("bold", "Camera", font_size, 10, on_bold_trackbar)
cv2.createTrackbar("red", "Camera", font_color[0], 255, on_red_trackbar)
cv2.createTrackbar("green", "Camera", font_color[1], 255, on_green_trackbar)
cv2.createTrackbar("blue", "Camera", font_color[2], 255, on_blue_trackbar)


# 성공적으로 video device 가 열렸으면 while 문 반복
while(cap.isOpened()):
    # 한 프레임을 읽어옴
    ret, frame = cap.read()
    if ret is False:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Text 
    cv2.putText(frame, "Hello Open cv!!",
        topLeft, cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, 1+font_size)


    # Display
    cv2.imshow("Camera",frame)

    # 1 ms 동안 대기하며 키 입력을 받고 'q' 입력 시 종료
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
