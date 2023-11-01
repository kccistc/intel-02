import numpy as np
import cv2
# Read from the first camera device
cap = cv2.VideoCapture(0)

topLeft = (150, 150)
bold = 1
size=3
R=255
G=255
B=255

# Callback function for the trackbar
def on_bold_trackbar(value):
    #print("Trackbar value:", value)
    global bold
    bold = value
    
def on_size_trackbar(value):
    #print("Trackbar value:", value)
    global size
    size = value
    
def R_trackbar(value):
    #print("Trackbar value:", value)
    global R
    R = value
    
def G_trackbar(value):
    #print("Trackbar value:", value)
    global G
    G = value

def B_trackbar(value):
    #print("Trackbar value:", value)
    global B
    B = value

cv2.namedWindow("Camera")
cv2.createTrackbar("bold", "Camera", bold, 10, on_bold_trackbar)
cv2.createTrackbar("size", "Camera", size, 10, on_size_trackbar)
cv2.createTrackbar("R", "Camera", R, 255, R_trackbar)
cv2.createTrackbar("G", "Camera", G, 255, G_trackbar)
cv2.createTrackbar("B", "Camera", B, 255, B_trackbar)

# 성공적으로 video device 가 열렸으면 while 문 반복
while(cap.isOpened()):
    # 한 프레임을 읽어옴
    ret, frame = cap.read()
    if ret is False:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Text 
    cv2.putText(frame, "Hello",
        topLeft, cv2.FONT_HERSHEY_SIMPLEX, size, (B, G, R), 1 + bold)


    # Display
    cv2.imshow("Camera",frame)

    # 1 ms 동안 대기하며 키 입력을 받고 'q' 입력 시 종료
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
