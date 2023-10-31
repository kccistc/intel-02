import numpy as np
import cv2
# Read from the first camera device
cap = cv2.VideoCapture(0)

topLeft = (50, 50)
bold = 0
R = 0
G = 0
B = 0
size = 0


# Callback function for the trackbar
def on_bold_trackbar(value):
    #print("Trackbar value:", value)
    global bold
    bold = value

def on_size_trackbar(value):
    global size
    size = value

def Red(value):
    global R
    R = value

def Green(value):
    global G
    G = value

def Blue(value):
    global B
    B = value

cv2.namedWindow("Camera")
cv2.createTrackbar("bold", "Camera", bold, 10, on_bold_trackbar)
cv2.createTrackbar("size","Camera", size, 30, on_size_trackbar)
cv2.createTrackbar('R', 'Camera', 0, 255, Red)
cv2.createTrackbar('G', 'Camera', 0, 255, Green)
cv2.createTrackbar('B', 'Camera', 0, 255, Blue)

# 성공적으로 video device 가 열렸으면 while 문 반복
while(cap.isOpened()):
    # 한 프레임을 읽어옴
    ret, frame = cap.read()
    if ret is False:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Text 
    cv2.putText(frame, "TEXT",
        topLeft, cv2.FONT_HERSHEY_TRIPLEX, size+2, (B, G, R), 4 + bold)
    


    #Display
    cv2.imshow("Camera",frame)

    # 1 ms 동안 대기하며 키 입력을 받고 'q' 입력 시 종료
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
