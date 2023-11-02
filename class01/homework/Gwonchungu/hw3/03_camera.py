
import numpy as np
import cv2

# Global variables for font properties and font color
font_size = 2
font_bold = 0
font_color = (0, 255, 255)  # Default color: Yellow (BGR format)

# Callback function for font size trackbar
def on_font_size_trackbar(value):
    global font_size
    font_size = value

# Callback function for font bold trackbar
def on_font_bold_trackbar(value):
    global font_bold
    font_bold = value

# Callback function for font color trackbars (R, G, B)
def on_font_color_trackbar_r(value):
    global font_color
    font_color = (value, font_color[1], font_color[2])

def on_font_color_trackbar_g(value):
    global font_color
    font_color = (font_color[0], value, font_color[2])

def on_font_color_trackbar_b(value):
    global font_color
    font_color = (font_color[0], font_color[1], value)

# Read from the first camera device
cap = cv2.VideoCapture(0)

cv2.namedWindow("Camera")

# Create trackbars for font size, font bold, and font color (R, G, B)
cv2.createTrackbar("Font Size", "Camera", font_size, 10, on_font_size_trackbar)
cv2.createTrackbar("Font Bold", "Camera", font_bold, 5, on_font_bold_trackbar)
cv2.createTrackbar("Font Color (R)", "Camera", font_color[0], 255, on_font_color_trackbar_r)
cv2.createTrackbar("Font Color (G)", "Camera", font_color[1], 255, on_font_color_trackbar_g)
cv2.createTrackbar("Font Color (B)", "Camera", font_color[2], 255, on_font_color_trackbar_b)

# 성공적으로 video device 가 열렸으면 while 문 반복
while(cap.isOpened()):
    # 한 프레임을 읽어옴
    ret, frame = cap.read()
    if ret is False:
        print("Can't receive frame (stream end?). Exiting ...")
        break

# Get the selected font color
    font_color_name = f"R:{font_color[0]}, G:{font_color[1]}, B:{font_color[2]}"

    # Text with font properties and color
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f'Font Size: {font_size}, Font Bold: {font_bold}, Font Color: {font_color_name}',
                (10, 30), font, font_size / 2, font_color, 1 + font_bold)


    # Display
    cv2.imshow("Camera",frame)

    # 1 ms 동안 대기하며 키 입력을 받고 'q' 입력 시 종료
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
