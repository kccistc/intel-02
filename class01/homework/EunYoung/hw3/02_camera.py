import numpy as np
import cv2
from random import shuffle

# Read from the first camera device
cap = cv2.VideoCapture(0)

def onMouse(event, x, y, flags, param):
    global frame, b, g, r

    if event == cv2.EVENT_LBUTTONDBLCLK:
        shuffle(b), shuffle(g), shuffle(r)
        cv2.circle(frame, (x, y), 50, (255,255,255), 5)

cv2.namedWindow("Camera")

# Initialize variables
frame = None
b, g, r = [0], [0], [0]

while(cap.isOpened()):
    # Read a frame
    ret, frame = cap.read()
    if ret is False:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Line
    topLeft = (50, 50)
    bottomRight = (300, 300)
    cv2.line(frame, topLeft, bottomRight, (0, 255, 0), 5)

    # Rectangle
    cv2.rectangle(frame, (topLeft[0] + 30, topLeft[1] + 30), (bottomRight[0] - 30, bottomRight[1] - 30), (0, 0, 255), 5)

    # Text
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, 'HELP!', (topLeft[0] + 95, topLeft[1]), font, 1, (200, 0, 255), 5)
    cv2.circle(frame, (465, 360), 50, (255, 0, 255), 5)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break

    cv2.setMouseCallback("Camera", onMouse)
    cv2.imshow("Camera", frame)

cap.release()
cv2.destroyAllWindows()

