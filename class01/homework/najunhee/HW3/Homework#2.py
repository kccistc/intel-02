import numpy as np
import cv2
# Read from the first camera device
cap = cv2.VideoCapture(0)
h= cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
w= cap.get(cv2.CAP_PROP_FRAME_WIDTH)
topLeft = (100, 100)
bottomRight = (300, 350)
msg = "Capture"
color = (255,255,0) # cyan
point = (0,0)
#mouse click event
def Mouse_event(event, x, y, flag,param):
    global point
    if event ==cv2.EVENT_LBUTTONDOWN:
        print("Mouse Clicked and the Position is {},{}".format(x,y))
        point = (x,y)
cv2.namedWindow("Camera")
cv2.setMouseCallback("Camera",Mouse_event)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_file = 'homework_opencv_1.mp4'
fps = 25.0
out = cv2.VideoWriter(output_file, fourcc, fps, (640,480))

# 성공적으로 video device 가 열렸으면 while 문 반복
while(cap.isOpened()):
    # 한 프레임을 읽어옴
    ret, frame = cap.read()
    if ret is False:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Line
    cv2.line(frame, topLeft, bottomRight, (0, 255, 0), 5)
    # Rectangle
    cv2.rectangle(frame,
        [pt+30 for pt in topLeft], [pt-30 for pt in bottomRight], (0, 0, 255), 5)
    #Circle
    cv2.circle(frame, (int(w//2),int(h//2)), 25, (0,255,255),2)
    # Text 
    font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
    cv2.putText(frame, msg, [pt+80 for pt in topLeft], font, 3, color, 5)
    #draw mouse pos
    cv2.circle(frame, point, 25, (255,0, 255), 5)
    # Display
    cv2.imshow("Camera",frame)
    out.write(frame)
# 1 ms 동안 대기하며 키 입력을 받고 'q' 입력 시 종료
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

