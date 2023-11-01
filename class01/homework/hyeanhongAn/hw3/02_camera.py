import numpy as np
import cv2
# Read from the first camera device
cap = cv2.VideoCapture(0)

topLeft = (100, 100)
bottomRight = (300, 300)

#circle position 
#circle def
circle_pos = None

def draw_circle(event, x,y,flags, param):
    global circle_pos
    if event == cv2.EVENT_LBUTTONDOWN:
        circle_pos = (x,y)
        print(x,y)
        #check circle_pos

cv2.namedWindow("Camera")
cv2.setMouseCallback("Camera",draw_circle)

# 성공적으로 video device 가 열렸으면 while 문 반복
while(cap.isOpened()):
    # 한 프레임을 읽어옴
    ret, frame = cap.read()
    if ret is False:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Line
    cv2.line(frame, topLeft, bottomRight, (0, 255, 250), 7)
    
    #color change


    # Rectangle
    cv2.rectangle(frame, 
        [pt+50 for pt in topLeft], [pt-50 for pt in bottomRight], (255, 255, 255), 10) 
    #color -> white  pos change

    # Text 
    font = cv2.FONT_HERSHEY_COMPLEX 
    # font change FONT_HERSHEY_SIMPLEX -> FONT_HERSHEY_COMPLEX
    cv2.putText(frame, 'hello opencv!!', [pt+80 for pt in topLeft], font, 2, (255, 255, 255), 10)
    # change size, color

    #add circle
    if circle_pos is not None:
        cv2.circle(frame,circle_pos,50,(255,255,255),5)

    # Display
    cv2.imshow("Camera",frame)

    # 1 ms 동안 대기하며 키 입력을 받고 'q' 입력 시 종료
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
