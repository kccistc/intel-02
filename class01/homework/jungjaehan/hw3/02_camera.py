import numpy as np
import cv2
# Read from the first camera device
cap = cv2.VideoCapture(0)

topLeft = (50, 50)
bottomRight = (300, 300)
x, y = 0, 0

def mouse_event(event, mouse_x, mouse_y, flags, param):
    global x, y
    if event == cv2.EVENT_LBUTTONDOWN:
        x, y = mouse_x, mouse_y
        '''cv.circle(img(x,y),100,(125,155,189),-1)'''
        '''print(str(x))'''
        

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

    # Text 
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, 'me?', (100,100), font, 2, (255, 0, 0), 10)

    # draw circles
    cv2.namedWindow("Camera")
    cv2.imshow("Camera",frame)
    cv2.setMouseCallback("Camera", mouse_event)
    cv2.circle(frame, (x,y), 100, (0,255,255),3)
        
    
    # Display
    cv2.imshow("Camera",frame)

    # 1 ms 동안 대기하며 키 입력을 받고 'q' 입력 시 종료
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
