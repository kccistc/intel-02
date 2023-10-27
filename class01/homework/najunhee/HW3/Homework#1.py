import cv2
topLeft = (50,50)
bold = 0
fontsize = 0
redcolor = 0
bluecolor = 0
greencolor = 0
cap = cv2.VideoCapture(0)
h= cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
fps = cap.get(cv2.CAP_PROP_FPS)
#Callback func for the trackbar
def on_bold_trackbar(value):
    global bold
    bold = value

def on_fontsize_trackbar(value):
    global fontsize
    fontsize = value

def on_RedColor_trackbar(value):
    global redcolor
    redcolor = value

def on_GreenColor_trackbar(value):
    global greencolor
    greencolor = value

def on_BlueColor_trackbar(value):
    global bluecolor
    bluecolor = value


cv2.namedWindow("Camera")
cv2.createTrackbar("bold", "Camera", bold, 10, on_bold_trackbar)
cv2.createTrackbar("font Size","Camera",fontsize, 10, on_fontsize_trackbar) 
#font color
cv2.createTrackbar("Red Color","Camera",redcolor, 255,on_RedColor_trackbar) 
cv2.createTrackbar("Green Color","Camera",greencolor, 255, on_GreenColor_trackbar)
cv2.createTrackbar("Blue Color","Camera",bluecolor, 255, on_BlueColor_trackbar)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_file = 'homework_opencv_2.mp4'
fps = 25.0
out = cv2.VideoWriter(output_file, fourcc, fps, (640,480))

while (cap.isOpened()):
    re, frame = cap.read()
    if re==0:
        print("can't read frame...")
        break
    Color = (1+bluecolor,1+greencolor, 1+redcolor)
    cv2.putText(frame, "Text", topLeft, cv2.FONT_HERSHEY_SIMPLEX, 1+fontsize,Color, 1+bold)
    cv2.imshow("camera",frame)
    out.write(frame)
    if cv2.waitKey(1)& 0xff ==ord("q"):
        break
cap.release()
out.release()
cv2.destroyAllWindows()



