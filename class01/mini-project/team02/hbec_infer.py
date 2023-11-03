import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from imutils.contours import sort_contours
import imutils
from collections import deque


labels=['/', '*', '+', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '[', ']']
operator = {'[' : 4, ']' : 4, '*' : 2, '/' : 2, '+' : 3, '-' : 3}
read_model = tf.keras.saving.load_model("model/hbec_model.h5")

def calc(t1, t, t2):
    t1 = float(t1)
    t2 = float(t2)
    if t == '+': return t1 + t2
    elif t == '-': return t1 - t2
    elif t == '*': return t1 * t2
    elif t == '/': return t1 / t2

def stackCalc(splitedEq):
    stack_operator = deque()
    stack_result = deque()
    for t in splitedEq:
        if '0' <= t <= '9' or len(t) > 1:
            stack_result.append(t)
        elif t == '[' or not len(stack_operator) or stack_operator[-1] == '[':
            stack_operator.append(t)
        elif operator[t] < operator[stack_operator[-1]] and t != ']':
            stack_operator.append(t)
        else:
            while len(stack_operator) and operator[t] >= operator[stack_operator[-1]]:
                if stack_operator[-1] == '[':
                    stack_operator.pop()
                    break
                stack_result.append(stack_operator.pop())
            if t != ']':
                stack_operator.append(t)
        
    while stack_operator:
        temp = stack_operator.pop()
        if temp != '[':
            stack_result.append(temp)

    for t in stack_result:
        #if stack_operator: break
        if '0'<= t <= '9' or len(t) > 1:
            stack_operator.append(t)
        else:
            t2 = stack_operator.pop()
            t1 = stack_operator.pop()
            stack_operator.append(calc(t1, t, t2))

    return stack_operator[0]


def prediction(img):
    plt.imshow(img, cmap = 'gray')
    img = cv2.resize(img,(40, 40))
    norm_image = cv2.normalize(img, None, alpha = 0, beta = 1, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
    norm_image = norm_image.reshape((norm_image.shape[0], norm_image.shape[1], 1))
    case = np.asarray([norm_image])
    pred = read_model.predict([case]).argmax(axis=-1)
    
    return labels[pred[0]]

frame = cv2.imread('input/samples/equation_ex4.png')
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blurred, 30, 150)
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sort_contours(cnts, method="left-to-right")[0]
chars=[]
for c in cnts:
    (x, y, w, h) = cv2.boundingRect(c)
    if w*h>1200:
        roi = gray[y:y + h, x:x + w]
        chars.append(prediction(roi))
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


splitedEq = []
#backSplitedEq = []

tmp = ""
#flag = False
for c in chars:
    if '0' <= c <= '9':
        tmp += c
    #elif c == '=':
    #    flag = True
    else:
        if len(tmp) > 0:
            #if flag: back
            splitedEq.append(tmp)
            #else: frontSplitedEq.append(tmp)
            tmp = ""
        #if flag: back
        splitedEq.append(c)
        #else: frontSplitedEq.append(c)
if len(tmp) > 0:
    splitedEq.append(tmp)
    

print("splitedEquation : ", splitedEq)

print(stackCalc(splitedEq))
#if stackCalc(frontSplitedEq) == stackCalc(backSplitedEq):
#    print("right")
#else:
#    print("is not right")

