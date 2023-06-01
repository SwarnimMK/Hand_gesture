import os
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np

#Variables  #1
width, height = 1280, 720  #webcam - dimensions
folderPath = "Presentation"

#camera setup
cap = cv2.VideoCapture(0) #id_no
cap.set(3, width)
cap.set(4, height)

#Get the list of presentation images #1way
# pathImages = sorted(os.listdir(folderPath))
# print(pathImages)

# lst = os.listdir(folderPath) #2ndway
# lst.sort()
# print(lst)

import re  #2#
def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(data, key=alphanum_key)

pathImages = sorted_alphanumeric(os.listdir(folderPath))
# print(pathImages)


#Variables #4#
imgNumber = 0
hs, ws = int(120*2), int(213*2) #another small img
gestureThreshold = 300
buttonPressed = False
buttonCounter = 0
buttonDelay = 12
annotations = [[]]
annotationNumber = -1
annotationStart = False

#Hand Detector
detector = HandDetector(detectionCon=0.8, maxHands=1)

while True: #3#
    #IMPORT IMAGES
    success, img = cap.read()
    img = cv2.flip(img, 1) #1=horizontalflip, 0=verticalflip
    pathFullImage = os.path.join(folderPath, pathImages[imgNumber]) #[0]=1.jpg
    imgCurrent = cv2.imread(pathFullImage)

    hands, img = detector.findHands(img)
    # gesture threshold line
    cv2.line(img, (0, gestureThreshold), (width, gestureThreshold), (0, 255, 0), 10) #if hand below 300, detect

    if hands and buttonPressed is False:
        hand = hands[0] #1 hand, but utne hi max value hai
        fingers = detector.fingersUp(hand)
        cx, cy = hand['center']
        lmList = hand['lmList'] #List of 21 landmark points
        indexFinger = lmList[8][0], lmList[8][1] #list of which fingers are up

        #Constrain values for easier drawing
        xVal = int(np.interp(lmList[8][0], [width // 2, width], [0, width]))
        yVal = int(np.interp(lmList[8][1], [150, height - 150], [0, height]))
        indexFinger = xVal, yVal

        if cy <=gestureThreshold: #if hand is at the height of the face

            #Gesture 1 = Left
            if fingers == [1, 0, 0, 0, 0]:
                print("Left!")
                if imgNumber > 0:
                    buttonPressed = True
                    annotations = [[]]
                    annotationNumber = -1
                    annotationStart = False
                    imgNumber -= 1 #reduce the imgno

            #Gesture 2 = Right
            if fingers == [0, 0, 0, 0, 1]:
                print("Right!")
                if imgNumber < len(pathImages) - 1:
                    buttonPressed = True
                    annotations = [[]]
                    annotationNumber = -1
                    annotationStart = False
                    imgNumber += 1 #increase the imgno

        # Gesture 3  - Show Pointer
        if fingers == [0, 1, 1, 0, 0]:
            cv2.circle(imgCurrent, indexFinger, 15, (0, 0, 255), cv2.FILLED)

        # Gesture 4  - Draw Pointer
        if fingers == [0, 1, 0, 0, 0]:
            if annotationStart is False:
                annotationStart = True
                annotationNumber += 1
                annotations.append([])
            cv2.circle(imgCurrent, indexFinger, 15, (0, 0, 255), cv2.FILLED)
            annotations[annotationNumber].append(indexFinger)
        else:
            annotationStart = False


        #Gesture 5 - Erase
        if fingers == [0, 1, 1, 1, 0]:
            if annotations:
                annotations.pop(-1)
                annotationNumber -= 1
                buttonPressed = True

    #Button Pressed ITERATIONS
    if buttonPressed:
        buttonCounter += 1
        if buttonCounter > buttonDelay:
            buttonCounter = 0
            buttonPressed = False


    for i in range (len(annotations)):
        for j in range(len(annotations[i])):
            if j != 0:
                cv2.line(imgCurrent, annotations[i][j-1], annotations[i][j], (0,0,200), 15)


    #Adding webcam Image on the slides
    imgSmall = cv2.resize(img, (ws, hs))
    h, w, _ = imgCurrent.shape #w, h of the slides
    imgCurrent[0:hs, w-ws:w] = imgSmall #topright corner placement[startpoint:endpoint]


    cv2.imshow("Image", img)
    cv2.imshow("Slides", imgCurrent)

    key = cv2.waitKey(1)
    if key == ord('q'): #if we press q, while loop breaks
        break



