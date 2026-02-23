import cv2
import numpy as np
import mediapipe as mp
import time
import os

# ------------------- HAND DETECTOR CLASS -------------------

class HandDetector:
    def __init__(self, detectionCon=0.7, maxHands=1):
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=False,
            max_num_hands=maxHands,
            min_detection_confidence=detectionCon,
            min_tracking_confidence=0.5
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, img):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(
                    img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[0]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
        return lmList

    def fingersUp(self, lmList):
        fingers = []
        if len(lmList) != 0:
            # Thumb
            if lmList[self.tipIds[0]][1] > lmList[self.tipIds[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)

            # Other fingers
            for id in range(1, 5):
                if lmList[self.tipIds[id]][2] < lmList[self.tipIds[id] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)
        return fingers

# ------------------- MAIN APP -------------------

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = HandDetector()

brushThickness = 10
eraserThickness = 80
drawColor = (255, 0, 255)

imgCanvas = np.zeros((720, 1280, 3), np.uint8)

xp, yp = 0, 0
pTime = 0

# Brush size slider
cv2.namedWindow("Controls")
cv2.createTrackbar("Brush Size", "Controls", 25, 100, lambda x: None)

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)

    brushThickness = cv2.getTrackbarPos("Brush Size", "Controls")

    img = detector.findHands(img)
    lmList = detector.findPosition(img)

    # ------------------- MODERN TOOLBAR -------------------

    cv2.rectangle(img, (0, 0), (1280, 100), (30, 30, 30), -1)

    # Buttons
    buttons = ["CLEAR", "PINK", "BLUE", "GREEN", "ERASER", "SAVE"]
    positions = [150, 350, 550, 750, 950, 1150]

    for i, text in enumerate(buttons):
        cv2.rectangle(img, (positions[i]-90, 20),
                      (positions[i]+90, 80),
                      (60, 60, 60), -1)
        cv2.putText(img, text,
                    (positions[i]-60, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2)

    # Current color indicator
    cv2.circle(img, (50, 50), 25, drawColor, -1)

    # ------------------- HAND LOGIC -------------------

    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        fingers = detector.fingersUp(lmList)

        # Selection Mode
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0

            if y1 < 100:
                if 60 < x1 < 240:
                    imgCanvas = np.zeros((720, 1280, 3), np.uint8)

                elif 260 < x1 < 440:
                    drawColor = (255, 0, 255)

                elif 460 < x1 < 640:
                    drawColor = (255, 0, 0)

                elif 660 < x1 < 840:
                    drawColor = (0, 255, 0)

                elif 860 < x1 < 1040:
                    drawColor = (0, 0, 0)

                elif 1060 < x1 < 1240:
                    filename = f"drawing_{int(time.time())}.png"
                    cv2.imwrite(filename, imgCanvas)
                    print(f"Saved as {filename}")

        # Drawing Mode
        if fingers[1] and not fingers[2]:

            cv2.circle(img, (x1, y1), 15, drawColor, -1)

            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            if drawColor == (0, 0, 0):
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
            else:
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)

            xp, yp = x1, y1

    # ------------------- MERGE CANVAS -------------------

    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)

    # ------------------- FPS -------------------

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f"FPS: {int(fps)}",
                (20, 700),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2)

    cv2.imshow("Air Canvas", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()