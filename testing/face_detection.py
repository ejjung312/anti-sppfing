from utils.FaceDetectionModule import FaceDetector
from utils.Utils import *
import cv2

cap = cv2.VideoCapture("../video/person1.mp4")

detector = FaceDetector(minDetectionCon=0.5, modelSelection=0)

while True:
    success, img = cap.read()

    img, bboxs = detector.findFaces(img, draw=False)

    if bboxs:
        for bbox in bboxs:
            center = bbox["center"]
            x, y, w, h = bbox['bbox']
            score = int(bbox['score'][0] * 100)

            cv2.circle(img, center, 5, (255, 0, 255), cv2.FILLED)
            putTextRect(img, f'{score}%', (x, y - 10))
            cornerRect(img, (x, y, w, h))

    cv2.imshow("Image", img)
    cv2.waitKey(30)
