from utils.FaceDetectionModule import FaceDetector
import cv2
from time import time

from utils.Utils import putTextRect


########################################
classId = 0 # 0: 가짜 1: 진짜
outputFolderPath = "dataset/datacollect"
confidence = 0.8
save = True
blurThreshold = 35 # Larger is more focus

offsetPercentageW = 10
offsetPercentageH = 20
camWidth, camHeight = 640, 480
floatingPoint = 6
########################################

cap = cv2.VideoCapture("data/person3.mp4")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, camWidth)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camHeight)
detector = FaceDetector()

while True:
    success, img = cap.read()
    originalImg = img.copy()
    img, bboxs = detector.findFaces(img, draw=False)

    imgH, imgW, _ = img.shape

    listBlur = []
    listInfo = []

    if bboxs:
        for bbox in bboxs:
            x,y,w,h = bbox["bbox"]
            score = bbox["score"]

            if score < confidence:
                continue

            # -------- 얼굴 감지 영역 여백 추가 ---------
            offsetW = (offsetPercentageW/100) * w
            x = max(0, int(x - offsetW))
            w = int(w + offsetW * 2)

            offsetH = (offsetPercentageH / 100) * h
            y = max(0, int(y - offsetH * 3))
            h = int(h + offsetH * 3.5)

            # print(x, y, w, h, imgW, imgH)

            # -------- 흐릿한 영역 찾기 ---------
            imgFace = img[y:y+h, x:x+w]
            # cv2.imshow("Face", imgFace)
            # 흐림의 정도 측정. 값이 크면 선명, 작으면 흐림
            blurValue = int(cv2.Laplacian(imgFace, cv2.CV_64F).var())

            if blurValue > blurThreshold:
                listBlur.append(True)
            else:
                listBlur.append(False)

            # -------- 정규화 ---------
            xCenter, yCenter = x+w/2, y+h/2
            xCenterNorm, yCenterNorm = min(round(xCenter/imgW, floatingPoint), 1), min(round(yCenter/imgH, floatingPoint), 1)
            wNorm, hNorm = min(round(w/imgW, floatingPoint), 1), min(round(h/imgH, floatingPoint), 1)
            print(xCenterNorm, yCenterNorm, wNorm, hNorm)

            listInfo.append(f"{classId} {xCenterNorm} {yCenterNorm} {wNorm} {hNorm}\n")

            # -------- Drawing ---------
            cv2.rectangle(img, (x,y,w,h), (255,0,0), 3)
            putTextRect(img, f'Score: {int(score*100)}% Blur: {blurValue}', (x,y), scale=2, thickness=3)

        if save:
            if all(listBlur) and listBlur != []:
                # 이미지 저장
                timeNow = time()
                timeNow = str(timeNow).split('.')
                timeNow = timeNow[0] + timeNow[1]
                cv2.imwrite(f"{outputFolderPath}/{timeNow}.jpg", originalImg)

                # 레이블 저장
                for info in listInfo:
                    f = open(f"{outputFolderPath}/{timeNow}.txt", 'a')
                    f.write(info)
                    f.close()


    cv2.imshow("Image", img)
    if cv2.waitKey(30) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
