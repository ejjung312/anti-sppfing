from utils.FaceDetectionModule import FaceDetector
import cv2

from utils.Utils import putTextRect

########################################
confidence = 0.8
offsetPercentageW = 10
offsetPercentageH = 20
camWidth, camHeight = 640, 480
########################################

cap = cv2.VideoCapture("data/person2.mp4")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, camWidth)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camHeight)
detector = FaceDetector()

while True:
    success, img = cap.read()
    img, bboxs = detector.findFaces(img, draw=False)

    imgH, imgW, _ = img.shape

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

            print(x, y, w, h, imgW, imgH)

            # -------- 흐릿한 영역 찾기 ---------
            imgFace = img[y:y+h, x:x+w]
            cv2.imshow("Face", imgFace)
            # 흐림의 정도 측정. 값이 크면 선명, 작으면 흐림
            blurValue = int(cv2.Laplacian(imgFace, cv2.CV_64F).var())

            # -------- 정규화 ---------


            # -------- Drawing ---------
            cv2.rectangle(img, (x,y,w,h), (255,0,0), 3)
            putTextRect(img, f'Blur: {blurValue}', (x,y-20))


    cv2.imshow("Image", img)
    cv2.waitKey(30)
