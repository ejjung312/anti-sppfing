import cv2
import mediapipe as mp

class FaceDetector:
    def __init__(self, minDetectionCon=0.5, modelSelection=0):
        self.minDetectionCon = minDetectionCon
        self.modelSelection = modelSelection
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(min_detection_confidence=self.minDetectionCon,
                                                                model_selection=modelSelection)

    def findFaces(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        bboxs = []
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                if detection.score[0] > self.minDetectionCon:
                    # 감지된 얼굴의 바운딩 박스 상대 좌표
                    bboxC = detection.location_data.relative_bounding_box
                    imageH, imageW, imageChannel = img.shape
                    # 상대 좌표를 실제 좌표로 변환
                    bbox = int(bboxC.xmin * imageW), int(bboxC.ymin * imageH), int(bboxC.width * imageW), int(bboxC.height * imageH)
                    # x축 중심 좌표, y축 중심 좌표
                    # bbox.x + bbox.width//2, bbox.y + bbox.height//2
                    cx, cy = bbox[0] + (bbox[2]//2), bbox[1] + (bbox[3]//2)
                    bboxInfo = {"Id": id, "bbox": bbox, "score": detection.score[0], "center": (cx,cy)}
                    bboxs.append(bboxInfo)

                    if draw:
                        img = cv2.rectangle(img, bbox, (255,0,255), 2)
                        cv2.putText(img, f'{int(detection.score[0] * 100)}%', (bbox[0], bbox[1]-20),
                                    cv2.FONT_HERSHEY_PLAIN, 2, (255,0,255), 2)

        return img, bboxs