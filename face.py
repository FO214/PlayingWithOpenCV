import cv2 
import mediapipe as mp
import time

cap = cv2.VideoCapture(1)

pTime = 0
cTime = 0

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection(0.75)

while True:
    success, image = cap.read()

    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imageRGB)

    if results.detections:
        for id, detection in enumerate(results.detections):
            #mpDraw.draw_detection(image, detection)
            bboxC = detection.location_data.relative_bounding_box
            height, width, channels = image.shape

            bbox = int(bboxC.xmin * width), int(bboxC.ymin * height), int(bboxC.width * width), int(bboxC.height * height)
            cv2.rectangle(image, bbox, (255,0,255), 2)
            cv2.putText(image, "Confidence: " + str(int(detection.score[0]*100)) + "%", (bbox[0], bbox[1]-20), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0), 3)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(image, "FPS: " + str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)

    cv2.imshow("image", image)

    if cv2.waitKey(1) == ord("q"):
        break