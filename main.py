from ultralytics import YOLO
import cv2
import math
import cvzone
from sort import *

cap = cv2.VideoCapture('ju2.mp4')
mask = cv2.imread('jumask3.png')
model = YOLO('yolov8l.pt')
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
limits = [390, 350, 700, 350]
totalCount = []
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

while True:
    success, img = cap.read()
    output_width = 800
    output_height = 450
    img = cv2.resize(img, (output_width, output_height))
    mask = cv2.resize(mask, (output_width, output_height))
    imgwithmask = cv2.bitwise_and(img, mask)
    results = model(imgwithmask, stream=True)
    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            currentClass = classNames[cls]
            if currentClass == "car" or currentClass == "truck" or currentClass == "bus" \
                    or currentClass == "motorbike" and conf > 0.5:
                    currentArray = np.array([x1, y1, x2, y2, conf])
                    detections = np.vstack((detections, currentArray))
    resultsTracker = tracker.update(detections)

    for result in resultsTracker:
        x1, y1,x2, y2, id = result
        x1, y1, x2, y2, id = int(x1), int(y1), int(x2), int(y2), int(id)
        w, h = x2 - x1, y2 - y1
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        #cv2.circle(img, (cx, cy), 3,(0, 0, 255), cv2.FILLED)
        cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)
        if limits[0] < cx < limits[2] and limits[1] - 5 < cy < limits[1] + 5 :
            if totalCount.count(id) == 0:
                totalCount.append(id)
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)
        cvzone.putTextRect(img, text= f' Count: {len(totalCount)}', pos=(10, 50),
                            scale=3, thickness=3, colorT=(255,255,255),
                            colorR=(0,0,0), font=cv2.FONT_HERSHEY_PLAIN,
                            offset=10, border=None, colorB=(0, 0, 0))
        #cvzone.putTextRect(img,text = f' Count: {len(totalCount)}', pos=(50, 50))
        #putTextRect(img, text, pos, scale=3, thickness=3, colorT=(255, 255, 255),

    cv2.imshow('Image', img)
    cv2.waitKey(1)