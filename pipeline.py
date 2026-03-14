import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("models/yolov8s.pt")

vehicle_classes = [2,3,5,7]

THRESHOLD = 8

cap = cv2.VideoCapture("videoplayback.mp4") 

lane_A = np.array([[0,360],[200,360],[260,0],[120,0]])
lane_B = np.array([[200,360],[400,360],[380,0],[260,0]])
lane_C = np.array([[400,360],[640,360],[520,0],[380,0]])

def inside_lane(point, lane):
    return cv2.pointPolygonTest(lane, point, False) >= 0

while True:

    ret, frame = cap.read()
    if not ret:
        break

    counts = {"A":0,"B":0,"C":0}

    results = model(frame, conf=0.4)[0]

    for box in results.boxes:

        cls = int(box.cls[0])

        if cls in vehicle_classes:

            x1,y1,x2,y2 = map(int,box.xyxy[0])

            cx = (x1+x2)//2
            cy = (y1+y2)//2

            if inside_lane((cx,cy), lane_A):
                counts["A"] += 1
            elif inside_lane((cx,cy), lane_B):
                counts["B"] += 1
            elif inside_lane((cx,cy), lane_C):
                counts["C"] += 1

            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

    cv2.polylines(frame,[lane_A],True,(255,0,0),2)
    cv2.polylines(frame,[lane_B],True,(255,0,0),2)
    cv2.polylines(frame,[lane_C],True,(255,0,0),2)

    y=40

    for lane,count in counts.items():

        status="HIGH" if count>THRESHOLD else "NORMAL"
        color=(0,0,255) if status=="HIGH" else (0,255,0)

        cv2.putText(
            frame,
            f"Lane {lane}: {count} ({status})",
            (20,y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

        y+=30

    cv2.imshow("Lane Density",frame)

    if cv2.waitKey(1)==27:
        break

cap.release()
cv2.destroyAllWindows()