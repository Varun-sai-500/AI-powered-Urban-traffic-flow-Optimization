import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLO model
model = YOLO("models/yolov8s.pt")

vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck

THRESHOLD = 5

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

    results = model(frame, conf=0.4, verbose=False)[0]

    for box in results.boxes:

        cls = int(box.cls[0])

        if cls in vehicle_classes:

            x1,y1,x2,y2 = map(int,box.xyxy[0])

            cx = (x1+x2)//2
            cy = (y1+y2)//2

            # ignore far vehicles
            if cy < 120:
                continue

            if inside_lane((cx,cy), lane_A):
                counts["A"] += 1
            elif inside_lane((cx,cy), lane_B):
                counts["B"] += 1
            elif inside_lane((cx,cy), lane_C):
                counts["C"] += 1

            # draw vehicle box
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

    status_A = "HIGH" if counts["A"] > THRESHOLD else "NORMAL"
    status_B = "HIGH" if counts["B"] > THRESHOLD else "NORMAL"
    status_C = "HIGH" if counts["C"] > THRESHOLD else "NORMAL"

    # terminal live update
    print(
        f"Lane A: {status_A} -> {counts['A']} | "
        f"Lane B: {status_B} -> {counts['B']} | "
        f"Lane C: {status_C} -> {counts['C']}",
        end="\r",
        flush=True
    )

    cv2.imshow("Vehicle Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()