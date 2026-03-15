import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.linear_model import LinearRegression

vehicle_classes = [2, 3, 5, 7]

class VehicleDetector:
    def __init__(self, model_path="models/yolov8s.pt"):
        self.model = YOLO(model_path)

    def detect(self, frame, conf=0.4):
        results = self.model(frame, conf=conf)[0]
        detections = []
        for box in results.boxes:
            cls = int(box.cls[0])
            if cls in vehicle_classes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                detections.append((cx, cy, x1, y1, x2, y2))
        return detections

class LaneManager:
    def __init__(self, lanes):
        self.lanes = lanes  # dict: {"A": np.array(...), ...}

    def get_lane(self, point):
        for lane_name, lane in self.lanes.items():
            if cv2.pointPolygonTest(lane, point, False) >= 0:
                return lane_name
        return None

    def draw_lanes(self, frame):
        for lane in self.lanes.values():
            cv2.polylines(frame, [lane], True, (255, 0, 0), 2)

class TrafficAnalyzer:
    def __init__(self, threshold=8):
        self.threshold = threshold
        self.history = {"A": [], "B": [], "C": []}

    def analyze(self, detections, lane_manager):
        counts = {"A": 0, "B": 0, "C": 0}
        for cx, cy, x1, y1, x2, y2 in detections:
            lane = lane_manager.get_lane((cx, cy))
            if lane:
                counts[lane] += 1
        # Update history
        for lane in counts:
            self.history[lane].append(counts[lane])
            if len(self.history[lane]) > 100:
                self.history[lane].pop(0)
        statuses = {}
        for lane, count in counts.items():
            statuses[lane] = "HIGH" if count > self.threshold else "NORMAL"
        return counts, statuses

    def predict_congestion(self, lane):
        # Linear regression on the last N data points to predict the next count
        data = self.history[lane]
        if len(data) < 5:
            return 0
        N = min(len(data), 20)
        y = np.array(data[-N:]).reshape(-1, 1)
        X = np.arange(len(y)).reshape(-1, 1)
        model = LinearRegression()
        model.fit(X, y)
        pred = model.predict([[len(y)]])[0][0]
        return float(max(0, pred))

class SignalController:
    def __init__(self):
        self.green_times = {"A": 30, "B": 30, "C": 30}

    def adjust_timings(self, counts, predictions):
        total_density = sum(counts.values()) + sum(predictions.values())
        if total_density == 0:
            return self.green_times
        for lane in self.green_times:
            density = counts[lane] + predictions[lane]
            self.green_times[lane] = max(10, int(60 * density / total_density))
        return self.green_times

def process_video(video_path="videoplayback.mp4"):
    detector = VehicleDetector()
    lanes = {
        "A": np.array([[0, 360], [200, 360], [260, 0], [120, 0]]),
        "B": np.array([[200, 360], [400, 360], [380, 0], [260, 0]]),
        "C": np.array([[400, 360], [640, 360], [520, 0], [380, 0]])
    }
    lane_manager = LaneManager(lanes)
    analyzer = TrafficAnalyzer()
    signal_controller = SignalController()

    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame)
        counts, statuses = analyzer.analyze(detections, lane_manager)

        predictions = {lane: analyzer.predict_congestion(lane) for lane in counts}
        timings = signal_controller.adjust_timings(counts, predictions)

        # Draw detections
        for cx, cy, x1, y1, x2, y2 in detections:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        lane_manager.draw_lanes(frame)

        y = 40
        for lane, count in counts.items():
            status = statuses[lane]
            color = (0, 0, 255) if status == "HIGH" else (0, 255, 0)
            pred = predictions[lane]
            timing = timings[lane]
            cv2.putText(
                frame,
                f"Lane {lane}: {count} ({status}) Pred: {pred:.1f} Green: {timing}s",
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )
            y += 30

        cv2.imshow("Lane Density", frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_video()