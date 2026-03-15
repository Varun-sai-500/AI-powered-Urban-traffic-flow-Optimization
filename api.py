from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import threading
import time
import cv2
import numpy as np
import base64
import sqlite3
from datetime import datetime
from pipeline import VehicleDetector, LaneManager, TrafficAnalyzer, SignalController

app = FastAPI()

# Allow Streamlit UI to fetch status frames via browser JS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# DB setup
conn = sqlite3.connect('traffic.db', check_same_thread=False)
cursor = conn.cursor()
cursor.execute('''CREATE TABLE IF NOT EXISTS traffic_history (
    id INTEGER PRIMARY KEY,
    timestamp TEXT,
    lane TEXT,
    count INTEGER,
    prediction REAL,
    timing INTEGER
)''')
conn.commit()

# Global state
processing = False
processing_thread = None
stop_event = threading.Event()

config = {
    "video_source": "videoplayback.mp4",
    "confidence": 0.4,
    "threshold": 8
}

latest_data = {
    "counts": {"A": 0, "B": 0, "C": 0},
    "statuses": {"A": "NORMAL", "B": "NORMAL", "C": "NORMAL"},
    "predictions": {"A": 0.0, "B": 0.0, "C": 0.0},
    "timings": {"A": 30, "B": 30, "C": 30},
    "frame": None
}

detector = VehicleDetector()
lanes = {
    "A": np.array([[0, 360], [200, 360], [260, 0], [120, 0]]),
    "B": np.array([[200, 360], [400, 360], [380, 0], [260, 0]]),
    "C": np.array([[400, 360], [640, 360], [520, 0], [380, 0]])
}
lane_manager = LaneManager(lanes)
analyzer = TrafficAnalyzer()
signal_controller = SignalController()

class LaneUpdate(BaseModel):
    lanes: dict  # {"A": [[x,y], ...], ...}

class ThresholdUpdate(BaseModel):
    threshold: int

def video_processing_thread(video_path=None):
    global processing, latest_data
    if video_path is None:
        video_path = config["video_source"]

    current_source = video_path
    cap = cv2.VideoCapture(current_source)

    while processing and not stop_event.is_set():
        # Hot reload video source when config changes
        if config.get("video_source") != current_source:
            cap.release()
            current_source = config["video_source"]
            cap = cv2.VideoCapture(current_source)

        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop video
            continue

        detections = detector.detect(frame, conf=config.get("confidence", 0.4))
        counts, statuses = analyzer.analyze(detections, lane_manager)
        predictions = {lane: analyzer.predict_congestion(lane) for lane in counts}
        timings = signal_controller.adjust_timings(counts, predictions)

        # Draw on frame
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

        # Encode frame
        _, buffer = cv2.imencode('.jpg', frame)
        frame_b64 = base64.b64encode(buffer).decode('utf-8')

        latest_data = {
            "counts": counts,
            "statuses": statuses,
            "predictions": predictions,
            "timings": timings,
            "frame": frame_b64
        }

        # Save to DB
        timestamp = datetime.now().isoformat()
        for lane in counts:
            cursor.execute('INSERT INTO traffic_history (timestamp, lane, count, prediction, timing) VALUES (?, ?, ?, ?, ?)',
                           (timestamp, lane, counts[lane], predictions[lane], timings[lane]))
        conn.commit()

        time.sleep(0.1)  # slow down for demo

    cap.release()

@app.get("/status")
def get_status():
    return latest_data

@app.post("/start")
def start_processing(video_source: str = None):
    global processing, processing_thread, stop_event

    # If a new source is provided, update config
    if video_source:
        config["video_source"] = video_source

    # If already running, attempt a clean restart
    if processing and processing_thread is not None and processing_thread.is_alive():
        stop_event.set()
        processing = False
        processing_thread.join(timeout=2)

    stop_event.clear()
    processing = True
    processing_thread = threading.Thread(
        target=video_processing_thread,
        args=(config["video_source"],),
        daemon=True,
    )
    processing_thread.start()

    return {"message": "Processing started", "video_source": config["video_source"]}

@app.post("/stop")
def stop_processing():
    global processing, stop_event
    stop_event.set()
    processing = False
    return {"message": "Processing stopped"}

@app.put("/lanes")
def update_lanes(update: LaneUpdate):
    global lane_manager
    new_lanes = {k: np.array(v) for k, v in update.lanes.items()}
    lane_manager = LaneManager(new_lanes)
    return {"message": "Lanes updated"}

class ThresholdUpdate(BaseModel):
    threshold: int

class ConfigUpdate(BaseModel):
    video_source: str = None
    confidence: float = None
    threshold: int = None

@app.put("/threshold")
def update_threshold(update: ThresholdUpdate):
    analyzer.threshold = update.threshold
    config["threshold"] = update.threshold
    return {"message": "Threshold updated", "threshold": update.threshold}

@app.put("/config")
def update_config(update: ConfigUpdate):
    if update.video_source:
        config["video_source"] = update.video_source
    if update.confidence is not None:
        config["confidence"] = update.confidence
    if update.threshold is not None:
        config["threshold"] = update.threshold
        analyzer.threshold = update.threshold

    return {"message": "Config updated", "config": config}

@app.get("/config")
def get_config():
    return {"config": config}

@app.get("/analytics")
def get_analytics():
    cursor.execute('SELECT * FROM traffic_history ORDER BY timestamp DESC LIMIT 100')
    rows = cursor.fetchall()
    return {"history": rows}

@app.get("/analytics/export")
def export_analytics():
    import csv
    import io
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['id', 'timestamp', 'lane', 'count', 'prediction', 'timing'])
    cursor.execute('SELECT * FROM traffic_history')
    rows = cursor.fetchall()
    writer.writerows(rows)
    return {"csv": output.getvalue()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)