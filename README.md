---
title: AI Powered Urban Traffic Flow Optimization
emoji: 🚦
colorFrom: blue
colorTo: green
sdk: docker
app_file: app.py
pinned: false
---
# AI-powered-Urban-traffic-flow-Optimization

A prototype for AI-driven urban traffic management using computer vision and predictive modeling.

## Features
- Real-time vehicle detection using YOLOv8
- Lane-based traffic density analysis
- Congestion prediction with simple ML
- Adaptive signal timing simulation
- Web dashboard for monitoring and control
- Historical analytics and export

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Ensure `videoplayback.mp4` is in the root directory (or update path in code)
3. Run the API server: `python api.py`
4. In another terminal, run the UI: `streamlit run ui/app.py`

## Usage
- Start processing to begin video analysis
- View live feed, counts, predictions, and signal timings
- Adjust congestion threshold, YOLO confidence, and video source
- Update lane regions via JSON input
- Check analytics tab for historical data and charts
- Export data to CSV

## Architecture
- `pipeline.py`: Core classes for detection, analysis, prediction
- `api.py`: FastAPI server with endpoints for control and data
- `ui/app.py`: Streamlit dashboard
- `models/yolov8s.pt`: Pretrained YOLO model


### hugging face deployment 
- Repo - https://huggingface.co/spaces/madhi9/smartflow-ai
- Link - https://madhi9-smartflow-ai.hf.space/