import cv2
import pickle
import os
import logging
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.templating import Jinja2Templates

# --- Industry Standard Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("Traffic-IQ")

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, 'data', 'models', 'knn_model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'data', 'models', 'scaler.pkl')
VIDEO_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'video4.mp4') 

app = FastAPI(title="Traffic-IQ Professional")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "src", "api", "templates"))

# --- Robust Asset Loading ---
try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    logger.info("✅ Machine Learning assets (Model & Scaler) loaded.")
except Exception as e:
    logger.error(f"❌ Critical Error loading assets: {e}")
    raise SystemExit("Required ML files missing.")

# CV Setup
object_detector = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=60, detectShadows=True)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
counts = {"Car": 0, "Truck": 0, "Bike": 0}
LINE_Y = 450 

def generate_frames():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        logger.error(f"❌ Failed to open video source at {VIDEO_PATH}")
        return

    while True:
        success, frame = cap.read()
        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        try:
            frame = cv2.resize(frame, (1020, 600))
            roi = frame[250:600, 0:1020]
            mask = object_detector.apply(roi)
            
            _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.medianBlur(mask, 5)
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.line(frame, (0, LINE_Y), (1020, LINE_Y), (255, 255, 255), 2)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if 800 < area < 80000: 
                    x, y, w, h = cv2.boundingRect(cnt)
                    y += 250
                    cx, cy = x + w//2, y + h//2
                    aspect_ratio = float(w) / h
                    
                    feat = scaler.transform([[area, aspect_ratio]])
                    pred_idx = model.predict(feat)[0]
                    label = {0: "Car", 1: "Truck", 2: "Bike"}[pred_idx]
                    
                    if label == "Truck" and area < 5000: label = "Car"
                    if label == "Bike" and area > 3000: label = "Car"

                    if LINE_Y - 6 < cy < LINE_Y + 6:
                        counts[label] += 1
                        logger.info(f"📈 {label} detected. Current Counts: {counts}")
                        cv2.circle(frame, (cx, cy), 10, (0, 255, 255), -1)

                    color = (0, 255, 0) if label == "Car" else (0, 0, 255)
                    if label == "Bike": color = (255, 255, 0)
                    
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, f"{label} {int(area)}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
        except Exception as frame_err:
            logger.warning(f"⚠️ Error processing frame: {frame_err}")
            continue

# --- API Endpoints ---
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/stats")
async def get_stats():
    return counts

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)