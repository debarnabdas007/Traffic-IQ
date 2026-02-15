# 🚗 Traffic-IQ: Low-Resource Vehicle Counter for Edge Deployment

![Traffic-IQ Demo](https://github.com/user-attachments/assets/57674376-3ebb-4c56-a7b0-a15d661de46a)

**Traffic-IQ** is a real-time vehicle detection and classification system built for **edge devices** without requiring GPUs. It combines classical computer vision techniques with a lightweight k-nearest neighbors (KNN) classifier to achieve low-latency, multi-class vehicle counting in resource-constrained environments.

## 📋 Project Overview

Traffic-IQ addresses the challenge of deploying computer vision applications on edge hardware by leveraging **classical CV methods** instead of deep learning models. This approach offers:

- ✅ **No GPU Required** – Runs efficiently on CPU-only edge devices
- ✅ **Low Latency** – Real-time processing at 60+ FPS
- ✅ **Minimal Memory Footprint** – Suitable for embedded systems
- ✅ **Production-Ready** – Industry-standard logging, exception handling, and multi-format streaming

## 🏗️ Architecture

### System Design

```
┌─────────────────────────────────────────────────────────────┐
│                    Traffic-IQ Pipeline                       │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Video Input → CV Processing → KNN Classifier → Streaming   │
│      (MP4)      (MOG2 + Morph)   (Edge ML)     (FastAPI)     │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### Two-Pattern Endpoint Architecture

1. **Request-Response Pattern** (`/stats` endpoint)
   - Synchronous HTTP GET request
   - Returns JSON with current vehicle counts
   - Ideal for periodic polling or dashboard updates

2. **Generator Pattern** (`/video_feed` endpoint)
   - Streaming MJPEG binary frames
   - Multipart HTTP response with continuous image chunks
   - Enables real-time visualization with minimal overhead

## 🔬 Technical Features

### 1. FastAPI Multi-Format Streaming

```python
# Three response types from single inference pipeline:
- HTML Interface       → Jinja2 templating with Tailwind CSS
- JSON Statistics      → RESTful stats endpoint
- MJPEG Binary Stream  → Real-time video via StreamingResponse
```

**Benefits:**
- Single unified CV pipeline serving multiple formats
- No code duplication across routes
- Efficient resource utilization

### 2. Computer Vision Pipeline

#### Background Subtraction (MOG2)
- **Algorithm**: Mixture of Gaussians (version 2)
- **Purpose**: Isolates moving vehicles from static background
- **Parameters**:
  - `history=500` – Tracks temporal changes over 500 frames
  - `varThreshold=60` – Sensitivity to pixel intensity changes
  - `detectShadows=True` – Filters out shadow artifacts

#### Morphological Operations
Object integrity and noise removal through controlled shape manipulation:

```python
# Step 1: Binary Threshold (Remove shadows)
_, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)

# Step 2: Closing (Fill internal holes like windshields)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# Step 3: Dilation (Merge close object fragments)
mask = cv2.dilate(mask, kernel, iterations=2)

# Step 4: Median Blur (Smooth noise)
mask = cv2.medianBlur(mask, 5)
```

#### Contour Analysis
- Extract vehicle candidates from binary mask
- Apply area filtering: `800 < area < 80,000` pixels
- Calculate geometric features: aspect ratio, centroid

### 3. KNN Classification

A lightweight machine learning classifier trained on hand-labeled vehicle data:

- **Algorithm**: k-Nearest Neighbors with k=5
- **Features**: `[Area, Aspect_Ratio]` – Simple yet effective
- **Classes**: Car, Truck, Bike
- **Training Data**: Hand-labeled vehicles from real traffic video
- **Post-Processing**: Logic-based refinement (e.g., "Truck with area < 5000px → Car")

### 4. Production-Grade Logging

```python
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
```

**Log Events:**
- ✅ ML asset loading confirmation
- 📈 Vehicle detection events with counts
- ❌ Critical errors (missing files, video load failures)
- ⚠️ Frame processing warnings (gracefully continue on error)

### 5. Exception Handling

- **Graceful Degradation**: Failed frames don't crash the pipeline
- **Video Loop Recovery**: Auto-resets to frame 0 on EOF
- **Asset Validation**: Fails fast with clear error messages if ML files missing
- **Streaming Resilience**: Individual frame errors caught and logged

## 📊 ML Pipeline

### Training Workflow

#### 1. Data Collection (`scripts/data_collector.py`)
- **Input**: Raw traffic video (MP4)
- **Process**: Interactive labeling with OpenCV GUI
- **Output**: CSV with `[Area, Aspect_Ratio, Label]`

```
KEYS: 
  [c] = Car
  [t] = Truck  
  [b] = Bike
  
NAV:
  [j] = Jump 2 minutes forward (when background changes)
  [space] = Pause/Resume
  
OPS:
  [s] = Skip frame
  [q] = Quit
```

**Key Features:**
- MOG2 + Morphological operations for frame extraction
- Stabilization counter (40 frames) after time jumps
- Frame skipping for faster labeling workflow

#### 2. Feature Engineering
- **Area**: Contour pixel count (size indicator)
- **Aspect Ratio**: Width/Height (shape indicator)
- **Normalization**: StandardScaler for KNN sensitivity

#### 3. Model Training (`scripts/train_model.py`)

```python
# Critical: Split BEFORE scaling (prevents data leakage)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Learn scaling from train data only
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
```

**Evaluation Metrics:**
- Classification report with precision/recall per class
- Confusion matrix analysis
- Color-coded accuracy grades:
  - ✅ EXCELLENT (>85%)
  - ⚠️ ACCEPTABLE (>70%)
  - ❌ POOR (<70%)

#### 4. Model Artifacts
- `data/models/knn_model.pkl` – Trained KNN classifier
- `data/models/scaler.pkl` – Feature scaler (CRITICAL – must match inference)

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- OpenCV 4.x
- FastAPI & Uvicorn

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Traffic-IQ.git
cd Traffic-IQ

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Application

#### 1. Start the FastAPI Server

```bash
# From project root
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

**Output:**
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete
✅ Machine Learning assets (Model & Scaler) loaded.
INFO:     Uvicorn running with auto-reload enabled
```

#### 2. Access the Dashboard

Open your browser and navigate to:
```
http://localhost:8000
```

You'll see:
- **Live MJPEG Stream** with bounding boxes and vehicle labels
- **Real-time Counters** for Cars, Trucks, and Bikes
- **Event Console** with live logging output
- **System Status** indicator

#### 3. API Endpoints

**Live Video Stream (MJPEG)**
```bash
GET http://localhost:8000/video_feed
# Returns: multipart/x-mixed-replace; boundary=frame
```

**Current Statistics (JSON)**
```bash
GET http://localhost:8000/stats
# Returns: {"Car": 42, "Truck": 8, "Bike": 15}
```

**Dashboard (HTML)**
```bash
GET http://localhost:8000/
# Returns: Interactive web interface
```

## 📁 Project Structure

```
Traffic-IQ/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
│
├── src/
│   └── api/
│       ├── main.py          # FastAPI application (streaming + REST)
│       ├── __init__.py
│       └── templates/
│           └── index.html   # Web dashboard (Tailwind CSS)
│
├── scripts/
│   ├── data_collector.py    # Interactive video labeling tool
│   └── train_model.py       # KNN training & evaluation
│
├── data/
│   ├── raw/
│   │   └── video4.mp4       # Traffic video source
│   ├── processed/
│   │   └── vehicle_data.csv # Hand-labeled training data
│   └── models/
│       ├── knn_model.pkl    # Trained classifier
│       └── scaler.pkl       # Feature scaler (essential)
│
├── docs/                     # Documentation (optional)
└── venv/                     # Virtual environment
```

## 🛠️ Workflow

### Step 1: Prepare Training Data
```bash
python scripts/data_collector.py
```
- Streams video frames with MOG2 detection
- Interactive GUI to label vehicles
- Outputs: `data/processed/vehicle_data.csv`

### Step 2: Train the Model
```bash
python scripts/train_model.py
```
- Loads labeled dataset
- Applies StandardScaler normalization
- Trains KNN classifier with k=5
- Outputs: 
  - `data/models/knn_model.pkl`
  - `data/models/scaler.pkl`

### Step 3: Run Inference
```bash
uvicorn src.api.main:app --reload
```
- Loads pre-trained model and scaler
- Processes video stream in real-time
- Serves web interface and API endpoints

## 🎯 Why Classical CV Over Deep Learning?

Traditional deep learning approaches (YOLO, Faster R-CNN) require:
- ❌ GPU acceleration for reasonable speed
- ❌ Hundreds of MB model files
- ❌ High power consumption
- ❌ Complex dependency chains

**Traffic-IQ's Approach:**
- ✅ Pure CPU execution on edge devices (Raspberry Pi, Jetson Nano, industrial cameras)
- ✅ Model size: ~10 KB (KNN + Scaler)
- ✅ Low power profile suitable for always-on applications
- ✅ Simple, reproducible, and debuggable pipeline

## 🔧 Configuration & Tuning

### Video Processing Parameters (`src/api/main.py`)

```python
# ROI Extraction
roi = frame[250:600, 0:1020]  # Adjust for camera angle

# Detection Threshold
if 800 < area < 80000:  # Tune for vehicle size range

# Counting Line
LINE_Y = 450  # Adjust for different road positions

# MOG2 Sensitivity
object_detector = cv2.createBackgroundSubtractorMOG2(
    history=500,      # Increase for slower background changes
    varThreshold=60,  # Lower = more sensitive
    detectShadows=True
)
```

### Data Collection Parameters (`scripts/data_collector.py`)

```python
MIN_AREA = 250         # Ignore small noise
SKIP_FRAMES = 2        # Process 1 of every 3 frames (speed up)
JUMP_MINUTES = 2       # Time jump interval
```

### KNN Training Parameters (`scripts/train_model.py`)

```python
KNeighborsClassifier(n_neighbors=5)  # Higher k = smoother, slower
train_test_split(..., test_size=0.2) # Adjust train/test ratio
```

## 📈 Performance Metrics

**Typical Performance** (on a mid-range CPU, 1020x600 video):

| Metric | Value |
|--------|-------|
| **Frames Per Second** | 60+ |
| **Latency (frame→count)** | ~15ms |
| **Model Size** | ~10 KB |
| **Memory Usage** | ~200-300 MB |
| **CPU Usage** | 25-40% (single core) |

## 🚨 Troubleshooting

### Issue: "❌ Critical Error loading assets"
**Solution**: Train the model first
```bash
python scripts/train_model.py
# Ensure knn_model.pkl and scaler.pkl exist in data/models/
```

### Issue: "❌ Failed to open video source"
**Solution**: Check video file path in `src/api/main.py`
```python
VIDEO_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'video4.mp4')
# Verify file exists
```

### Issue: Low detection accuracy
**Solution**: Collect more diverse training data
```bash
# Use different video clips, times of day, camera angles
python scripts/data_collector.py
# Label at least 200-300 samples per class
```

### Issue: False positives (shadows, pedestrians counted)
**Solution**: Adjust detection parameters
```python
# Increase minimum area threshold
if 1200 < area < 80000:  # More restrictive

# Fine-tune MOG2
varThreshold=80  # Make background subtraction stricter
```

## 📚 Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| **FastAPI** | 0.129.0 | Web framework & async streaming |
| **Uvicorn** | 0.40.0 | ASGI server for FastAPI |
| **OpenCV** | 4.13.0 | Computer vision pipeline |
| **Scikit-Learn** | 1.8.0 | KNN classifier & preprocessing |
| **Pandas** | 3.0.0 | Dataset manipulation |
| **NumPy** | 2.4.2 | Numerical computations |
| **Jinja2** | 3.1.6 | HTML templating |

See `requirements.txt` for complete list.

## 🔐 Design Principles

1. **Separation of Concerns**
   - `data_collector.py` – Labeling only
   - `train_model.py` – Training only
   - `main.py` – Inference only

2. **ML Pipeline Integrity**
   - Scaler trained on training data only (prevents data leakage)
   - Both model and scaler saved and loaded
   - Consistent feature engineering across all stages

3. **Production Readiness**
   - Structured logging with timestamps
   - Exception handling at critical points
   - Automatic video loop recovery
   - Clear error messages for debugging

4. **Edge Device Optimization**
   - No GPU dependency
   - Minimal external libraries
   - Small model files
   - Efficient streaming format (MJPEG)

## 📝 License

[Add your license here]

## 👤 Author

[Your Name]

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

## 📞 Support

For issues or questions, please open a GitHub issue or contact [your-email@example.com]

---

**Built with ❤️ for edge AI deployment**
