```markdown
# Vehicle Speed Detection Web Application

A real-time vehicle speed detection system using computer vision, YOLOv8 tracking, and perspective transformation.  
This web application streams live video from an IP camera and estimates vehicle speeds in km/h.

---

## Features

- Real-time vehicle detection and tracking using YOLOv8  
- Speed estimation using homography-based perspective correction  
- Live video streaming in browser  
- Multi-threaded frame capture for smooth performance  
- Exponential Moving Average (EMA) speed smoothing  
- Vehicle stop detection  
- Calibration zone overlay for measurement accuracy  

---

## Technologies Used

**Backend:** Python, Flask  
**Computer Vision:** OpenCV  
**Object Detection:** YOLOv8 (Ultralytics)  
**Tracking:** BoT-SORT Tracker  
**Math & Processing:** NumPy, SciPy  
**Frontend:** HTML, CSS  

---

## Project Structure

```

vehicle-speed-detection/
│
├── app.py               # Main Flask server & speed detection logic
├── yolov8m.pt           # YOLOv8 model weights
├── yolov8n.pt           # Lightweight YOLOv8 model (optional)
├── templates/
│   └── index.html       # Web dashboard template
├── static/
│   └── style.css        # UI styling
└── README.md

````

---

## How It Works

1. YOLOv8 detects and tracks vehicles in each frame.  
2. A homography matrix converts pixel movement into real-world distance.  
3. Speed is calculated from distance over time.  
4. EMA smoothing stabilizes the speed readings.  
5. Speeds are displayed above each tracked vehicle in km/h.  

---

## Requirements

- Python 3.9+  
- IP camera stream (Android IP Webcam or RTSP camera)  
- GPU recommended but not required  

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/vehicle-speed-detection.git
cd vehicle-speed-detection
````

### 2. Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Configuration

Open `app.py` and update the camera stream URL:

```python
IP_CAMERA_URL = "http://YOUR_IP:PORT/videofeed"
```

Adjust homography source points if your camera angle or road layout changes.

---

## Run the Application

```bash
python app.py
```

Then open your browser:

```
http://localhost:5000
```

---

## Speed Detection Logic

* Vehicles are tracked with unique IDs
* Distance is calculated in meters after perspective transformation
* Speed is computed as:

```
speed (km/h) = (distance / time) × 3.6
```

* EMA smoothing reduces sudden spikes
* Minimum and maximum speed thresholds filter noise

---

## Disclaimer

This system is for educational and experimental purposes.
Speed measurements depend on correct camera calibration and may not be legally accurate.

---

## Future Improvements

* Automatic lane detection
* License plate recognition
* Speed violation alerts
* Database logging
* Cloud deployment

---

## License

This project is open-source and available under the MIT License.

```
```
