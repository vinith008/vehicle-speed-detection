# Vehicle Speed Detection Web Application

A real-time vehicle speed detection system using YOLOv8 object tracking, homography-based perspective correction, and a Flask web interface for live streaming.

This project estimates vehicle speeds from an IP camera feed and displays them directly in a web browser.

---

## Features

- Real-time vehicle detection and tracking using YOLOv8
- Speed estimation in km/h using perspective transformation
- Exponential Moving Average (EMA) smoothing for stable speed values
- Live video streaming in browser using Flask
- Multi-threaded frame capture for smooth performance

---

## Technologies Used

Backend:
- Python
- Flask
- OpenCV
- Ultralytics YOLOv8
- NumPy
- SciPy

Frontend:
- HTML
- CSS

---

## Project Structure

Vehicle-Speed-Detection/
│
├── app.py # Main Flask server and speed detection logic
├── yolov8m.pt # YOLOv8 model file (medium)
├── yolov8n.pt # YOLOv8 model file (nano, optional)
├── static/
│ └── style.css # Dashboard styling
│
├── templates/
│ └── index.html # Web dashboard page
│
└── README.md
