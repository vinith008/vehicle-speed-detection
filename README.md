# Vehicle Speed Detection Web App

Real-time vehicle speed estimation from IP camera using YOLOv8, perspective correction (homography), Flask streaming, and EMA smoothing.

## Features
- Live video feed in browser
- Vehicle tracking & speed display in km/h
- Perspective correction for accurate real-world speeds
- Multi-threaded smooth capture

## Requirements
- Python 3.9+
- IP Webcam app running on phone (or any RTSP/MJPG camera)

## Quick Setup Instructions

1. Clone the repo
   ```bash
   git clone https://github.com/yourusername/vehicle-speed-detection-web.git
   cd vehicle-speed-detection-web