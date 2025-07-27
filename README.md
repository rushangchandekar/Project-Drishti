# 👁️ Project Drishti: AI-Powered Event Safety System

Project Drishti is a real-time, AI-driven crowd management system designed to detect, forecast, and respond to potential hazards during large-scale public events. By integrating drone video, advanced AI models, and automated dispatch systems, Drishti acts as an intelligent central nervous system for event safety.

## 🚀 Key Features

- **Drone Video Ingestion**: Real-time aerial video capture for live monitoring.
- **Gemini Vision Pro Integration**: Multimodal AI detects crowd behaviors and anomalies.
- **Heatmap Generation**: Visualizes live crowd density and high-risk zones.
- **Automated Anomaly Detection**: Identifies panic, congestion, counter-flow, static overcrowding, and fire/smoke.
- **Proactive Bottleneck Forecasting**: Predicts dangerous surges using historical and live data.
- **LLM Summary Agent**: Converts raw data into actionable, human-readable briefs.
- **Intelligent Alerting**: Categorized, confidence-scored alerts dispatched via Google Cloud Pub/Sub.
- **Agent Dispatch System**: Automatically routes ground personnel with directions via Google Maps API.
- **System Resilience**: Handles interruptions with intelligent fallback logic and alerting.

## 🧠 Architecture Overview

Drishti follows an event-driven architecture on Google Cloud Platform, with key components:

- Cloud Run / VMs – For video processing and agent services  
- Cloud Functions – For alert generation, summarization, and database writing  
- Vertex AI + Gemini Vision Pro – For visual analysis and prediction  
- Cloud Firestore – Stores incidents, alerts, and summaries  
- Cloud Pub/Sub – Manages real-time event communication  
- Cloud Monitoring – Ensures health, logs, and fallback triggers

A full architecture diagram is available in the repository under `/docs/`.

## 🧰 Technologies Used

- Vertex AI – Forecasting & custom ML models  
- Gemini Vision Pro – Multimodal visual analysis  
- Google Maps API – Smart routing for field agents  
- Cloud Pub/Sub – Messaging between AI agents & services  
- Cloud Firestore – Incident & alert database  
- Cloud Run / Functions – Scalable backend logic  
- Python, OpenCV, FFmpeg – Core processing

## ⚙️ Installation & Setup

### ✅ Prerequisites

- GCP project with billing enabled  
- Python 3.9+  
- gcloud CLI installed and authenticated  
- ffmpeg installed  
- Vertex AI, Pub/Sub, Firestore APIs enabled

### 📦 Setup

```bash
git clone https://github.com/your-username/project-drishti.git
cd project-drishti

python3 -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\activate.bat

pip install -r requirements.txt
```

### 🔐 Environment Variables

Create a `.env` file in the root directory and add the following:

```env
GCP_PROJECT_ID=your-project-id
GEMINI_API_KEY=your-gemini-api-key
```

## ☁️ Google Cloud Deployment

1. **Enable Services**  
- Firestore (Native mode)  
- Vertex AI  
- Cloud Pub/Sub  
- Cloud Functions

2. **Create Pub/Sub Topics**  
- raw-vision-data  
- anomaly-alerts  
- bottleneck-alerts  
- llm-summaries  
- system-health-alerts

3. **Deploy Cloud Functions (example)**

```bash
gcloud functions deploy anomaly_agent \
  --runtime python311 \
  --trigger-topic raw-vision-data \
  --entry-point handle_anomaly_event \
  --region your-region \
  --service-account your-svc-account@your-project-id.iam.gserviceaccount.com \
  --set-env-vars GCP_PROJECT_ID=your-project-id \
  --memory 512MB
```

Repeat for other functions like `forecast_agent`, `llm_summary_agent`, `firestore_writer`, etc.

## 🧪 Simulate Local Processing

To simulate live drone feed processing locally:

- Place a sample video named `sample_video.mp4` in the project root  
- Run the processor:

```bash
python main_app/video_processor.py
```

This script will:  
- Extract frames from the video  
- Send frames to Gemini Vision Pro for analysis  
- Detect crowd anomalies and forecast bottlenecks  
- Publish alerts to Pub/Sub

## 📊 Monitoring

- Firestore Console – View the `incidents` collection for stored alerts and summaries  
- Cloud Logging – Inspect logs for Cloud Functions and processing scripts  
- Cloud Pub/Sub – Monitor message flow between components  
- Cloud Monitoring – Set custom metrics and health alerts

## 🔮 Future Enhancements

- Autonomous drone dispatch for real-time surveillance  
- Facial recognition-based Lost & Found system  
- Sentiment analysis from social media inputs  
- Augmented reality overlays for on-ground staff  
- Integration with smart city IoT systems (e.g., smart lights, PA systems)  
- Continuous learning feedback loop for AI improvement

## 🤝 Contributing

We welcome contributions to Project Drishti! Please follow these steps:

1. Fork the repository  
2. Create a feature branch (`git checkout -b feature-name`)  
3. Commit your changes  
4. Push to your fork  
5. Submit a Pull Request  

For major changes, open an issue first to discuss your idea.

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## ✉️ Contact

For support, questions, or collaboration opportunities, please open an issue on the repository.
