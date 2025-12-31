"""
backend/main.py
Main FastAPI server for Project Drishti
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
import asyncio
import cv2
import httpx
import sys
import os

# Add parent directory to path to import detection modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detection.yolo_detector import DrishtiDetector
from detection.crowd_analyzer import CrowdAnalyzer
from backend.config import get_settings
from backend.rules_engine import RulesEngine
from backend.gemini_client import GeminiAnalyzer

# Initialize FastAPI app
app = FastAPI(
    title="Project Drishti API",
    description="Real-time AI crowd safety platform",
    version="1.0.0"
)

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
settings = get_settings()
detector = None
analyzer = None
rules_engine = None
gemini = None
video_capture = None
current_state = {}


# Pydantic models for API
class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    context: Dict[str, Any]


class StatusResponse(BaseModel):
    person_count: int
    density_level: str
    trend: str
    risk_score: float
    fire_detected: bool
    anomaly: Optional[str]
    recommendation: str


@app.on_event("startup")
async def startup_event():
    """Initialize all components on startup"""
    global detector, analyzer, rules_engine, gemini, video_capture
    
    print("\n" + "="*50)
    print("🚀 Project Drishti Starting Up...")
    print("="*50)
    
    # Initialize detector
    print("\n[1/5] Initializing YOLO detector...")
    detector = DrishtiDetector(
        model_path=settings.YOLO_MODEL_PATH,
        confidence=settings.DETECTION_CONFIDENCE,
        crowd_threshold_warning=settings.CROWD_THRESHOLD_WARNING,
        crowd_threshold_critical=settings.CROWD_THRESHOLD_CRITICAL
    )
    
    # Initialize analyzer
    print("\n[2/5] Initializing crowd analyzer...")
    analyzer = CrowdAnalyzer()
    
    # Initialize rules engine
    print("\n[3/5] Initializing rules engine...")
    rules_engine = RulesEngine(n8n_base_url=settings.N8N_WEBHOOK_BASE_URL)
    
    # Initialize Gemini
    print("\n[4/5] Initializing Gemini AI...")
    try:
        gemini = GeminiAnalyzer(api_key=settings.GEMINI_API_KEY)
    except Exception as e:
        print(f"⚠️  Warning: Gemini initialization failed: {e}")
        print("   Continuing without Gemini features...")
        gemini = None
    
    # Initialize video capture
    print("\n[5/5] Initializing video source...")
    video_source = settings.VIDEO_SOURCE
    if video_source.isdigit():
        video_source = int(video_source)
    
    video_capture = cv2.VideoCapture(video_source)
    
    if not video_capture.isOpened():
        print(f"❌ Error: Could not open video source: {video_source}")
    else:
        print(f"✅ Video source opened: {video_source}")
    
    # Start background detection loop
    asyncio.create_task(detection_loop())
    
    print("\n" + "="*50)
    print("✅ Project Drishti is READY!")
    print(f"📡 API available at: http://localhost:{settings.PORT}")
    print("="*50 + "\n")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global video_capture
    
    if video_capture:
        video_capture.release()
    
    print("\n👋 Project Drishti shutting down...")


async def detection_loop():
    """
    Main detection loop running in background
    Processes video frames and triggers actions
    """
    global current_state, video_capture, detector, analyzer, rules_engine, gemini
    
    print("[DETECTION LOOP] Started")
    
    while True:
        try:
            if video_capture and video_capture.isOpened():
                ret, frame = video_capture.read()
                
                if ret:
                    # Run detection
                    detection_result = detector.detect(frame)
                    
                    # Run analysis
                    analysis = analyzer.update(
                        person_count=detection_result.person_count,
                        fire_detected=detection_result.fire_detected,
                        timestamp=detection_result.timestamp,
                        density_level=detection_result.crowd_density,
                        zones=detection_result.zones
                    )
                    
                    # Update global state
                    current_state = {
                        "person_count": detection_result.person_count,
                        "density_level": detection_result.crowd_density,
                        "fire_detected": detection_result.fire_detected,
                        "trend": analysis.trend,
                        "rate_of_change": analysis.rate_of_change,
                        "risk_score": analysis.risk_score,
                        "anomaly_detected": analysis.anomaly_detected,
                        "anomaly_type": analysis.anomaly_type,
                        "recommendation": analysis.recommendation,
                        "zones": detection_result.zones,
                        "predicted_count_1min": analysis.predicted_count_1min,
                        "frame": detection_result.frame
                    }
                    
                    # Evaluate rules
                    actions = rules_engine.evaluate(
                        person_count=detection_result.person_count,
                        density_level=detection_result.crowd_density,
                        fire_detected=detection_result.fire_detected,
                        anomaly_detected=analysis.anomaly_detected,
                        anomaly_type=analysis.anomaly_type,
                        trend=analysis.trend,
                        rate_of_change=analysis.rate_of_change,
                        risk_score=analysis.risk_score,
                        zones=detection_result.zones
                    )
                    
                    # Execute immediate actions
                    for action in actions:
                        if action.immediate and action.webhook_endpoint:
                            asyncio.create_task(trigger_webhook(
                                action.webhook_endpoint,
                                action.payload
                            ))
                    
                    # Log status
                    print(f"\r[LIVE] Persons: {detection_result.person_count:3d} | "
                          f"Density: {detection_result.crowd_density:8s} | "
                          f"Risk: {analysis.risk_score:5.1f} | "
                          f"Actions: {len([a for a in actions if a.immediate])}", 
                          end="")
            
            await asyncio.sleep(0.1)  # ~10 FPS
            
        except Exception as e:
            print(f"\n❌ Error in detection loop: {e}")
            await asyncio.sleep(1)


async def trigger_webhook(url: str, payload: Dict[str, Any]):
    """Send webhook to n8n"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, timeout=5.0)
            if response.status_code == 200:
                print(f"\n✅ Webhook sent: {url}")
            else:
                print(f"\n⚠️  Webhook failed: {url} - Status {response.status_code}")
    except Exception as e:
        print(f"\n❌ Webhook error: {e}")


def generate_frames():
    """Generator for video streaming"""
    global current_state
    
    while True:
        if "frame" in current_state:
            frame = current_state["frame"]
            
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        asyncio.sleep(0.033)  # ~30 FPS


# API Endpoints

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "Project Drishti",
        "version": "1.0.0"
    }


@app.get("/status", response_model=StatusResponse)
async def get_status():
    """Get current crowd status"""
    if not current_state:
        raise HTTPException(status_code=503, detail="System initializing...")
    
    return StatusResponse(
        person_count=current_state.get("person_count", 0),
        density_level=current_state.get("density_level", "UNKNOWN"),
        trend=current_state.get("trend", "STABLE"),
        risk_score=current_state.get("risk_score", 0),
        fire_detected=current_state.get("fire_detected", False),
        anomaly=current_state.get("anomaly_type"),
        recommendation=current_state.get("recommendation", "Initializing...")
    )


@app.get("/video-feed")
async def video_feed():
    """Live video stream endpoint"""
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Natural language query endpoint"""
    if not gemini:
        raise HTTPException(
            status_code=503, 
            detail="Gemini AI not available"
        )
    
    if not current_state:
        raise HTTPException(
            status_code=503,
            detail="No data available yet"
        )
    
    # Remove frame from context (too large for Gemini)
    context = {k: v for k, v in current_state.items() if k != "frame"}
    
    answer = gemini.answer_query(
        question=request.question,
        context=context
    )
    
    return QueryResponse(
        answer=answer,
        context=context
    )


@app.get("/summary")
async def get_summary():
    """Get AI-generated situation summary"""
    if not gemini:
        raise HTTPException(
            status_code=503,
            detail="Gemini AI not available"
        )
    
    if not current_state:
        raise HTTPException(
            status_code=503,
            detail="No data available yet"
        )
    
    summary = gemini.generate_situation_summary(
        person_count=current_state.get("person_count", 0),
        density_level=current_state.get("density_level", "UNKNOWN"),
        trend=current_state.get("trend", "STABLE"),
        rate_of_change=current_state.get("rate_of_change", 0),
        predicted_count=current_state.get("predicted_count_1min", 0),
        risk_score=current_state.get("risk_score", 0),
        anomaly_type=current_state.get("anomaly_type"),
        zones=current_state.get("zones", {})
    )
    
    return {"summary": summary}


@app.get("/detailed-state")
async def get_detailed_state():
    """Get complete system state (without frame)"""
    if not current_state:
        raise HTTPException(
            status_code=503,
            detail="No data available yet"
        )
    
    # Return all state except frame
    return {k: v for k, v in current_state.items() if k != "frame"}


# Run server
if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True
    )