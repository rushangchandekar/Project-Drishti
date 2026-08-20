"""
backend/main.py
Project Drishti - Integrated Backend
Detection + Intelligence + n8n Webhook Integration
"""

import sys
import os
import cv2
import asyncio
import time
from pathlib import Path
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager
from sqlalchemy.orm import Session

from fastapi import FastAPI, HTTPException, File, UploadFile, Request, Depends
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from backend.intelligence.decision_engine import DecisionIntelligence
    from backend.intelligence.context_analyzer import ContextAnalyzer
except ImportError:
    from intelligence.decision_engine import DecisionIntelligence
    from intelligence.context_analyzer import ContextAnalyzer

# Backend internal imports
from backend.config import get_settings
from backend.database import get_db
from backend import crud
from backend.models import QueryRequest, VideoSourceRequest, SystemConfigRequest
from backend.models_db import init_db
from backend import state
from backend.video_processing import intelligent_detection_loop
from backend.video_stream import generate_frames, generate_frames_fast
from backend.webhooks import query_n8n_agent

settings = get_settings()

# ============================================================================
# LIFESPAN MANAGEMENT
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    
    print("\n" + "="*70)
    print("PROJECT DRISHTI - COMPLETE SYSTEM STARTUP (PERSISTENT DB)")
    print("="*70)
    
    # ===== DATABASE INITIALIZATION =====
    print("\n[1/3] Initializing Database & Seed Actuators...")
    try:
        init_db()
        print("   [OK] Database initialized & models ready.")
    except Exception as db_err:
        print(f"   [WARN] Database init warning: {db_err}")

    # ===== INTELLIGENCE LAYER =====
    print("\n[2/3] Initializing Intelligence Layer...")
    
    state.context_analyzer = ContextAnalyzer()
    
    try:
        state.decision_intelligence = DecisionIntelligence(gemini_api_key=settings.GEMINI_API_KEY)
        print("   [OK] Decision intelligence ready (with Gemini)")
    except Exception as e:
        state.decision_intelligence = DecisionIntelligence(gemini_api_key=None)
        print(f"   [WARN] Decision intelligence ready (without Gemini): {e}")
    
    print(f"   [INFO] n8n Webhook Base: {settings.N8N_WEBHOOK_BASE_URL}")
    
    # ===== MULTI-STREAM MANAGER =====
    print("\n[3/3] Initializing Multi-Stream Manager...")
    from backend.multi_stream import MultiStreamManager
    state.stream_manager = MultiStreamManager(max_streams=settings.MAX_CAMERA_STREAMS)
    print(f"   [OK] Multi-stream manager ready (max {settings.MAX_CAMERA_STREAMS} streams)")

    # ===== BACKGROUND TASKS =====
    print("\n[4/4] Starting Background Tasks...")
    
    state.detection_task = asyncio.create_task(intelligent_detection_loop())
    print("   [OK] Intelligent detection loop created (running in background)")
    
    print("\n" + "="*70)
    print("PROJECT DRISHTI IS FULLY OPERATIONAL!")
    print(f"API available at: http://localhost:{settings.PORT}")
    print(f"Dashboard: http://localhost:{settings.PORT}/video-feed")
    print("="*70 + "\n")
    
    # ===== RUN =====
    yield
    
    # ===== SHUTDOWN =====
    print("\n[SHUTDOWN] Project Drishti shutting down...")
    
    if state.detection_task and not state.detection_task.done():
        state.detection_task.cancel()
        try:
            await asyncio.wait_for(state.detection_task, timeout=2.0)
        except (Exception, asyncio.CancelledError):
            pass
    
    if state.stream_manager:
        state.stream_manager.shutdown()
        print("📹 Multi-stream manager shut down")

    if state.video_capture and state.video_capture.isOpened():
        state.video_capture.release()
        print("📹 Video capture released")
    
    print("✅ Shutdown complete\n")


# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(
    title="Project Drishti API",
    description="Complete AI-powered crowd safety system with intelligent agents",
    version="3.0.0",
    lifespan=lifespan
)

from fastapi.staticfiles import StaticFiles
from backend.routes_auth import router as auth_router

# Register authentication routes (/auth/register, /auth/login, /auth/refresh, /auth/me)
app.include_router(auth_router)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data folder path (used by list-videos and static mount)
import os
data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Health check and system info"""
    return {
        "status": "online",
        "service": "Project Drishti",
        "version": "3.0.0",
        "components": {
            "fire_detection": state.fire_detector is not None,
            "crowd_analysis": state.crowd_detector is not None,
            "intelligence": state.decision_intelligence is not None,
            "n8n_webhooks": True
        }
    }

@app.get("/status")
async def get_status():
    """Get current system status"""
    with state.frame_lock:
        if not state.current_state:
            raise HTTPException(status_code=503, detail="System initializing...")
        
        return {
            "person_count": state.current_state.get("person_count", 0),
            "density_level": state.current_state.get("density_level", "UNKNOWN"),
            "density_value": state.current_state.get("density_value", 0),
            "trend": state.current_state.get("trend", "STABLE"),
            "risk_score": state.current_state.get("risk_score", 0),
            "fire_detected": state.current_state.get("fire_detected", False),
            "fire_confidence": state.current_state.get("fire_confidence", 0),
            "anomaly_detected": state.current_state.get("anomaly_detected", False),
            "anomaly_type": state.current_state.get("anomaly_type"),
            "anomaly_severity": state.current_state.get("anomaly_severity"),
            "situation_severity": state.current_state.get("situation_severity", "UNKNOWN"),
            "recommendation": state.current_state.get("recommendation", "Initializing..."),
            "webhooks_sent": state.current_state.get("webhooks_sent", 0),
            "strategic_guidance": state.current_state.get("strategic_guidance", ""),
            "zones": state.current_state.get("zones", {}),
            "detection_time_ms": state.current_state.get("detection_time_ms", 0),
            "decision_time_ms": state.current_state.get("decision_time_ms", 0),
            "total_loop_time_ms": state.current_state.get("total_loop_time_ms", 0),
            "venue_name": state.current_state.get("venue_name", "Loading Venue..."),
            "area_m2": state.current_state.get("area_m2", 0),
            "recent_agent_actions": state.recent_agent_actions,
            "agents_active": state.current_state.get("agents_active", 0),
            "autonomous_actions": state.current_state.get("autonomous_actions", []),
            "activities": state.current_state.get("activities", []),
            "scene_mood": state.current_state.get("scene_mood", "CALM"),
            "dominant_activity": state.current_state.get("dominant_activity"),
        }

@app.get("/video-feed")
async def video_feed():
    """Optimized live video stream"""
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/video-feed-fast")
async def video_feed_fast():
    """Ultra-fast low-latency video stream"""
    return StreamingResponse(
        generate_frames_fast(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/detailed-state")
async def get_detailed_state():
    """Get complete system state"""
    with state.frame_lock:
        if not state.current_state:
            raise HTTPException(status_code=503, detail="No data available")
        
        return {k: v for k, v in state.current_state.items() if k != 'frame'}

@app.get("/agent-statuses")
async def get_agent_statuses(db: Session = Depends(get_db)):
    """Get current statuses of all 9 agents from persistent database"""
    agents = crud.get_all_agents(db)
    if not agents:
        with state.frame_lock:
            return state.agent_statuses
            
    result = {}
    with state.frame_lock:
        st_copy = dict(state.agent_statuses)

    for agent in agents:
        st = st_copy.get(agent.agent_code, {})
        result[agent.agent_code] = {
            "agent_id": agent.agent_code,
            "name": agent.agent_name,
            "category": agent.category,
            "status": st.get("status", "completed" if agent.invocation_count > 0 else "idle"),
            "latency": f"{agent.last_latency_ms:.1f}ms",
            "execution_time_ms": agent.last_latency_ms,
            "invocations": agent.invocation_count,
            "invocation_count": agent.invocation_count,
            "last_active": agent.last_active_at.strftime("%H:%M:%S") if agent.last_active_at else "Now",
            "description": agent.description,
            "trigger_reason": st.get("trigger_reason"),
            "last_result": st.get("last_result"),
            "last_error": st.get("last_error")
        }
    return result

@app.get("/incidents")
async def get_incidents(severity: Optional[str] = None, db: Session = Depends(get_db)):
    """Fetch persistent incident audit log"""
    incidents = crud.get_recent_incidents(db, limit=50, severity=severity)
    return [
        {
            "id": inc.id,
            "timestamp": inc.created_at.strftime("%Y-%m-%dT%H:%M:%S"),
            "threat_level": inc.threat_level,
            "anomaly_code": inc.anomaly_code,
            "crowd_count": inc.crowd_count,
            "density_m2": inc.crowd_density_per_m2,
            "risk_score": inc.risk_score,
            "fire_detected": inc.fire_detected,
            "gemini_assessment": inc.gemini_assessment
        }
        for inc in incidents
    ]

@app.post("/agent-callback")
async def agent_callback(request: Request, db: Session = Depends(get_db)):
    """Receive actions taken by n8n agents and persist to database"""
    data = await request.json()
    agent_code = data.get('agent', 'AnomalyAgent')
    action_name = data.get('action', f"Execution by {agent_code}")
    latency_ms = data.get('latency_ms', 950.0)

    # 1. Update Agent Stats in DB
    crud.update_agent_stats(db, agent_code=agent_code, latency_ms=latency_ms)

    # 2. Log Autonomous Action in DB
    action_rec = crud.log_autonomous_action(
        db,
        action_name=action_name,
        target_channel=data.get('channel', 'WEBHOOK_N8N'),
        execution_status=data.get('status', 'EXECUTED'),
        payload_data=data
    )
    
    action_record = {
        'id': action_rec.id,
        'agent': agent_code,
        'timestamp': action_rec.executed_at.strftime("%Y-%m-%dT%H:%M:%S"),
        'status': action_rec.execution_status,
        'data': data
    }
    
    with state.frame_lock:
        state.recent_agent_actions.insert(0, action_record)
        if len(state.recent_agent_actions) > 50:
            state.recent_agent_actions.pop()
            
    return {"success": True, "message": "Action logged to database"}

@app.get("/n8n-status")
async def get_n8n_status():
    """Check n8n webhook connectivity"""
    import httpx
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                settings.N8N_WEBHOOK_BASE_URL.replace('/webhook', ''),
                timeout=5.0
            )
            return {
                'n8n_reachable': True,
                'status_code': response.status_code,
                'webhook_base': settings.N8N_WEBHOOK_BASE_URL,
                'total_webhooks_sent': state.performance_metrics['total_webhooks_sent']
            }
    except Exception as e:
        return {
            'n8n_reachable': False,
            'error': str(e),
            'webhook_base': settings.N8N_WEBHOOK_BASE_URL,
            'total_webhooks_sent': state.performance_metrics['total_webhooks_sent']
        }

@app.get("/intelligence-stats")
async def get_intelligence_stats():
    """Get decision intelligence statistics"""
    if not state.decision_intelligence:
        raise HTTPException(status_code=503, detail="Intelligence system not initialized")
    
    return state.decision_intelligence.get_stats()

@app.get("/performance")
async def get_performance():
    """Get system performance metrics"""
    return state.performance_metrics

# ============================================================================
# MULTI-STREAM API ENDPOINTS
# ============================================================================

@app.get("/streams")
async def list_streams():
    """List all camera streams managed by the multi-stream manager"""
    if not state.stream_manager:
        raise HTTPException(status_code=503, detail="Stream manager not initialized")
    return {
        "streams": state.stream_manager.list_streams(),
        "active_count": state.stream_manager.get_active_count(),
        "max_streams": state.stream_manager.max_streams
    }

@app.post("/streams/add")
async def add_stream(
    stream_id: str,
    source: str,
    source_type: str = "webcam",
    name: str = "Camera"
):
    """Add a new camera stream at runtime"""
    if not state.stream_manager:
        raise HTTPException(status_code=503, detail="Stream manager not initialized")
    
    success = state.stream_manager.add_stream(
        stream_id=stream_id,
        source=source,
        source_type=source_type,
        name=name
    )
    
    if not success:
        raise HTTPException(status_code=400, detail=f"Failed to add stream '{stream_id}'. Source may be invalid or max streams reached.")
    
    return {"success": True, "message": f"Stream '{stream_id}' added", "info": state.stream_manager.get_stream_info(stream_id)}

@app.post("/streams/remove")
async def remove_stream(stream_id: str):
    """Remove a camera stream"""
    if not state.stream_manager:
        raise HTTPException(status_code=503, detail="Stream manager not initialized")
    
    success = state.stream_manager.remove_stream(stream_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Stream '{stream_id}' not found")
    
    return {"success": True, "message": f"Stream '{stream_id}' removed"}

@app.get("/streams/{stream_id}/feed")
async def stream_feed(stream_id: str):
    """Get MJPEG video feed for a specific stream"""
    if not state.stream_manager:
        raise HTTPException(status_code=503, detail="Stream manager not initialized")
    
    info = state.stream_manager.get_stream_info(stream_id)
    if not info:
        raise HTTPException(status_code=404, detail=f"Stream '{stream_id}' not found")
    
    def generate_stream_frames():
        while True:
            frame = state.stream_manager.get_frame(stream_id)
            if frame is None:
                time.sleep(0.05)
                continue
            
            # Resize for streaming
            h, w = frame.shape[:2]
            if w > 800:
                scale = 800 / w
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
            
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 65])
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
            time.sleep(1.0 / 15)  # 15 FPS
    
    return StreamingResponse(
        generate_stream_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

def _fallback_query_handler(question: str, query_state: Dict[str, Any]) -> str:
    """Fallback handler when AI is unavailable"""
    q = question.lower()
    
    if 'count' in q or 'people' in q or 'crowd' in q:
        return f"There are currently {query_state.get('person_count', 0)} people detected in the venue."
    
    if 'fire' in q or 'smoke' in q:
        if query_state.get('fire_detected'):
            return f"WARNING: Fire detected! Confidence: {query_state.get('fire_confidence', 0):.1%}. Please evacuate immediately."
        return "No fire detected at this time."
    
    if 'risk' in q or 'danger' in q or 'safety' in q:
        score = query_state.get('risk_score', 0)
        return f"Current risk score is {score:.1f}/100. Situation is {query_state.get('situation_severity', 'stable')}."
    
    if 'status' in q or 'situation' in q:
        return f"System is effective. Crowd density is {query_state.get('density_level', 'NORMAL')} with {query_state.get('person_count', 0)} people."
        
    return "I am currently running in fallback mode (AI unavailable). I can answer basic questions about crowd count, fire status, and risk levels."


@app.post("/query")
async def query(request: QueryRequest):
    """Natural language query - tries n8n AI agent first, then local fallbacks"""
    with state.frame_lock:
        if not state.current_state:
            raise HTTPException(status_code=503, detail="System initializing...")
        state_copy = {k: v for k, v in state.current_state.items() if k != 'frame'}
    
    answer = None
    source = "unknown"
    
    # PRIORITY 1: Try n8n AI agent webhook
    try:
        n8n_answer = await query_n8n_agent(request.question, state_copy)
        if n8n_answer:
            answer = n8n_answer
            source = "n8n_agent"
    except Exception as e:
        print(f"[QUERY] n8n agent failed: {e}")
    
    # PRIORITY 2: Try direct Gemini
    if not answer:
        try:
            if state.decision_intelligence and state.decision_intelligence.gemini:
                answer = state.decision_intelligence.gemini.answer_query(request.question, state_copy)
                source = "gemini"
        except Exception as e:
            print(f"[QUERY] Gemini failed: {e}")
    
    # PRIORITY 3: Hardcoded fallback
    if not answer:
        answer = _fallback_query_handler(request.question, state_copy)
        source = "fallback"
    
    return {
        "question": request.question,
        "answer": answer,
        "source": source,
        "timestamp": time.time()
    }

@app.get("/summary")
async def get_summary():
    """Get AI-generated situation summary"""
    if not state.decision_intelligence or not state.decision_intelligence.gemini:
        raise HTTPException(status_code=503, detail="AI summary not available")
    
    with state.frame_lock:
        if not state.current_state:
            raise HTTPException(status_code=503, detail="No data available")
        
        summary = state.decision_intelligence.gemini.generate_situation_summary(
            person_count=state.current_state.get('person_count', 0),
            density_level=state.current_state.get('density_level', 'UNKNOWN'),
            trend=state.current_state.get('trend', 'STABLE'),
            rate_of_change=state.current_state.get('rate_of_change', 0),
            predicted_count=state.current_state.get('predicted_count_1min', 0),
            risk_score=state.current_state.get('risk_score', 0),
            anomaly_type=state.current_state.get('anomaly_type'),
            zones=state.current_state.get('zones', {})
        )
    
    return {"summary": summary, "timestamp": time.time()}

@app.get("/list-videos")
async def list_videos():
    """List available videos"""
    videos = []
    for file in Path(data_dir).glob("*"):
        if file.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
            size = file.stat().st_size
            # Skip files larger than 150MB to prevent potential streaming buffer issues
            if size > 150 * 1024 * 1024:
                continue
            videos.append({
                "name": file.name,
                "path": str(file.resolve()),
                "size": file.stat().st_size
            })
    
    return {"videos": videos, "count": len(videos)}

@app.post("/upload-video")
async def upload_video(file: UploadFile = File(...)):
    """Upload video file"""
    import shutil
    
    data_path = Path("../data")
    data_path.mkdir(exist_ok=True)
    file_path = data_path / file.filename
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    return {
        "success": True,
        "filename": file.filename,
        "path": str(file_path),
        "size": file_path.stat().st_size
    }

@app.post("/switch-source")
async def switch_source(request: VideoSourceRequest):
    """Switch video source"""
    try:
        if state.video_capture:
            state.video_capture.release()
            await asyncio.sleep(0.5)
        
        if request.type == "webcam":
            state.video_capture = cv2.VideoCapture(0)
            source_name = "Webcam"
        else:
            state.video_capture = cv2.VideoCapture(request.path)
            source_name = Path(request.path).name
        
        if not state.video_capture.isOpened():
            raise HTTPException(status_code=400, detail="Cannot open video source")
        
        state.video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        return {
            "success": True,
            "source": source_name,
            "type": request.type
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/configure")
async def configure_system(request: SystemConfigRequest):
    """Configure venue and switch video source"""
    try:
        # Convert sq ft to sq meters
        area_m2 = request.square_feet * 0.092903
        state.density_calculator.calibrate(area_m2)
        
        # Release current video
        if state.video_capture:
            state.video_capture.release()
            await asyncio.sleep(0.5)
            
        # Open new source
        if request.video_source_type == "webcam":
            state.video_capture = cv2.VideoCapture(0)
            source_name = "Webcam"
        else:
            state.video_capture = cv2.VideoCapture(request.video_path)
            source_name = Path(request.video_path).name
            
        if not state.video_capture.isOpened():
            raise HTTPException(status_code=400, detail="Cannot open video source")
            
        state.video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Save venue properties to state
        with state.frame_lock:
            state.current_state['venue_name'] = request.venue_name
            state.current_state['area_m2'] = float(area_m2)
            
        return {
            "success": True,
            "venue_name": request.venue_name,
            "source": source_name
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

from fastapi import WebSocket, WebSocketDisconnect

@app.websocket("/ws/telemetry")
async def websocket_telemetry(websocket: WebSocket):
    """Real-time WebSocket endpoint pushing telemetry updates to clients"""
    await websocket.accept()
    state.active_websockets.add(websocket)
    try:
        # Send initial state
        with state.frame_lock:
            if state.current_state:
                state_copy = {k: v for k, v in state.current_state.items() if k != 'frame'}
                await websocket.send_json(state_copy)
        
        while True:
            # Keep-alive loop
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        state.active_websockets.discard(websocket)

# Mount data folder for video previews (MUST be after all API routes)
if os.path.exists(data_dir):
    app.mount("/data", StaticFiles(directory=data_dir), name="data")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=False,
        log_level="info"
    )