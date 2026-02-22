"""
backend/main.py
Project Drishti - Integrated Backend
Detection + Intelligence + n8n Webhook Integration
"""

from fastapi import FastAPI, HTTPException, File, UploadFile, BackgroundTasks
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager
import asyncio
import cv2
import httpx
import sys
import os
from pathlib import Path
import time
import json
import threading

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Detection imports
from detection.yolo_detector import DrishtiDetector
from detection.fire_detector import AdvancedFireDetector
from detection.enhanced_crowd_detector import EnhancedCrowdDetector
from detection.density_calculator import IntelligentDensityCalculator
from detection.anomaly_detector import CrowdAnomalyDetector
from detection.crowd_analyzer import CrowdAnalyzer

# Intelligence imports
from intelligence.decision_engine import DecisionIntelligence
from intelligence.context_analyzer import ContextAnalyzer

# No local agents — agents are managed in n8n

# Config
from backend.config import get_settings

# ============================================================================
# PERFORMANCE SETTINGS
# ============================================================================

STREAM_QUALITY = 65          # JPEG quality (lower = faster, 50-80 recommended)
STREAM_FPS = 15              # Target streaming FPS
DETECTION_FRAME_SKIP = 3     # Process every Nth frame
MAX_STREAM_WIDTH = 800       # Max width for streaming
ENABLE_FRAME_RESIZE = True   # Resize frames for streaming

# ============================================================================
# GLOBAL STATE
# ============================================================================

settings = get_settings()

# Detection components
detector = None
fire_detector = None
crowd_detector = None
density_calculator = None
anomaly_detector = None
crowd_analyzer = None

# Intelligence components
context_analyzer = None
decision_intelligence = None

# System state
video_capture = None
current_state = {}
detection_task = None
frame_lock = threading.Lock()  # Thread-safe frame access

# Performance tracking
performance_metrics = {
    'total_detections': 0,
    'total_webhooks_sent': 0,
    'avg_detection_time_ms': 0,
    'frames_streamed': 0
}

# Track last webhook send times to avoid spamming n8n
_last_webhook_times = {}


# ============================================================================
# LIFESPAN MANAGEMENT
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    global detector, fire_detector, crowd_detector, density_calculator
    global anomaly_detector, crowd_analyzer, context_analyzer
    global decision_intelligence, video_capture, detection_task
    
    print("\n" + "="*70)
    print("🚀 PROJECT DRISHTI - COMPLETE SYSTEM STARTUP (OPTIMIZED)")
    print("="*70)
    
    # ===== DETECTION LAYER =====
    print("\n[1/5] 🔍 Initializing Detection Layer...")
    
    detector = DrishtiDetector(
        model_path=settings.YOLO_MODEL_PATH,
        confidence=settings.DETECTION_CONFIDENCE,
        crowd_threshold_warning=settings.CROWD_THRESHOLD_WARNING,
        crowd_threshold_critical=settings.CROWD_THRESHOLD_CRITICAL
    )
    
    fire_detector = AdvancedFireDetector(mode='hybrid')
    crowd_detector = EnhancedCrowdDetector(enable_tracking=True)
    density_calculator = IntelligentDensityCalculator(mode='uncalibrated', venue_type='general')
    anomaly_detector = CrowdAnomalyDetector()
    crowd_analyzer = CrowdAnalyzer()
    
    print("   ✅ Detection layer ready")
    
    # ===== INTELLIGENCE LAYER =====
    print("\n[2/5] 🧠 Initializing Intelligence Layer...")
    
    context_analyzer = ContextAnalyzer()
    
    try:
        decision_intelligence = DecisionIntelligence(gemini_api_key=settings.GEMINI_API_KEY)
        print("   ✅ Decision intelligence ready (with Gemini)")
    except Exception as e:
        decision_intelligence = DecisionIntelligence(gemini_api_key=None)
        print(f"   ⚠️  Decision intelligence ready (without Gemini): {e}")
    
    print(f"   🤖 n8n Webhook Base: {settings.N8N_WEBHOOK_BASE_URL}")
    
    # ===== VIDEO SOURCE =====
    print("\n[3/5] 📹 Initializing Video Source...")
    
    video_source = settings.VIDEO_SOURCE
    if video_source.isdigit():
        video_source = int(video_source)
    
    video_capture = cv2.VideoCapture(video_source)
    
    # Optimize video capture settings
    if video_capture.isOpened():
        video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer
        video_capture.set(cv2.CAP_PROP_FPS, 30)
        print(f"   ✅ Video source opened: {video_source}")
    else:
        print(f"   ❌ Failed to open video source: {video_source}")
    
    # ===== BACKGROUND TASKS =====
    print("\n[4/5] ⚙️  Starting Background Tasks...")
    
    detection_task = asyncio.create_task(intelligent_detection_loop())
    print("   ✅ Intelligent detection loop started")
    
    # ===== SYSTEM CHECK =====
    print("\n[5/5] ✅ System Health Check...")
    print(f"   • Stream Quality: {STREAM_QUALITY}%")
    print(f"   • Stream FPS: {STREAM_FPS}")
    print(f"   • Frame Skip: {DETECTION_FRAME_SKIP}")
    print(f"   • Max Width: {MAX_STREAM_WIDTH}px")
    
    health = {
        'Detection': detector is not None,
        'Fire Detection': fire_detector is not None,
        'Crowd Analysis': crowd_detector is not None,
        'Intelligence': decision_intelligence is not None,
        'n8n Webhooks': True,
        'Video': video_capture.isOpened() if video_capture else False
    }
    
    for component, status in health.items():
        symbol = "✅" if status else "❌"
        print(f"   {symbol} {component}")
    
    print("\n" + "="*70)
    print("✨ PROJECT DRISHTI IS FULLY OPERATIONAL!")
    print(f"📡 API available at: http://localhost:{settings.PORT}")
    print(f"📊 Dashboard: http://localhost:{settings.PORT}/video-feed")
    print("="*70 + "\n")
    
    # ===== RUN =====
    yield
    
    # ===== SHUTDOWN =====
    print("\n👋 Project Drishti shutting down...")
    
    if detection_task and not detection_task.done():
        detection_task.cancel()
        try:
            await asyncio.wait_for(detection_task, timeout=2.0)
        except:
            pass
    
    if video_capture and video_capture.isOpened():
        video_capture.release()
        print("📹 Video capture released")
    
    print("✅ Shutdown complete\n")


# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(
    title="Project Drishti API",
    description="Complete AI-powered crowd safety system with intelligent agents",
    version="2.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# INTELLIGENT DETECTION LOOP (OPTIMIZED)
# ============================================================================

def _run_detection_sync(frame, crowd_detector, fire_detector, density_calculator,
                        anomaly_detector, crowd_analyzer, context_analyzer,
                        decision_intelligence):
    """
    Run ALL heavy detection/analysis work in a single synchronous function.
    This is executed in a thread pool to avoid blocking the async event loop.
    """
    loop_start = time.time()

    # ===== STEP 1: DETECTION =====
    crowd_result = crowd_detector.detect(frame, conf_threshold=0.35)
    fire_result = fire_detector.detect(frame)

    density_metrics = density_calculator.calculate(
        crowd_result['person_count'],
        frame_shape=frame.shape
    )

    anomaly_result = anomaly_detector.detect(
        person_count=crowd_result['person_count'],
        zones=crowd_result['zones'],
        detections=crowd_result['detections'],
        timestamp=time.time()
    )

    analysis = crowd_analyzer.update(
        person_count=crowd_result['person_count'],
        fire_detected=fire_result.detected,
        timestamp=time.time(),
        density_level=density_metrics.level,
        zones=crowd_result['zones']
    )

    # ===== STEP 2: CONTEXT BUILDING =====
    context = context_analyzer.build_context(
        detection_result={'frame': frame},
        crowd_analysis={
            'person_count': crowd_result['person_count'],
            'zones': crowd_result['zones'],
            'is_dense': crowd_result['is_dense'],
            'detection_time_ms': crowd_result['detection_time_ms'],
            'trend': anomaly_detector.get_trend(),
            'rate_of_change': 0,
            'predicted_count': crowd_result['person_count'],
            'risk_score': analysis.risk_score
        },
        density_metrics=density_metrics,
        anomaly_detection=anomaly_result,
        fire_detection=fire_result
    )

    # ===== STEP 3: RULE-BASED GUIDANCE =====
    decision = decision_intelligence.make_decision(context)

    # ===== STEP 4: DETERMINE WHICH N8N WEBHOOKS TO TRIGGER =====
    webhooks_to_fire = []

    if fire_result.detected:
        webhooks_to_fire.append(('fire-alert', {
            'event': 'fire_detected',
            'confidence': float(fire_result.confidence),
            'person_count': crowd_result['person_count'],
            'locations': [list(b) for b in fire_result.bounding_boxes],
            'severity': 'CRITICAL',
            'timestamp': time.time()
        }))

    if density_metrics.level in ('CRITICAL', 'VERY_HIGH'):
        webhooks_to_fire.append(('crowd-alert', {
            'event': 'high_density',
            'density_level': density_metrics.level,
            'person_count': crowd_result['person_count'],
            'zones': crowd_result['zones'],
            'risk_score': float(analysis.risk_score),
            'timestamp': time.time()
        }))

    if anomaly_result.detected:
        webhooks_to_fire.append(('anomaly-alert', {
            'event': 'anomaly_detected',
            'anomaly_type': str(anomaly_result.anomaly_type),
            'severity': str(anomaly_result.severity),
            'affected_zones': anomaly_result.affected_zones,
            'person_count': crowd_result['person_count'],
            'timestamp': time.time()
        }))

    if analysis.risk_score > 80:
        webhooks_to_fire.append(('security-alert', {
            'event': 'high_risk',
            'risk_score': float(analysis.risk_score),
            'density_level': density_metrics.level,
            'person_count': crowd_result['person_count'],
            'recommendation': str(analysis.recommendation),
            'timestamp': time.time()
        }))

    # ===== STEP 5: ANNOTATE FRAME =====
    annotated_frame = _annotate_frame_complete(
        frame, crowd_result, fire_result,
        density_metrics, anomaly_result, analysis
    )

    loop_time = (time.time() - loop_start) * 1000

    return {
        'crowd_result': crowd_result,
        'fire_result': fire_result,
        'density_metrics': density_metrics,
        'anomaly_result': anomaly_result,
        'analysis': analysis,
        'context': context,
        'decision': decision,
        'webhooks_to_fire': webhooks_to_fire,
        'annotated_frame': annotated_frame,
        'loop_time': loop_time
    }


async def intelligent_detection_loop():
    """
    Main intelligent detection loop - OPTIMIZED VERSION
    Heavy CV/YOLO work is offloaded to a thread pool via asyncio.to_thread()
    so the FastAPI event loop stays responsive for video streaming and API calls.
    """
    global current_state, video_capture
    global detector, fire_detector, crowd_detector, density_calculator
    global anomaly_detector, crowd_analyzer, context_analyzer
    global decision_intelligence
    global performance_metrics, _last_webhook_times

    print("[INTELLIGENT LOOP] Started (Optimized - threaded)")

    frame_count = 0
    last_detection_time = 0
    min_detection_interval = 1.0 / 10  # Max 10 detections per second
    WEBHOOK_COOLDOWN = 10  # seconds between same webhook type

    try:
        while True:
            try:
                if not video_capture or not video_capture.isOpened():
                    await asyncio.sleep(1)
                    continue

                # Read frame (fast, minimal blocking)
                ret, frame = await asyncio.to_thread(video_capture.read)
                if not ret:
                    video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    await asyncio.sleep(0.1)
                    continue

                frame_count += 1
                current_time = time.time()

                # Skip frames for performance
                if frame_count % DETECTION_FRAME_SKIP != 0:
                    with frame_lock:
                        current_state['frame'] = frame.copy()
                    await asyncio.sleep(0.01)
                    continue

                # Rate limit detections
                if current_time - last_detection_time < min_detection_interval:
                    await asyncio.sleep(0.01)
                    continue

                last_detection_time = current_time

                # ===== OFFLOAD ALL HEAVY WORK TO THREAD =====
                result = await asyncio.to_thread(
                    _run_detection_sync, frame,
                    crowd_detector, fire_detector, density_calculator,
                    anomaly_detector, crowd_analyzer, context_analyzer,
                    decision_intelligence
                )

                crowd_result = result['crowd_result']
                fire_result = result['fire_result']
                density_metrics = result['density_metrics']
                anomaly_result = result['anomaly_result']
                analysis = result['analysis']
                context = result['context']
                decision = result['decision']
                webhooks_to_fire = result['webhooks_to_fire']
                annotated_frame = result['annotated_frame']
                loop_time = result['loop_time']

                # ===== SEND WEBHOOKS TO N8N (with cooldown) =====
                webhooks_sent = 0
                for webhook_path, payload in webhooks_to_fire:
                    last_sent = _last_webhook_times.get(webhook_path, 0)
                    if current_time - last_sent >= WEBHOOK_COOLDOWN:
                        url = f"{settings.N8N_WEBHOOK_BASE_URL}/{webhook_path}"
                        asyncio.create_task(trigger_webhook(url, payload))
                        _last_webhook_times[webhook_path] = current_time
                        webhooks_sent += 1
                        performance_metrics['total_webhooks_sent'] += 1

                # ===== UPDATE STATE (Thread-safe) =====
                with frame_lock:
                    current_state = {
                        'person_count': int(crowd_result['person_count']),
                        'density_level': str(density_metrics.level),
                        'density_value': float(density_metrics.density_value),
                        'risk_score': float(analysis.risk_score),
                        'fire_detected': bool(fire_result.detected),
                        'fire_confidence': float(fire_result.confidence),
                        'trend': anomaly_detector.get_trend(),
                        'rate_of_change': 0.0,
                        'predicted_count_1min': int(crowd_result['person_count']),
                        'anomaly_detected': bool(anomaly_result.detected),
                        'anomaly_type': str(anomaly_result.anomaly_type) if anomaly_result.anomaly_type else None,
                        'anomaly_severity': str(anomaly_result.severity),
                        'zones': {k: int(v) for k, v in crowd_result['zones'].items()},
                        'webhooks_sent': webhooks_sent,
                        'strategic_guidance': str(decision['strategic_guidance']),
                        'decision_method': str(decision['method']),
                        'situation_severity': str(context['situation_severity']),
                        'recommendation': str(analysis.recommendation),
                        'detection_time_ms': float(crowd_result['detection_time_ms']),
                        'decision_time_ms': float(decision.get('decision_time_ms', 0)),
                        'total_loop_time_ms': float(loop_time),
                        'frame': annotated_frame,
                        'timestamp': time.time()
                    }

                # Update metrics
                performance_metrics['total_detections'] += 1

                # Log status (less frequently)
                if frame_count % 30 == 0:
                    print(f"\r[LIVE] People: {crowd_result['person_count']:3d} | "
                          f"Density: {density_metrics.level:12s} | "
                          f"Risk: {analysis.risk_score:5.1f} | "
                          f"Fire: {'YES' if fire_result.detected else 'NO ':3s} | "
                          f"Time: {loop_time:5.0f}ms",
                          end="")

                await asyncio.sleep(0.01)

            except asyncio.CancelledError:
                print("\n[INTELLIGENT LOOP] Shutdown signal received")
                raise
            except Exception as e:
                print(f"\n❌ Error in detection loop: {e}")
                import traceback
                traceback.print_exc()
                await asyncio.sleep(1)

    except asyncio.CancelledError:
        print("[INTELLIGENT LOOP] Stopping gracefully...")
    finally:
        print("[INTELLIGENT LOOP] Stopped")


def _annotate_frame_complete(frame, crowd_result, fire_result, density_metrics, 
                             anomaly_result, analysis):
    """Annotate frame with all detection information"""
    
    annotated = frame.copy()
    h, w = annotated.shape[:2]
    
    # Color scheme based on density
    density_colors = {
        "EMPTY": (128, 128, 128),
        "VERY_LOW": (0, 255, 0),
        "LOW": (0, 255, 0),
        "MODERATE": (0, 255, 255),
        "HIGH": (0, 165, 255),
        "VERY_HIGH": (0, 100, 255),
        "CRITICAL": (0, 0, 255)
    }
    
    color = density_colors.get(density_metrics.level, (255, 255, 255))
    
    # Draw person bounding boxes
    for det in crowd_result['detections']:
        bbox = det['bbox']
        x1, y1, x2, y2 = bbox
        
        # Draw box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        
        # Draw ID if tracked
        if 'id' in det:
            cv2.putText(annotated, f"ID:{det['id']}", (x1, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    # Draw fire bounding boxes
    if fire_result.detected:
        for bbox in fire_result.bounding_boxes:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(annotated, "FIRE!", (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Status bar
    bar_height = 100
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, annotated, 0.3, 0, annotated)
    
    # Row 1
    cv2.putText(annotated, f"PEOPLE: {crowd_result['person_count']}", (10, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    cv2.putText(annotated, f"DENSITY: {density_metrics.level}", (200, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    cv2.putText(annotated, f"RISK: {analysis.risk_score:.0f}/100", (450, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Row 2
    if fire_result.detected:
        cv2.putText(annotated, f"FIRE: DETECTED ({fire_result.confidence:.0%})", 
                   (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.rectangle(annotated, (0, 0), (w-1, h-1), (0, 0, 255), 8)
    else:
        cv2.putText(annotated, "FIRE: CLEAR", (10, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Anomaly indicator
    if anomaly_result.detected:
        cv2.putText(annotated, f"ANOMALY: {anomaly_result.anomaly_type}", 
                   (200, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    
    # Timestamp
    timestamp = time.strftime("%H:%M:%S")
    cv2.putText(annotated, timestamp, (w-100, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return annotated


async def trigger_webhook(url: str, payload: Dict[str, Any]) -> Optional[Dict]:
    """Send webhook to n8n and return the response data"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, timeout=10.0)
            if response.status_code == 200:
                print(f"\n✅ Webhook sent: {url}")
                try:
                    return response.json()
                except Exception:
                    return {"response": response.text}
            else:
                print(f"\n⚠️  Webhook failed: {url} - Status {response.status_code}")
                return None
    except httpx.ConnectError:
        print(f"\n❌ Webhook connection refused: {url} - Is n8n running?")
        return None
    except httpx.TimeoutException:
        print(f"\n❌ Webhook timeout: {url}")
        return None
    except Exception as e:
        print(f"\n❌ Webhook error: {url} - {e}")
        return None


async def query_n8n_agent(question: str, context: Dict[str, Any]) -> Optional[str]:
    """Send a question to the n8n AI agent webhook and return the response"""
    webhook_url = f"{settings.N8N_WEBHOOK_BASE_URL}/chat"
    payload = {
        "question": question,
        "context": {k: v for k, v in context.items() if k != 'frame'},
        "timestamp": time.time()
    }
    
    result = await trigger_webhook(webhook_url, payload)
    
    if result:
        # n8n may return the answer in different fields depending on your workflow
        answer = (
            result.get('output') or
            result.get('answer') or
            result.get('response') or
            result.get('text') or
            result.get('message') or
            str(result)
        )
        return answer
    return None


# ============================================================================
# OPTIMIZED VIDEO STREAMING
# ============================================================================

def generate_frames():
    """OPTIMIZED generator for video streaming"""
    global current_state, performance_metrics
    
    frame_interval = 1.0 / STREAM_FPS
    last_frame_time = 0
    
    while True:
        try:
            current_time = time.time()
            
            # Rate limit frames
            if current_time - last_frame_time < frame_interval:
                time.sleep(0.01)
                continue
            
            with frame_lock:
                if "frame" not in current_state or current_state["frame"] is None:
                    time.sleep(0.05)
                    continue
                
                frame = current_state["frame"].copy()
            
            # Resize for faster streaming
            if ENABLE_FRAME_RESIZE:
                h, w = frame.shape[:2]
                if w > MAX_STREAM_WIDTH:
                    scale = MAX_STREAM_WIDTH / w
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
            # Encode with optimized settings
            encode_params = [
                cv2.IMWRITE_JPEG_QUALITY, STREAM_QUALITY,
                cv2.IMWRITE_JPEG_OPTIMIZE, 1
            ]
            
            ret, buffer = cv2.imencode('.jpg', frame, encode_params)
            
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
                performance_metrics['frames_streamed'] += 1
                last_frame_time = current_time
            
        except Exception as e:
            print(f"Stream error: {e}")
            time.sleep(0.1)


def generate_frames_fast():
    """Ultra-fast frame generator - direct from camera"""
    global video_capture
    
    frame_interval = 1.0 / 20  # 20 FPS
    last_frame_time = 0
    
    while True:
        try:
            current_time = time.time()
            
            if current_time - last_frame_time < frame_interval:
                time.sleep(0.01)
                continue
            
            if video_capture and video_capture.isOpened():
                ret, frame = video_capture.read()
                
                if ret:
                    # Resize to small size
                    frame_small = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_LINEAR)
                    
                    # Fast encode
                    ret, buffer = cv2.imencode('.jpg', frame_small, 
                                               [cv2.IMWRITE_JPEG_QUALITY, 50])
                    
                    if ret:
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + 
                               buffer.tobytes() + b'\r\n')
                        
                        last_frame_time = current_time
            
            time.sleep(0.01)
            
        except Exception as e:
            time.sleep(0.1)


# ============================================================================
# API ENDPOINTS
# ============================================================================

# Pydantic models
class QueryRequest(BaseModel):
    question: str

class VideoSourceRequest(BaseModel):
    type: str  # 'webcam' or 'file'
    path: Optional[str] = None


@app.get("/")
async def root():
    """Health check and system info"""
    return {
        "status": "online",
        "service": "Project Drishti",
        "version": "3.0.0",
        "components": {
            "detection": detector is not None,
            "fire_detection": fire_detector is not None,
            "crowd_analysis": crowd_detector is not None,
            "intelligence": decision_intelligence is not None,
            "n8n_webhooks": True
        }
    }


@app.get("/status")
async def get_status():
    """Get current system status"""
    with frame_lock:
        if not current_state:
            raise HTTPException(status_code=503, detail="System initializing...")
        
        return {
            "person_count": current_state.get("person_count", 0),
            "density_level": current_state.get("density_level", "UNKNOWN"),
            "density_value": current_state.get("density_value", 0),
            "trend": current_state.get("trend", "STABLE"),
            "risk_score": current_state.get("risk_score", 0),
            "fire_detected": current_state.get("fire_detected", False),
            "fire_confidence": current_state.get("fire_confidence", 0),
            "anomaly_detected": current_state.get("anomaly_detected", False),
            "anomaly_type": current_state.get("anomaly_type"),
            "anomaly_severity": current_state.get("anomaly_severity"),
            "situation_severity": current_state.get("situation_severity", "UNKNOWN"),
            "recommendation": current_state.get("recommendation", "Initializing..."),
            "webhooks_sent": current_state.get("webhooks_sent", 0),
            "strategic_guidance": current_state.get("strategic_guidance", ""),
            "zones": current_state.get("zones", {}),
            "detection_time_ms": current_state.get("detection_time_ms", 0),
            "decision_time_ms": current_state.get("decision_time_ms", 0),
            "total_loop_time_ms": current_state.get("total_loop_time_ms", 0)
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
    with frame_lock:
        if not current_state:
            raise HTTPException(status_code=503, detail="No data available")
        
        # Return all except frame
        return {k: v for k, v in current_state.items() if k != 'frame'}


@app.get("/n8n-status")
async def get_n8n_status():
    """Check n8n webhook connectivity"""
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
                'total_webhooks_sent': performance_metrics['total_webhooks_sent']
            }
    except Exception as e:
        return {
            'n8n_reachable': False,
            'error': str(e),
            'webhook_base': settings.N8N_WEBHOOK_BASE_URL,
            'total_webhooks_sent': performance_metrics['total_webhooks_sent']
        }


@app.get("/intelligence-stats")
async def get_intelligence_stats():
    """Get decision intelligence statistics"""
    if not decision_intelligence:
        raise HTTPException(status_code=503, detail="Intelligence system not initialized")
    
    return decision_intelligence.get_stats()


@app.get("/performance")
async def get_performance():
    """Get system performance metrics"""
    return performance_metrics


@app.post("/query")
async def query(request: QueryRequest):
    """Natural language query - tries n8n AI agent first, then local fallbacks"""
    with frame_lock:
        if not current_state:
            raise HTTPException(status_code=503, detail="System initializing...")
        state_copy = {k: v for k, v in current_state.items() if k != 'frame'}
    
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
            if decision_intelligence and decision_intelligence.gemini:
                answer = decision_intelligence.gemini.answer_query(request.question, state_copy)
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


def _fallback_query_handler(question: str, state: Dict[str, Any]) -> str:
    """Fallback handler when AI is unavailable"""
    q = question.lower()
    
    if 'count' in q or 'people' in q or 'crowd' in q:
        return f"There are currently {state.get('person_count', 0)} people detected in the venue."
    
    if 'fire' in q or 'smoke' in q:
        if state.get('fire_detected'):
            return f"WARNING: Fire detected! Confidence: {state.get('fire_confidence', 0):.1%}. Please evacuate immediately."
        return "No fire detected at this time."
    
    if 'risk' in q or 'danger' in q or 'safety' in q:
        score = state.get('risk_score', 0)
        return f"Current risk score is {score:.1f}/100. Situation is {state.get('situation_severity', 'stable')}."
    
    if 'status' in q or 'situation' in q:
        return f"System is effective. Crowd density is {state.get('density_level', 'NORMAL')} with {state.get('person_count', 0)} people."
        
    if 'agent' in q:
        return f"Active agents: {state.get('agents_activated', 0)}. Actions executed: {state.get('actions_executed', 0)}."
        
    return "I am currently running in fallback mode (AI unavailable). I can answer basic questions about crowd count, fire status, and risk levels."


@app.get("/summary")
async def get_summary():
    """Get AI-generated situation summary"""
    if not decision_intelligence or not decision_intelligence.gemini:
        raise HTTPException(status_code=503, detail="AI summary not available")
    
    with frame_lock:
        if not current_state:
            raise HTTPException(status_code=503, detail="No data available")
        
        summary = decision_intelligence.gemini.generate_situation_summary(
            person_count=current_state.get('person_count', 0),
            density_level=current_state.get('density_level', 'UNKNOWN'),
            trend=current_state.get('trend', 'STABLE'),
            rate_of_change=current_state.get('rate_of_change', 0),
            predicted_count=current_state.get('predicted_count_1min', 0),
            risk_score=current_state.get('risk_score', 0),
            anomaly_type=current_state.get('anomaly_type'),
            zones=current_state.get('zones', {})
        )
    
    return {"summary": summary, "timestamp": time.time()}


@app.get("/list-videos")
async def list_videos():
    """List available videos"""
    data_path = Path("../data")
    data_path.mkdir(exist_ok=True)
    
    videos = []
    for file in data_path.glob("*"):
        if file.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
            videos.append({
                "name": file.name,
                "path": str(file),
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
    global video_capture
    
    try:
        # Release current
        if video_capture:
            video_capture.release()
            await asyncio.sleep(0.5)
        
        # Open new source
        if request.type == "webcam":
            video_capture = cv2.VideoCapture(0)
            source_name = "Webcam"
        else:
            video_capture = cv2.VideoCapture(request.path)
            source_name = Path(request.path).name
        
        if not video_capture.isOpened():
            raise HTTPException(status_code=400, detail="Cannot open video source")
        
        # Optimize settings
        video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        return {
            "success": True,
            "source": source_name,
            "type": request.type
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=False,
        log_level="info"
    )