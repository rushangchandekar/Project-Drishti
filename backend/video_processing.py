import time
import asyncio
import cv2
import traceback
from typing import Dict, Any

from backend.config import get_settings
from backend import state
from backend.webhooks import trigger_webhook

settings = get_settings()

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


def _run_detection_sync(frame):
    """
    Run ALL heavy detection/analysis work in a single synchronous function.
    This is executed in a thread pool to avoid blocking the async event loop.
    """
    loop_start = time.time()

    # ===== STEP 1: DETECTION =====
    crowd_result = state.crowd_detector.detect(frame, conf_threshold=0.35)
    fire_result = state.fire_detector.detect(frame)

    density_metrics = state.density_calculator.calculate(
        crowd_result['person_count'],
        frame_shape=frame.shape
    )

    anomaly_result = state.anomaly_detector.detect(
        person_count=crowd_result['person_count'],
        zones=crowd_result['zones'],
        detections=crowd_result['detections'],
        timestamp=time.time()
    )

    analysis = state.crowd_analyzer.update(
        person_count=crowd_result['person_count'],
        fire_detected=fire_result.detected,
        timestamp=time.time(),
        density_level=density_metrics.level,
        zones=crowd_result['zones']
    )

    # ===== STEP 2: CONTEXT BUILDING =====
    context = state.context_analyzer.build_context(
        detection_result={'frame': frame},
        crowd_analysis={
            'person_count': crowd_result['person_count'],
            'zones': crowd_result['zones'],
            'is_dense': crowd_result['is_dense'],
            'detection_time_ms': crowd_result['detection_time_ms'],
            'trend': state.anomaly_detector.get_trend(),
            'rate_of_change': 0,
            'predicted_count': crowd_result['person_count'],
            'risk_score': analysis.risk_score
        },
        density_metrics=density_metrics,
        anomaly_detection=anomaly_result,
        fire_detection=fire_result
    )

    # ===== STEP 3: RULE-BASED GUIDANCE =====
    decision = state.decision_intelligence.make_decision(context)

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
    print("[INTELLIGENT LOOP] Started (Optimized - threaded)")

    frame_count = 0
    last_detection_time = 0
    min_detection_interval = 1.0 / 10  # Max 10 detections per second
    WEBHOOK_COOLDOWN = 10  # seconds between same webhook type

    try:
        while True:
            try:
                if not state.video_capture or not state.video_capture.isOpened():
                    await asyncio.sleep(1)
                    continue

                # Read frame (fast, minimal blocking)
                ret, frame = await asyncio.to_thread(state.video_capture.read)
                if not ret:
                    state.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    await asyncio.sleep(0.1)
                    continue

                frame_count += 1
                current_time = time.time()

                # Skip frames for performance
                if frame_count % state.DETECTION_FRAME_SKIP != 0:
                    with state.frame_lock:
                        state.current_state['frame'] = frame.copy()
                    await asyncio.sleep(0.01)
                    continue

                # Rate limit detections
                if current_time - last_detection_time < min_detection_interval:
                    await asyncio.sleep(0.01)
                    continue

                last_detection_time = current_time

                # ===== OFFLOAD ALL HEAVY WORK TO THREAD =====
                result = await asyncio.to_thread(_run_detection_sync, frame)

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
                    last_sent = state._last_webhook_times.get(webhook_path, 0)
                    if current_time - last_sent >= WEBHOOK_COOLDOWN:
                        url = f"{settings.N8N_WEBHOOK_BASE_URL}/{webhook_path}"
                        asyncio.create_task(trigger_webhook(url, payload))
                        state._last_webhook_times[webhook_path] = current_time
                        webhooks_sent += 1
                        state.performance_metrics['total_webhooks_sent'] += 1

                # ===== UPDATE STATE (Thread-safe) =====
                with state.frame_lock:
                    state.current_state = {
                        'person_count': int(crowd_result['person_count']),
                        'density_level': str(density_metrics.level),
                        'density_value': float(density_metrics.density_value),
                        'risk_score': float(analysis.risk_score),
                        'fire_detected': bool(fire_result.detected),
                        'fire_confidence': float(fire_result.confidence),
                        'trend': state.anomaly_detector.get_trend(),
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
                state.performance_metrics['total_detections'] += 1

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
                traceback.print_exc()
                await asyncio.sleep(1)

    except asyncio.CancelledError:
        print("[INTELLIGENT LOOP] Stopping gracefully...")
    finally:
        print("[INTELLIGENT LOOP] Stopped")
