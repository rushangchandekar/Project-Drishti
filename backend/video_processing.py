import time
import asyncio
import cv2
import traceback
from typing import Dict, Any

from backend.config import get_settings
from backend import state
from backend.webhooks import trigger_webhook, invoke_agent_webhook
try:
    from backend.intelligence.agent_orchestrator import AGENT_REGISTRY
except ImportError:
    from intelligence.agent_orchestrator import AGENT_REGISTRY

from backend.twilio_service import send_emergency_whatsapp

settings = get_settings()

def _annotate_frame_complete(frame, crowd_result, fire_result, density_metrics, 
                             anomaly_result, analysis, activity_result=None):
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
    
    # Activity recognition overlay
    if activity_result and activity_result.activities:
        y_act = 85
        mood_colors = {
            'CALM': (0, 255, 0), 'ALERT': (0, 255, 255),
            'TENSE': (0, 165, 255), 'CHAOTIC': (0, 0, 255)
        }
        mood_color = mood_colors.get(activity_result.scene_mood, (255, 255, 255))
        cv2.putText(annotated, f"MOOD: {activity_result.scene_mood}",
                   (10, y_act), cv2.FONT_HERSHEY_SIMPLEX, 0.5, mood_color, 2)
        
        for act in activity_result.activities[:2]:  # Show top 2 activities
            act_color = (0, 0, 255) if act.severity in ('HIGH', 'CRITICAL') else (0, 255, 255)
            cv2.putText(annotated, f"ACTIVITY: {act.activity_type} ({act.confidence:.0%})",
                       (200, y_act), cv2.FONT_HERSHEY_SIMPLEX, 0.45, act_color, 2)
            y_act += 18
            
            # Draw circle at activity location
            if act.location:
                center_pt = (int(act.location[0]), int(act.location[1]))
                cv2.circle(annotated, center_pt, 30, act_color, 2)
    
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

    # ===== STEP 1.5: ACTIVITY RECOGNITION =====
    activity_result = None
    if state.activity_recognizer:
        activity_result = state.activity_recognizer.detect(
            detections=crowd_result['detections'],
            frame_shape=frame.shape[:2]
        )

    analysis = state.crowd_analyzer.update(
        person_count=crowd_result['person_count'],
        fire_detected=fire_result.detected,
        timestamp=time.time(),
        density_level=density_metrics.level,
        zones=crowd_result['zones'],
        dominant_activity=activity_result.dominant_activity if activity_result else None
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
        fire_detection=fire_result,
        activity_detection=activity_result
    )

    # ===== STEP 3: RULE-BASED GUIDANCE =====
    decision = state.decision_intelligence.make_decision(context)

    # ===== STEP 4: AGENT ORCHESTRATOR — RULE-BASED SELECTION =====
    selected_agents = {}
    autonomous_actions = []

    if state.agent_orchestrator:
        selected_agents = state.agent_orchestrator.rule_based_selection(context)

    # Derive autonomous actions from selected agents
    if 'FireAgent' in selected_agents:
        autonomous_actions.extend(["Activating Fire Sprinklers", "Contacting Local Fire Station", "Sent Emergency WhatsApp Admin Alert", "SOS Alarm Raised"])
    if 'EvacAgent' in selected_agents:
        autonomous_actions.extend(["Opening Emergency Exit Doors", "Activating PA System"])
    if 'CrowdAgent' in selected_agents and density_metrics.level == 'CRITICAL':
        autonomous_actions.extend(["Dispatching Crowd Control Security", "SOS Alarm Raised"])
    if 'MedicAgent' in selected_agents:
        autonomous_actions.extend(["Medical Team on Standby", "Ambulance Notified"])
    if 'DispatchAgent' in selected_agents:
        autonomous_actions.extend(["Dispatching Security Personnel"])

    # Activity-based agent triggers
    if activity_result and activity_result.activities:
        for act in activity_result.activities:
            if act.activity_type == 'FALL' and 'MedicAgent' not in selected_agents:
                selected_agents['MedicAgent'] = f"Fall detected: {act.description}"
                autonomous_actions.extend(["Medical Team on Standby", "Ambulance Notified"])
            elif act.activity_type == 'FIGHT' and 'DispatchAgent' not in selected_agents:
                selected_agents['DispatchAgent'] = f"Fight detected: {act.description}"
                autonomous_actions.extend(["Dispatching Security Personnel"])
            elif act.activity_type in ('PANIC', 'STAMPEDE') and 'EvacAgent' not in selected_agents:
                selected_agents['EvacAgent'] = f"{act.activity_type} detected: {act.description}"
                autonomous_actions.extend(["Opening Emergency Exit Doors", "Activating PA System"])
            elif act.activity_type == 'GATHERING' and act.severity in ('HIGH', 'CRITICAL') and 'CrowdAgent' not in selected_agents:
                selected_agents['CrowdAgent'] = f"Large gathering detected: {act.description}"

    autonomous_actions = list(set(autonomous_actions))

    # ===== STEP 5: ANNOTATE FRAME =====
    annotated_frame = _annotate_frame_complete(
        frame, crowd_result, fire_result,
        density_metrics, anomaly_result, analysis, activity_result
    )

    loop_time = (time.time() - loop_start) * 1000

    return {
        'crowd_result': crowd_result,
        'fire_result': fire_result,
        'density_metrics': density_metrics,
        'anomaly_result': anomaly_result,
        'activity_result': activity_result,
        'analysis': analysis,
        'context': context,
        'decision': decision,
        'selected_agents': selected_agents,
        'annotated_frame': annotated_frame,
        'loop_time': loop_time,
        'autonomous_actions': autonomous_actions,
        'raw_frame': frame,  # Keep raw frame for Gemini Vision
    }


async def intelligent_detection_loop():
    """
    Main intelligent detection loop - OPTIMIZED VERSION
    Heavy CV/YOLO work is offloaded to a thread pool via asyncio.to_thread()
    so the FastAPI event loop stays responsive for video streaming and API calls.
    """
    print("[INTELLIGENT LOOP] Started (Optimized - threaded)")

    # 1. Initialize heavy CV/AI components in background thread to keep startup instant
    def init_heavy_components():
        try:
            from detection.enhanced_crowd_detector import EnhancedCrowdDetector
            from detection.fire_detector import AdvancedFireDetector
            from detection.density_calculator import IntelligentDensityCalculator
            from detection.anomaly_detector import CrowdAnomalyDetector
            from detection.crowd_analyzer import CrowdAnalyzer
            from detection.activity_recognizer import ActivityRecognizer
            
            print("\n[BACKGROUND INIT] Initializing heavy components...")
            
            if not state.crowd_detector:
                print("   [BACKGROUND INIT] Loading YOLOv11 model for EnhancedCrowdDetector...")
                state.crowd_detector = EnhancedCrowdDetector(
                    enable_tracking=True,
                    input_size=settings.YOLO_INPUT_SIZE,
                    half_precision=settings.YOLO_HALF_PRECISION,
                    max_detections=settings.YOLO_MAX_DETECTIONS,
                    nms_iou=settings.YOLO_NMS_IOU
                )
            if not state.fire_detector:
                state.fire_detector = AdvancedFireDetector(mode='hybrid')
            if not state.density_calculator:
                state.density_calculator = IntelligentDensityCalculator(mode='uncalibrated', venue_type='general')
            if not state.anomaly_detector:
                state.anomaly_detector = CrowdAnomalyDetector()
            if not state.crowd_analyzer:
                state.crowd_analyzer = CrowdAnalyzer()
            if not state.activity_recognizer:
                print("   [BACKGROUND INIT] Initializing ActivityRecognizer...")
                state.activity_recognizer = ActivityRecognizer()
            if not state.agent_orchestrator:
                from intelligence.agent_orchestrator import AgentOrchestrator, _build_initial_agent_statuses
                print("   [BACKGROUND INIT] Initializing AgentOrchestrator...")
                gemini_client = None
                if state.decision_intelligence and state.decision_intelligence.gemini:
                    gemini_client = state.decision_intelligence.gemini.client
                state.agent_orchestrator = AgentOrchestrator(gemini_client=gemini_client)
                state.agent_statuses = _build_initial_agent_statuses()
                
            # Initialize video source if not opened
            if not state.video_capture:
                video_source = settings.VIDEO_SOURCE
                if video_source.isdigit():
                    video_source = int(video_source)
                print(f"   [BACKGROUND INIT] Opening video source: {video_source}")
                state.video_capture = cv2.VideoCapture(video_source)
                if state.video_capture.isOpened():
                    state.video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    state.video_capture.set(cv2.CAP_PROP_FPS, 30)
                    print(f"   [BACKGROUND INIT] Video source opened successfully: {video_source}")
                else:
                    print(f"   [BACKGROUND INIT] Warning: Could not open video source: {video_source}")
            return True
        except Exception as e:
            print(f"   [BACKGROUND INIT] Error during heavy component initialization: {e}")
            traceback.print_exc()
            return False

    print("   [BACKGROUND INIT] Initializing YOLO and OpenCV in background thread...")
    init_success = False
    while not init_success:
        init_success = await asyncio.to_thread(init_heavy_components)
        if not init_success:
            print("   [BACKGROUND INIT] Initialization failed. Retrying in 2 seconds...")
            await asyncio.sleep(2)
            
    print("   [BACKGROUND INIT] All heavy CV and AI components initialized successfully!\n")

    frame_count = 0
    last_detection_time = 0
    last_twilio_alert_time = 0
    last_detect_cache = None
    min_detection_interval = 1.0 / 10  # Max 10 detections per second
    WEBHOOK_COOLDOWN = 10  # seconds between same webhook type
    TWILIO_COOLDOWN = 60 # 1 minute cooldown for WhatsApp messages

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

                # Determine if we should skip heavy detection for this frame
                skip_heavy = False
                if frame_count % state.DETECTION_FRAME_SKIP != 0:
                    skip_heavy = True
                elif current_time - last_detection_time < min_detection_interval:
                    skip_heavy = True

                if skip_heavy:
                    if last_detect_cache:
                        # Re-annotate the fresh frame with the cached bounding boxes to prevent blinking
                        annotated = _annotate_frame_complete(
                            frame, 
                            last_detect_cache['crowd_result'], 
                            last_detect_cache['fire_result'], 
                            last_detect_cache['density_metrics'], 
                            last_detect_cache['anomaly_result'], 
                            last_detect_cache['analysis'],
                            last_detect_cache.get('activity_result')
                        )
                        with state.frame_lock:
                            state.current_state['frame'] = annotated
                    else:
                        with state.frame_lock:
                            state.current_state['frame'] = frame.copy()
                    
                    await asyncio.sleep(0.01)
                    continue

                last_detection_time = current_time

                # ===== OFFLOAD ALL HEAVY WORK TO THREAD =====
                result = await asyncio.to_thread(_run_detection_sync, frame)
                last_detect_cache = result

                crowd_result = result['crowd_result']
                fire_result = result['fire_result']
                density_metrics = result['density_metrics']
                anomaly_result = result['anomaly_result']
                analysis = result['analysis']
                context = result['context']
                decision = result['decision']
                annotated_frame = result['annotated_frame']
                loop_time = result['loop_time']
                autonomous_actions = result['autonomous_actions']

                # ===== TRIGGER TWILIO WHATSAPP ON CRITICAL =====
                should_trigger_twilio = fire_result.detected or density_metrics.level == 'CRITICAL'
                if should_trigger_twilio and (current_time - last_twilio_alert_time > TWILIO_COOLDOWN):
                    alert_type = 'FIRE EMERGENCY' if fire_result.detected else 'CRITICAL CROWD DENSITY'
                    details = f"Risk Score {analysis.risk_score}/100. People Count: {crowd_result['person_count']}."
                    asyncio.create_task(asyncio.to_thread(send_emergency_whatsapp, alert_type, details))
                    last_twilio_alert_time = current_time

                # ===== AGENT ORCHESTRATOR — INVOKE SELECTED AGENTS =====
                selected_agents = result.get('selected_agents', {})
                raw_frame = result.get('raw_frame')
                webhooks_sent = 0

                if selected_agents and state.agent_orchestrator:
                    # Gemini Vision analysis for critical situations only
                    if (state.agent_orchestrator.is_critical_situation(context)
                            and state.agent_orchestrator.can_call_vision()):
                        ai_agents = await asyncio.to_thread(
                            state.agent_orchestrator.gemini_vision_selection,
                            context, raw_frame
                        )
                        selected_agents = state.agent_orchestrator.merge_selections(
                            selected_agents, ai_agents
                        )
                        method = 'hybrid'
                    else:
                        method = 'rules'

                    # Log orchestration
                    state.agent_orchestrator.log_orchestration(
                        selected_agents, method, context
                    )

                    # Build common payload for agents
                    agent_payload = {
                        'person_count': int(crowd_result['person_count']),
                        'density_level': str(density_metrics.level),
                        'risk_score': float(analysis.risk_score),
                        'fire_detected': bool(fire_result.detected),
                        'fire_confidence': float(fire_result.confidence),
                        'anomaly_detected': bool(anomaly_result.detected),
                        'anomaly_type': str(anomaly_result.anomaly_type) if anomaly_result.anomaly_type else None,
                        'anomaly_severity': str(anomaly_result.severity),
                        'trend': state.anomaly_detector.get_trend(),
                        'zones': {k: int(v) for k, v in crowd_result['zones'].items()},
                        'situation_severity': str(context['situation_severity']),
                        'recommendation': str(analysis.recommendation),
                        'strategic_guidance': str(decision['strategic_guidance']),
                        'timestamp': time.time(),
                    }

                    # Invoke agents concurrently (with per-agent cooldown)
                    agent_tasks = []
                    for agent_id, reason in selected_agents.items():
                        last_sent = state._last_webhook_times.get(agent_id, 0)
                        if current_time - last_sent >= WEBHOOK_COOLDOWN:
                            agent_info = AGENT_REGISTRY.get(agent_id, {})
                            webhook_path = agent_info.get('webhook_path', '')
                            if not webhook_path:
                                continue

                            # Mark agent as running
                            with state.frame_lock:
                                if agent_id in state.agent_statuses:
                                    state.agent_statuses[agent_id]['status'] = 'running'
                                    state.agent_statuses[agent_id]['trigger_reason'] = reason
                                    state.agent_statuses[agent_id]['last_invoked'] = time.strftime('%Y-%m-%dT%H:%M:%S')

                            payload_with_reason = {**agent_payload, 'agent_id': agent_id, 'trigger_reason': reason}
                            agent_tasks.append(invoke_agent_webhook(agent_id, webhook_path, payload_with_reason))
                            state._last_webhook_times[agent_id] = current_time

                    if agent_tasks:
                        agent_results = await asyncio.gather(*agent_tasks, return_exceptions=True)

                        for ar in agent_results:
                            if isinstance(ar, Exception):
                                continue
                            aid = ar.get('agent_id')
                            if not aid or aid not in state.agent_statuses:
                                continue

                            with state.frame_lock:
                                st = state.agent_statuses[aid]
                                if ar.get('success'):
                                    st['status'] = 'completed'
                                    st['last_result'] = ar.get('response')
                                    st['last_error'] = None
                                else:
                                    st['status'] = 'error'
                                    st['last_result'] = None
                                    st['last_error'] = ar.get('error')
                                st['execution_time_ms'] = ar.get('execution_time_ms', 0)
                                st['last_completed'] = time.strftime('%Y-%m-%dT%H:%M:%S')
                                st['invocation_count'] = st.get('invocation_count', 0) + 1

                            # Also log as an agent action for the feed
                            action_record = {
                                'id': int(time.time() * 1000),
                                'agent': AGENT_REGISTRY.get(aid, {}).get('name', aid),
                                'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
                                'status': 'EXECUTED' if ar.get('success') else 'FAILED',
                                'data': ar.get('response') or {'error': ar.get('error')},
                                'trigger_reason': state.agent_statuses.get(aid, {}).get('trigger_reason', ''),
                            }
                            with state.frame_lock:
                                state.recent_agent_actions.insert(0, action_record)
                                if len(state.recent_agent_actions) > 50:
                                    state.recent_agent_actions.pop()

                            webhooks_sent += 1
                            state.performance_metrics['total_webhooks_sent'] += 1

                # ===== UPDATE STATE (Thread-safe) =====
                with state.frame_lock:
                    # Build activity summary for state
                    activity_summary = []
                    activity_mood = 'CALM'
                    dominant_activity = None
                    activity_result = result.get('activity_result')
                    if activity_result and activity_result.activities:
                        activity_mood = activity_result.scene_mood
                        dominant_activity = activity_result.dominant_activity
                        for act in activity_result.activities:
                            activity_summary.append({
                                'type': str(act.activity_type),
                                'severity': str(act.severity),
                                'confidence': float(act.confidence),
                                'description': str(act.description),
                                'involved_ids': [int(x) for x in act.involved_ids],
                                'location': [int(x) for x in act.location] if act.location else None,
                                'zone': str(act.zone)
                            })

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
                        'autonomous_actions': autonomous_actions,
                        'selected_agents': selected_agents,
                        'recent_agent_actions': list(state.recent_agent_actions),
                        'agents_active': len([s for s in state.agent_statuses.values() if s.get('status') in ('running', 'completed')]),
                        'activities': activity_summary,
                        'scene_mood': activity_mood,
                        'dominant_activity': dominant_activity,
                        'timestamp': time.time()
                    }

                    # Broadcast telemetry state to all connected WebSockets
                    if state.active_websockets:
                        state_copy = {k: v for k, v in state.current_state.items() if k != 'frame'}
                        for ws in list(state.active_websockets):
                            try:
                                await ws.send_json(state_copy)
                            except Exception:
                                state.active_websockets.discard(ws)

                # Update metrics
                state.performance_metrics['total_detections'] += 1

                # Log status (less frequently)
                if frame_count % 30 == 0:
                    active_agents = len(selected_agents)
                    act_info = f"Act: {dominant_activity or 'NONE':10s}" if activity_result else "Act: N/A       "
                    print(f"\r[LIVE] People: {crowd_result['person_count']:3d} | "
                          f"Density: {density_metrics.level:12s} | "
                          f"Risk: {analysis.risk_score:5.1f} | "
                          f"Fire: {'YES' if fire_result.detected else 'NO ':3s} | "
                          f"{act_info} | "
                          f"Mood: {activity_mood:7s} | "
                          f"Agents: {active_agents} | "
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
