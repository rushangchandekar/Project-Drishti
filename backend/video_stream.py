import time
import cv2
from backend import state

def generate_frames():
    """OPTIMIZED generator for video streaming"""
    frame_interval = 1.0 / state.STREAM_FPS
    last_frame_time = 0
    
    while True:
        try:
            current_time = time.time()
            
            # Rate limit frames
            if current_time - last_frame_time < frame_interval:
                time.sleep(0.01)
                continue
            
            with state.frame_lock:
                if "frame" not in state.current_state or state.current_state["frame"] is None:
                    time.sleep(0.05)
                    continue
                
                frame = state.current_state["frame"].copy()
            
            # Resize for faster streaming
            if state.ENABLE_FRAME_RESIZE:
                h, w = frame.shape[:2]
                if w > state.MAX_STREAM_WIDTH:
                    scale = state.MAX_STREAM_WIDTH / w
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
            # Encode with optimized settings
            encode_params = [
                cv2.IMWRITE_JPEG_QUALITY, state.STREAM_QUALITY,
                cv2.IMWRITE_JPEG_OPTIMIZE, 1
            ]
            
            ret, buffer = cv2.imencode('.jpg', frame, encode_params)
            
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
                state.performance_metrics['frames_streamed'] += 1
                last_frame_time = current_time
            
        except Exception as e:
            print(f"Stream error: {e}")
            time.sleep(0.1)


def generate_frames_fast():
    """Ultra-fast frame generator - direct from camera"""
    frame_interval = 1.0 / 20  # 20 FPS
    last_frame_time = 0
    
    while True:
        try:
            current_time = time.time()
            
            if current_time - last_frame_time < frame_interval:
                time.sleep(0.01)
                continue
            
            if state.video_capture and state.video_capture.isOpened():
                ret, frame = state.video_capture.read()
                
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
            
        except Exception:
            time.sleep(0.1)
