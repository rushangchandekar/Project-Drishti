"""
backend/multi_stream.py
Multi-Stream Video Manager for Project Drishti

Manages multiple concurrent camera feeds (webcam, RTSP, file) with:
- Independent capture threads per stream (non-blocking reads)
- Per-stream latest-frame buffer (always fresh, no queue backlog)
- Thread-safe access via per-stream locks
- Graceful add/remove of streams at runtime
"""

import cv2
import time
import threading
import numpy as np
from typing import Dict, Optional, List
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class StreamInfo:
    """Metadata and runtime state for a single camera stream"""
    stream_id: str
    source: str                           # "0" for webcam, file path, or RTSP URL
    source_type: str                      # "webcam" | "file" | "rtsp"
    name: str = "Unnamed Camera"
    is_active: bool = False
    fps: float = 0.0
    resolution: tuple = (0, 0)
    last_frame_time: float = 0.0
    total_frames_read: int = 0
    error_count: int = 0
    last_error: Optional[str] = None
    started_at: Optional[datetime] = None


class MultiStreamManager:
    """
    Manages multiple video capture sources concurrently.

    Each stream runs in its own daemon thread, continuously reading frames
    into a latest-frame buffer. Consumers call get_frame(stream_id) to
    retrieve the most recent frame without blocking.

    Usage:
        manager = MultiStreamManager()
        manager.add_stream("cam0", source="0", source_type="webcam", name="Main Gate")
        manager.add_stream("cam1", source="rtsp://...", source_type="rtsp", name="Parking")

        frame = manager.get_frame("cam0")  # Latest frame (numpy array or None)
        manager.remove_stream("cam1")
        manager.shutdown()
    """

    def __init__(self, max_streams: int = 8):
        self.max_streams = max_streams
        self._streams: Dict[str, StreamInfo] = {}
        self._captures: Dict[str, cv2.VideoCapture] = {}
        self._frames: Dict[str, Optional[np.ndarray]] = {}
        self._locks: Dict[str, threading.Lock] = {}
        self._threads: Dict[str, threading.Thread] = {}
        self._stop_events: Dict[str, threading.Event] = {}
        self._global_lock = threading.Lock()

        print(f"[MULTI-STREAM] Manager initialized (max {max_streams} streams)")

    # ========================================================================
    # Public API
    # ========================================================================

    def add_stream(
        self,
        stream_id: str,
        source: str,
        source_type: str = "webcam",
        name: str = "Camera",
        loop_file: bool = True
    ) -> bool:
        """
        Add and start a new camera stream.
        Returns True if stream was successfully opened, False otherwise.
        """
        with self._global_lock:
            if stream_id in self._streams:
                print(f"[MULTI-STREAM] Stream '{stream_id}' already exists, removing first")
                self._stop_stream_internal(stream_id)

            if len(self._streams) >= self.max_streams:
                print(f"[MULTI-STREAM] Cannot add stream: max {self.max_streams} reached")
                return False

        # Parse source
        video_source = int(source) if source.isdigit() else source

        # Open capture
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print(f"[MULTI-STREAM] Failed to open source: {source}")
            return False

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Get stream properties
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        info = StreamInfo(
            stream_id=stream_id,
            source=source,
            source_type=source_type,
            name=name,
            is_active=True,
            fps=fps,
            resolution=(width, height),
            started_at=datetime.now()
        )

        stop_event = threading.Event()
        lock = threading.Lock()

        with self._global_lock:
            self._streams[stream_id] = info
            self._captures[stream_id] = cap
            self._frames[stream_id] = None
            self._locks[stream_id] = lock
            self._stop_events[stream_id] = stop_event

        # Start capture thread
        thread = threading.Thread(
            target=self._capture_loop,
            args=(stream_id, loop_file),
            daemon=True,
            name=f"stream-{stream_id}"
        )
        self._threads[stream_id] = thread
        thread.start()

        print(f"[MULTI-STREAM] Added stream '{stream_id}' ({name}) — {width}x{height} @ {fps:.0f}fps")
        return True

    def remove_stream(self, stream_id: str) -> bool:
        """Stop and remove a stream"""
        with self._global_lock:
            if stream_id not in self._streams:
                return False
            self._stop_stream_internal(stream_id)
            print(f"[MULTI-STREAM] Removed stream '{stream_id}'")
            return True

    def get_frame(self, stream_id: str) -> Optional[np.ndarray]:
        """Get the latest frame from a stream (thread-safe, non-blocking)"""
        lock = self._locks.get(stream_id)
        if lock is None:
            return None
        with lock:
            frame = self._frames.get(stream_id)
            return frame.copy() if frame is not None else None

    def get_all_frames(self) -> Dict[str, Optional[np.ndarray]]:
        """Get latest frames from all active streams"""
        result = {}
        with self._global_lock:
            stream_ids = list(self._streams.keys())
        for sid in stream_ids:
            result[sid] = self.get_frame(sid)
        return result

    def get_stream_info(self, stream_id: str) -> Optional[dict]:
        """Get metadata for a specific stream"""
        info = self._streams.get(stream_id)
        if not info:
            return None
        return {
            "stream_id": info.stream_id,
            "name": info.name,
            "source": info.source,
            "source_type": info.source_type,
            "is_active": info.is_active,
            "fps": round(info.fps, 1),
            "resolution": f"{info.resolution[0]}x{info.resolution[1]}",
            "total_frames": info.total_frames_read,
            "errors": info.error_count,
            "last_error": info.last_error,
            "started_at": info.started_at.strftime("%H:%M:%S") if info.started_at else None
        }

    def list_streams(self) -> List[dict]:
        """List all stream metadata"""
        with self._global_lock:
            return [self.get_stream_info(sid) for sid in self._streams]

    def get_active_count(self) -> int:
        """Number of currently active streams"""
        with self._global_lock:
            return sum(1 for s in self._streams.values() if s.is_active)

    def shutdown(self):
        """Gracefully stop all streams"""
        with self._global_lock:
            stream_ids = list(self._streams.keys())
        for sid in stream_ids:
            self.remove_stream(sid)
        print("[MULTI-STREAM] All streams shut down")

    # ========================================================================
    # Internal
    # ========================================================================

    def _capture_loop(self, stream_id: str, loop_file: bool):
        """
        Continuous frame capture loop running in a dedicated thread.
        Reads frames as fast as the source provides them and stores
        only the latest frame (no queue/backlog).
        """
        stop_event = self._stop_events.get(stream_id)
        cap = self._captures.get(stream_id)
        info = self._streams.get(stream_id)
        lock = self._locks.get(stream_id)

        if not all([stop_event, cap, info, lock]):
            return

        target_interval = 1.0 / max(info.fps, 1)

        while not stop_event.is_set():
            try:
                ret, frame = cap.read()

                if not ret:
                    if loop_file and info.source_type == "file":
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        info.error_count += 1
                        info.last_error = "Frame read failed"
                        time.sleep(0.5)
                        continue

                with lock:
                    self._frames[stream_id] = frame

                info.last_frame_time = time.time()
                info.total_frames_read += 1

                # Throttle to avoid burning CPU on fast sources
                time.sleep(max(target_interval - 0.005, 0.005))

            except Exception as e:
                info.error_count += 1
                info.last_error = str(e)
                time.sleep(1)

        # Cleanup
        info.is_active = False

    def _stop_stream_internal(self, stream_id: str):
        """Stop a stream (must be called with _global_lock held)"""
        event = self._stop_events.pop(stream_id, None)
        if event:
            event.set()

        thread = self._threads.pop(stream_id, None)
        if thread and thread.is_alive():
            thread.join(timeout=2.0)

        cap = self._captures.pop(stream_id, None)
        if cap:
            cap.release()

        self._frames.pop(stream_id, None)
        self._locks.pop(stream_id, None)
        self._streams.pop(stream_id, None)
