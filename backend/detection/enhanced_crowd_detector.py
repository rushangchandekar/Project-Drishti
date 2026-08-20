import os
import sys
os.environ['ULTRALYTICS_CHECK_UPDATE'] = 'False'
os.environ['YOLOv5_VERBOSE'] = 'False'
if hasattr(sys, 'base_prefix'):
    venv_python = os.path.join(os.path.dirname(sys.executable), 'python.exe')
    os.environ['PYTHON'] = venv_python



"""
detection/enhanced_crowd_detector.py
Multi-scale crowd detection with tracking and dense crowd estimation

FIXED VERSION:
- Replaced fragile custom centroid tracker (fixed 80px radius, greedy matching,
  no motion prediction) with Ultralytics' built-in ByteTrack/BoT-SORT.
- This fixes ID continuity for fast/erratic motion (falls, fights, panic),
  which was previously causing IDs to reset mid-event and breaking all
  downstream activity recognition logic that depends on continuous history.
"""

import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Tuple
import time


class EnhancedCrowdDetector:
    """
    Production-grade crowd detection:
    - Configurable input resolution (320/480/640)
    - FP16 half-precision inference on CUDA GPUs
    - Robust object tracking via ByteTrack/BoT-SORT (Kalman-filter based)
    - Dense crowd estimation
    - Occlusion handling
    """

    def __init__(self, model_path="yolo11n.pt", enable_tracking=True,
                 input_size=640, half_precision=False,
                 max_detections=100, nms_iou=0.45,
                 tracker="bytetrack.yaml"):
        """
        Args:
            tracker: 'bytetrack.yaml' (faster, motion-only) or
                     'botsort.yaml' (slower, includes Re-ID — better for
                     crowds where people cross paths / occlude each other)
        """
        self.model = YOLO(model_path)
        self.enable_tracking = enable_tracking
        self.input_size = input_size
        self.half_precision = half_precision
        self.max_detections = max_detections
        self.nms_iou = nms_iou
        self.tracker = tracker

        # Enable FP16 if requested and CUDA available
        import torch
        if self.half_precision and torch.cuda.is_available():
            self.model.model.half()
            print(f"[ENHANCED CROWD DETECTOR] FP16 half-precision ENABLED on {torch.cuda.get_device_name(0)}")
        elif self.half_precision:
            self.half_precision = False
            print("[ENHANCED CROWD DETECTOR] FP16 requested but no CUDA GPU found — using FP32")

        # Lightweight bookkeeping for stats only (actual tracking state
        # lives inside the Ultralytics predictor when persist=True)
        self._last_active_ids: List[int] = []
        self._seen_ids_ever = set()

        # Performance metrics
        self.last_detection_time = 0

        print(f"[ENHANCED CROWD DETECTOR] Initialized (input={input_size}px, "
              f"fp16={self.half_precision}, max_det={max_detections}, "
              f"nms_iou={nms_iou}, tracking={'ByteTrack/BoT-SORT' if enable_tracking else 'OFF'}, "
              f"tracker_cfg={tracker if enable_tracking else 'N/A'})")

    def detect(self, frame: np.ndarray, conf_threshold=0.35) -> Dict:
        """
        Single-pass detection with robust tracking.

        Uses YOLO inference at configured input size, with tracking handled
        by Ultralytics' native ByteTrack/BoT-SORT implementation instead of
        a naive fixed-radius centroid matcher. This provides:
        - Kalman-filter motion prediction (survives fast/erratic movement)
        - Better occlusion handling in dense crowds
        - Optional Re-ID (BoT-SORT) for re-acquiring lost tracks

        Returns:
            {
                'person_count': int,
                'detections': List[dict],
                'zones': dict,
                'tracked_ids': List[int],
                'is_dense': bool,
                'detection_time_ms': float
            }
        """
        start_time = time.time()

        all_detections = self._detect_and_track(frame, conf_threshold)

        person_count = len(all_detections)

        # Dense crowd check and estimation (only if truly dense)
        is_dense = person_count > 40
        if is_dense:
            estimated_additional = self._estimate_dense_crowd(frame, all_detections)
            person_count += estimated_additional

        # Calculate zone distribution
        zones = self._calculate_zones(frame, all_detections)

        # Get tracked IDs
        tracked_ids = [d['id'] for d in all_detections if 'id' in d]
        self._last_active_ids = tracked_ids
        self._seen_ids_ever.update(tracked_ids)

        detection_time = (time.time() - start_time) * 1000

        return {
            'person_count': person_count,
            'detections': all_detections,
            'zones': zones,
            'tracked_ids': tracked_ids,
            'is_dense': is_dense,
            'detection_time_ms': round(detection_time, 2)
        }

    def _detect_and_track(self, frame: np.ndarray, conf: float) -> List[Dict]:
        """
        Run YOLO detection + tracking in one call.

        If enable_tracking=True, uses model.track() with persist=True so the
        tracker maintains internal state (Kalman filters, track history)
        across successive calls on the same detector instance. This REQUIRES
        that detect() be called on a temporally ordered sequence of frames
        (which is how video_processing.py already uses it).

        If enable_tracking=False, falls back to plain detection with no IDs.
        """
        classes_filter = [0]  # COCO class 0 = person (restricts tracker to persons only,
                               # avoids wasted compute + ID confusion from other classes)

        if self.enable_tracking:
            results = self.model.track(
                frame,
                persist=True,
                tracker=self.tracker,
                conf=conf,
                verbose=False,
                imgsz=self.input_size,
                half=self.half_precision,
                max_det=self.max_detections,
                iou=self.nms_iou,
                classes=classes_filter
            )[0]
        else:
            results = self.model(
                frame,
                conf=conf,
                verbose=False,
                imgsz=self.input_size,
                half=self.half_precision,
                max_det=self.max_detections,
                iou=self.nms_iou,
                classes=classes_filter
            )[0]

        detections = []

        if results.boxes is None or len(results.boxes) == 0:
            return detections

        boxes = results.boxes

        # Track IDs are only present when tracking is enabled AND the
        # tracker has confirmed the track (box.id may be None on very
        # first appearance for some trackers/configs).
        has_ids = self.enable_tracking and boxes.id is not None
        ids_tensor = boxes.id if has_ids else None

        for i in range(len(boxes)):
            confidence = float(boxes.conf[i])
            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            det = {
                'bbox': [x1, y1, x2, y2],
                'confidence': confidence,
                'centroid': [(x1 + x2) // 2, (y1 + y2) // 2]
            }

            if has_ids:
                det['id'] = int(ids_tensor[i])

            detections.append(det)

        return detections

    def _estimate_dense_crowd(self, frame: np.ndarray, detections: List[Dict]) -> int:
        """
        Estimate additional people in dense crowd areas
        Uses head detection in regions where YOLO struggles
        """
        h, w = frame.shape[:2]
        occupied_mask = np.zeros((h, w), dtype=np.uint8)

        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            occupied_mask[y1:y2, x1:x2] = 255

        unoccupied_mask = cv2.bitwise_not(occupied_mask)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_unoccupied = cv2.bitwise_and(gray, gray, mask=unoccupied_mask)

        circles = cv2.HoughCircles(
            gray_unoccupied,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=15,
            param1=50,
            param2=25,
            minRadius=8,
            maxRadius=35
        )

        additional_people = 0
        if circles is not None:
            additional_people = len(circles[0])
            additional_people = min(additional_people, 20)

        return additional_people

    def _calculate_zones(self, frame: np.ndarray, detections: List[Dict]) -> Dict[str, int]:
        """
        Calculate 3x3 zone distribution
        """
        h, w = frame.shape[:2]
        zone_w, zone_h = w // 3, h // 3

        zones = {
            "top_left": 0, "top_center": 0, "top_right": 0,
            "mid_left": 0, "mid_center": 0, "mid_right": 0,
            "bot_left": 0, "bot_center": 0, "bot_right": 0
        }

        zone_map = [
            ["top_left", "top_center", "top_right"],
            ["mid_left", "mid_center", "mid_right"],
            ["bot_left", "bot_center", "bot_right"]
        ]

        for det in detections:
            cx, cy = det['centroid']
            col = min(cx // max(zone_w, 1), 2)
            row = min(cy // max(zone_h, 1), 2)
            zone_name = zone_map[row][col]
            zones[zone_name] += 1

        return zones

    def get_tracking_stats(self) -> Dict:
        """
        Get tracking statistics.
        Note: actual tracker internal state (Kalman filters, lost tracks,
        etc.) lives inside the Ultralytics predictor when persist=True.
        This returns a lightweight external summary for monitoring/debugging.
        """
        return {
            'active_ids_last_frame': list(self._last_active_ids),
            'active_count': len(self._last_active_ids),
            'total_unique_ids_seen': len(self._seen_ids_ever)
        }

    def reset_tracker(self):
        """
        Reset tracking state completely (e.g. on video restart/loop,
        or when switching video sources) to avoid ID collisions between
        unrelated video segments.
        """
        self.model.predictor = None  # forces Ultralytics to reinit tracker state
        self._last_active_ids = []
        self._seen_ids_ever = set()
        print("[ENHANCED CROWD DETECTOR] Tracker state reset")


# Test
if __name__ == "__main__":
    detector = EnhancedCrowdDetector(enable_tracking=True)

    cap = cv2.VideoCapture(0)

    print("Testing enhanced crowd detector with ByteTrack...")
    print("Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = detector.detect(frame, conf_threshold=0.3)

        display = frame.copy()

        for det in result['detections']:
            bbox = det['bbox']
            x1, y1, x2, y2 = bbox

            conf = det['confidence']
            color = (0, int(255 * conf), int(255 * (1 - conf)))

            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)

            if 'id' in det:
                text = f"ID:{det['id']}"
                cv2.putText(display, text, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        stats_text = [
            f"People: {result['person_count']}",
            f"Dense: {result['is_dense']}",
            f"Time: {result['detection_time_ms']:.1f}ms",
            f"Tracked: {len(result['tracked_ids'])}"
        ]

        y_offset = 30
        for text in stats_text:
            cv2.putText(display, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_offset += 30

        cv2.imshow("Enhanced Crowd Detection (ByteTrack)", display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\nFinal tracking stats:", detector.get_tracking_stats())