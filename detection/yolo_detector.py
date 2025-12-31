"""
detection/yolo_detector.py
YOLO-based detection for crowd, fire, and anomalies
"""

from ultralytics import YOLO
import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
import time


@dataclass
class DetectionResult:
    """Structure to hold detection results"""
    person_count: int
    fire_detected: bool
    detections: List[dict]
    frame: np.ndarray
    timestamp: float
    crowd_density: str  # LOW, MEDIUM, HIGH, CRITICAL
    zones: dict  # Zone-wise breakdown


class DrishtiDetector:
    """
    Main detector class for Project Drishti
    Handles person counting and fire detection
    """
    
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence: float = 0.5,
        crowd_threshold_warning: int = 50,
        crowd_threshold_critical: int = 100
    ):
        """
        Initialize the detector
        
        Args:
            model_path: Path to YOLO model (will auto-download if not exists)
            confidence: Minimum confidence threshold
            crowd_threshold_warning: Person count for WARNING level
            crowd_threshold_critical: Person count for CRITICAL level
        """
        print(f"[INIT] Loading YOLO model: {model_path}")
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.threshold_warning = crowd_threshold_warning
        self.threshold_critical = crowd_threshold_critical
        
        # Class IDs in COCO dataset
        self.PERSON_CLASS_ID = 0
        
        # For fire detection, we'll use a simple color-based approach
        # (In production, use a fire-specific model)
        
        print("[INIT] Detector ready!")
    
    def detect(self, frame: np.ndarray) -> DetectionResult:
        """
        Run detection on a single frame
        
        Args:
            frame: BGR image from OpenCV
            
        Returns:
            DetectionResult with all detection info
        """
        timestamp = time.time()
        
        # Run YOLO inference
        results = self.model(frame, conf=self.confidence, verbose=False)[0]
        
        # Extract detections
        detections = []
        person_count = 0
        
        for box in results.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            detection = {
                "class_id": class_id,
                "class_name": results.names[class_id],
                "confidence": confidence,
                "bbox": [x1, y1, x2, y2]
            }
            detections.append(detection)
            
            # Count persons
            if class_id == self.PERSON_CLASS_ID:
                person_count += 1
        
        # Check for fire (color-based detection)
        fire_detected = self._detect_fire(frame)
        
        # Calculate crowd density level
        crowd_density = self._calculate_density_level(person_count)
        
        # Calculate zone-wise distribution (divide frame into 3x3 grid)
        zones = self._calculate_zones(frame, detections)
        
        # Draw detections on frame
        annotated_frame = self._annotate_frame(
            frame.copy(), 
            detections, 
            person_count, 
            fire_detected,
            crowd_density
        )
        
        return DetectionResult(
            person_count=person_count,
            fire_detected=fire_detected,
            detections=detections,
            frame=annotated_frame,
            timestamp=timestamp,
            crowd_density=crowd_density,
            zones=zones
        )
    
    def _calculate_density_level(self, count: int) -> str:
        """Determine crowd density level based on person count"""
        if count >= self.threshold_critical:
            return "CRITICAL"
        elif count >= self.threshold_warning:
            return "HIGH"
        elif count >= self.threshold_warning // 2:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _detect_fire(self, frame: np.ndarray) -> bool:
        """
        Simple fire detection using color analysis
        Looks for orange-red regions with high saturation
        
        Note: For production, use a dedicated fire detection model
        """
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define range for fire-like colors (orange-red)
        lower_fire = np.array([0, 120, 200])
        upper_fire = np.array([20, 255, 255])
        
        # Create mask
        mask = cv2.inRange(hsv, lower_fire, upper_fire)
        
        # Calculate percentage of fire-colored pixels
        fire_pixel_ratio = np.sum(mask > 0) / mask.size
        
        # If more than 1% of frame has fire colors, flag it
        return fire_pixel_ratio > 0.01
    
    def _calculate_zones(
        self, 
        frame: np.ndarray, 
        detections: List[dict]
    ) -> dict:
        """
        Divide frame into 3x3 grid and count persons in each zone
        
        Zones layout:
        ┌─────┬─────┬─────┐
        │ TL  │ TC  │ TR  │
        ├─────┼─────┼─────┤
        │ ML  │ MC  │ MR  │
        ├─────┼─────┼─────┤
        │ BL  │ BC  │ BR  │
        └─────┴─────┴─────┘
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
            if det["class_id"] != self.PERSON_CLASS_ID:
                continue
                
            # Get center of bounding box
            x1, y1, x2, y2 = det["bbox"]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            
            # Determine zone
            col = min(cx // zone_w, 2)
            row = min(cy // zone_h, 2)
            
            zone_name = zone_map[row][col]
            zones[zone_name] += 1
        
        return zones
    
    def _annotate_frame(
        self,
        frame: np.ndarray,
        detections: List[dict],
        person_count: int,
        fire_detected: bool,
        crowd_density: str
    ) -> np.ndarray:
        """Draw bounding boxes and status on frame"""
        
        # Color scheme based on density
        density_colors = {
            "LOW": (0, 255, 0),      # Green
            "MEDIUM": (0, 255, 255),  # Yellow
            "HIGH": (0, 165, 255),    # Orange
            "CRITICAL": (0, 0, 255)   # Red
        }
        
        # Draw person bounding boxes
        for det in detections:
            if det["class_id"] == self.PERSON_CLASS_ID:
                x1, y1, x2, y2 = det["bbox"]
                color = density_colors[crowd_density]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw status bar at top
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 80), (0, 0, 0), -1)
        
        # Person count
        cv2.putText(
            frame, 
            f"PERSONS: {person_count}", 
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.8, 
            (255, 255, 255), 
            2
        )
        
        # Density level
        cv2.putText(
            frame,
            f"DENSITY: {crowd_density}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            density_colors[crowd_density],
            2
        )
        
        # Fire warning
        if fire_detected:
            cv2.putText(
                frame,
                "🔥 FIRE DETECTED!",
                (frame.shape[1] - 250, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )
            # Red border around frame
            cv2.rectangle(
                frame, 
                (0, 0), 
                (frame.shape[1]-1, frame.shape[0]-1), 
                (0, 0, 255), 
                10
            )
        
        return frame


# Standalone test
if __name__ == "__main__":
    print("Testing DrishtiDetector...")
    
    # Initialize detector
    detector = DrishtiDetector(
        model_path="yolov8n.pt",
        confidence=0.5,
        crowd_threshold_warning=5,  # Low thresholds for testing
        crowd_threshold_critical=10
    )
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        exit()
    
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run detection
        result = detector.detect(frame)
        
        # Print stats
        print(f"\rPersons: {result.person_count} | "
              f"Density: {result.crowd_density} | "
              f"Fire: {result.fire_detected}", end="")
        
        # Show frame
        cv2.imshow("Drishti Detection", result.frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("\nTest complete!")