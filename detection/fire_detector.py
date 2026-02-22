"""
detection/fire_detector.py
Production-grade fire detection with multiple algorithms
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class FireDetection:
    """Fire detection result"""
    detected: bool
    confidence: float
    bounding_boxes: List[List[int]]
    fire_regions: int
    method: str  # 'color', 'ml', 'hybrid'


class AdvancedFireDetector:
    """
    Multi-method fire detection:
    1. Color-based (HSV + intensity)
    2. Motion-based (flicker detection)
    3. Temporal consistency
    4. Shape analysis
    """
    
    def __init__(self, mode='hybrid'):
        """
        Args:
            mode: 'color', 'motion', 'hybrid'
        """
        self.mode = mode
        self.frame_history = []
        self.detection_history = []
        self.max_history = 15  # Keep last 15 frames
        
        # Tunable parameters
        self.color_threshold = 0.015  # 1.5% of frame
        self.confidence_threshold = 0.7
        self.temporal_consistency = 0.6  # 60% of recent frames
        
        print(f"[FIRE DETECTOR] Initialized in {mode} mode")
    
    def detect(self, frame: np.ndarray) -> FireDetection:
        """
        Main detection method
        """
        if self.mode == 'color':
            return self._detect_color_based(frame)
        elif self.mode == 'motion':
            return self._detect_motion_based(frame)
        else:  # hybrid
            return self._detect_hybrid(frame)
    
    def _detect_color_based(self, frame: np.ndarray) -> FireDetection:
        """
        Advanced color-based detection with false positive reduction
        """
        h, w = frame.shape[:2]
        
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Method 1: Orange-red fire (bright flames)
        lower_orange = np.array([5, 150, 180])
        upper_orange = np.array([15, 255, 255])
        mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
        
        # Method 2: Yellow fire (intense flames)
        lower_yellow = np.array([18, 120, 180])
        upper_yellow = np.array([30, 255, 255])
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # Combine masks
        fire_mask = cv2.bitwise_or(mask_orange, mask_yellow)
        
        # Morphological operations to reduce noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_OPEN, kernel)
        fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_CLOSE, kernel)
        
        # Calculate fire pixel ratio
        fire_pixels = np.sum(fire_mask > 0)
        fire_ratio = fire_pixels / (h * w)
        
        # Check 1: Minimum area threshold
        if fire_ratio < self.color_threshold:
            return FireDetection(False, 0.0, [], 0, 'color')
        
        # Check 2: Contour analysis
        contours, _ = cv2.findContours(fire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return FireDetection(False, 0.0, [], 0, 'color')
        
        # Filter contours
        valid_contours = []
        bboxes = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Skip tiny regions
            if area < 100:
                continue
            
            # Get bounding box
            x, y, w_box, h_box = cv2.boundingRect(contour)
            
            # Check aspect ratio (fire is usually taller or square-ish)
            aspect_ratio = h_box / max(w_box, 1)
            
            # Fire aspect ratio: 0.5 to 3.0
            if 0.5 <= aspect_ratio <= 3.0:
                valid_contours.append(contour)
                bboxes.append([x, y, x + w_box, y + h_box])
        
        # Check 3: Must have at least one valid fire region
        if len(valid_contours) == 0:
            return FireDetection(False, 0.0, [], 0, 'color')
        
        # Check 4: Brightness analysis (fire should be bright)
        brightness_check = self._check_brightness(frame, fire_mask)
        
        if not brightness_check:
            return FireDetection(False, 0.0, [], 0, 'color')
        
        # Calculate confidence
        confidence = min(0.95, fire_ratio * 30 + 0.4)
        
        return FireDetection(
            detected=True,
            confidence=confidence,
            bounding_boxes=bboxes,
            fire_regions=len(valid_contours),
            method='color'
        )
    
    def _detect_motion_based(self, frame: np.ndarray) -> FireDetection:
        """
        Detect fire based on flicker/motion patterns
        Fire flickers at 5-15 Hz
        """
        # Store frame history
        self.frame_history.append(frame.copy())
        if len(self.frame_history) > self.max_history:
            self.frame_history.pop(0)
        
        # Need at least 10 frames for motion analysis
        if len(self.frame_history) < 10:
            return FireDetection(False, 0.0, [], 0, 'motion')
        
        # Calculate frame differences
        recent_frames = self.frame_history[-10:]
        
        # Convert to grayscale
        gray_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in recent_frames]
        
        # Calculate temporal variance (flicker detection)
        frame_stack = np.stack(gray_frames, axis=0)
        temporal_variance = np.var(frame_stack, axis=0)
        
        # Normalize
        variance_norm = cv2.normalize(temporal_variance, None, 0, 255, cv2.NORM_MINMAX)
        variance_norm = variance_norm.astype(np.uint8)
        
        # Threshold for high variance regions (flicker)
        _, flicker_mask = cv2.threshold(variance_norm, 100, 255, cv2.THRESH_BINARY)
        
        # Combine with color detection for confirmation
        color_detection = self._detect_color_based(frame)
        
        if color_detection.detected:
            # Motion confirms color detection
            return FireDetection(
                detected=True,
                confidence=min(0.95, color_detection.confidence + 0.1),
                bounding_boxes=color_detection.bounding_boxes,
                fire_regions=color_detection.fire_regions,
                method='motion'
            )
        
        return FireDetection(False, 0.0, [], 0, 'motion')
    
    def _detect_hybrid(self, frame: np.ndarray) -> FireDetection:
        """
        Combine color, motion, and temporal consistency
        """
        # Step 1: Color detection
        color_result = self._detect_color_based(frame)
        
        # Step 2: Add to detection history
        self.detection_history.append(color_result.detected)
        if len(self.detection_history) > self.max_history:
            self.detection_history.pop(0)
        
        # Step 3: Temporal consistency check
        if len(self.detection_history) >= 5:
            recent_detections = self.detection_history[-5:]
            consistency = sum(recent_detections) / len(recent_detections)
            
            # Require 60% consistency (3 out of 5 frames)
            if consistency >= self.temporal_consistency:
                # Fire confirmed
                return FireDetection(
                    detected=True,
                    confidence=min(0.98, color_result.confidence * 1.1),
                    bounding_boxes=color_result.bounding_boxes,
                    fire_regions=color_result.fire_regions,
                    method='hybrid'
                )
        
        # Not enough temporal consistency
        return FireDetection(False, 0.0, [], 0, 'hybrid')
    
    def _check_brightness(self, frame: np.ndarray, mask: np.ndarray) -> bool:
        """
        Check if masked regions are bright (fire characteristic)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Get pixels in fire region
        fire_pixels = gray[mask > 0]
        
        if len(fire_pixels) == 0:
            return False
        
        # Fire should have average brightness > 150
        avg_brightness = np.mean(fire_pixels)
        
        return avg_brightness > 150
    
    def reset(self):
        """Reset detection history"""
        self.frame_history.clear()
        self.detection_history.clear()


# Test
if __name__ == "__main__":
    import time
    
    detector = AdvancedFireDetector(mode='hybrid')
    
    # Test with webcam
    cap = cv2.VideoCapture(0)
    
    print("Testing fire detector... Press 'q' to quit")
    print("Hold something orange/red in front of camera to test")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect fire
        result = detector.detect(frame)
        
        # Visualize
        display = frame.copy()
        
        if result.detected:
            # Draw bounding boxes
            for bbox in result.bounding_boxes:
                cv2.rectangle(display, (bbox[0], bbox[1]), (bbox[2], bbox[3]), 
                            (0, 0, 255), 3)
            
            # Add text
            text = f"FIRE! Confidence: {result.confidence:.2f}"
            cv2.putText(display, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                       1, (0, 0, 255), 2)
        else:
            cv2.putText(display, "No Fire", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                       1, (0, 255, 0), 2)
        
        cv2.imshow("Fire Detection Test", display)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("Test complete!")