"""
detection/enhanced_crowd_detector.py
Multi-scale crowd detection with tracking and dense crowd estimation
"""

import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Tuple
from collections import defaultdict
import time


class EnhancedCrowdDetector:
    """
    Production-grade crowd detection:
    - Multi-scale detection
    - Object tracking
    - Dense crowd estimation
    - Occlusion handling
    """
    
    def __init__(self, model_path="yolov8n.pt", enable_tracking=True):
        self.model = YOLO(model_path)
        self.enable_tracking = enable_tracking
        
        # Tracking data
        self.tracked_objects = {}
        self.next_id = 0
        self.max_disappeared = 30  # Remove object after 30 frames
        self.disappeared_counts = defaultdict(int)
        
        # Performance metrics
        self.last_detection_time = 0
        
        print("[ENHANCED CROWD DETECTOR] Initialized")
    
    def detect(self, frame: np.ndarray, conf_threshold=0.35) -> Dict:
        """
        Optimized single-pass detection with tracking.
        
        Uses a single YOLO inference at 640px (the model's native training resolution)
        instead of multi-scale detection which caused 3x slower processing and
        phantom detections from imperfect NMS merging.
        
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
        
        h, w = frame.shape[:2]
        
        # Single optimized pass at 640px (YOLOv8n's native input size)
        all_detections = self._detect_at_scale(frame, conf_threshold, scale=1.0)
        
        # Tracking
        if self.enable_tracking:
            all_detections = self._update_tracking(all_detections)
        
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
        
        detection_time = (time.time() - start_time) * 1000
        
        return {
            'person_count': person_count,
            'detections': all_detections,
            'zones': zones,
            'tracked_ids': tracked_ids,
            'is_dense': is_dense,
            'detection_time_ms': round(detection_time, 2)
        }
    
    def _detect_at_scale(self, frame: np.ndarray, conf: float, scale: float) -> List[Dict]:
        """
        Run YOLO detection at specific scale
        """
        results = self.model(frame, conf=conf, verbose=False)[0]
        
        detections = []
        for box in results.boxes:
            class_id = int(box.cls[0])
            
            # Only process person class (class_id = 0 in COCO)
            if class_id != 0:
                continue
            
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            
            # Scale coordinates back to original size
            x1 = int(x1 / scale)
            y1 = int(y1 / scale)
            x2 = int(x2 / scale)
            y2 = int(y2 / scale)
            
            detections.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': confidence,
                'scale': scale,
                'centroid': [(x1 + x2) // 2, (y1 + y2) // 2]
            })
        
        return detections
    
    def _non_max_suppression(self, detections: List[Dict], iou_threshold: float = 0.4) -> List[Dict]:
        """
        Remove duplicate detections using NMS
        """
        if len(detections) == 0:
            return []
        
        # Sort by confidence
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        keep = []
        
        while detections:
            # Take highest confidence detection
            current = detections.pop(0)
            keep.append(current)
            
            # Remove overlapping detections
            detections = [
                det for det in detections
                if self._calculate_iou(current['bbox'], det['bbox']) < iou_threshold
            ]
        
        return keep
    
    def _calculate_iou(self, box1: List[int], box2: List[int]) -> float:
        """
        Calculate Intersection over Union
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def _update_tracking(self, detections: List[Dict]) -> List[Dict]:
        """
        Simple centroid-based tracking
        Assigns consistent IDs to people across frames
        """
        # Extract centroids from current detections
        current_centroids = {i: det['centroid'] for i, det in enumerate(detections)}
        
        # If no tracked objects yet, create new ones
        if len(self.tracked_objects) == 0:
            for i, det in enumerate(detections):
                self.tracked_objects[self.next_id] = det['centroid']
                det['id'] = self.next_id
                self.next_id += 1
            return detections
        
        # Match current detections to tracked objects
        tracked_centroids = list(self.tracked_objects.items())
        
        # Calculate distances between all pairs
        matched_ids = {}
        used_detection_indices = set()
        
        for obj_id, tracked_centroid in tracked_centroids:
            min_distance = float('inf')
            min_index = None
            
            for i, current_centroid in current_centroids.items():
                if i in used_detection_indices:
                    continue
                
                # Euclidean distance
                distance = np.sqrt(
                    (tracked_centroid[0] - current_centroid[0])**2 +
                    (tracked_centroid[1] - current_centroid[1])**2
                )
                
                if distance < min_distance:
                    min_distance = distance
                    min_index = i
            
            # If match is close enough (within 80 pixels)
            if min_distance < 80:
                matched_ids[min_index] = obj_id
                used_detection_indices.add(min_index)
                self.disappeared_counts[obj_id] = 0
            else:
                # Object disappeared
                self.disappeared_counts[obj_id] += 1
        
        # Assign IDs to detections
        for i, det in enumerate(detections):
            if i in matched_ids:
                det['id'] = matched_ids[i]
                self.tracked_objects[matched_ids[i]] = det['centroid']
            else:
                # New object
                det['id'] = self.next_id
                self.tracked_objects[self.next_id] = det['centroid']
                self.disappeared_counts[self.next_id] = 0
                self.next_id += 1
        
        # Remove objects that disappeared for too long
        to_remove = [
            obj_id for obj_id, count in self.disappeared_counts.items()
            if count > self.max_disappeared
        ]
        
        for obj_id in to_remove:
            del self.tracked_objects[obj_id]
            del self.disappeared_counts[obj_id]
        
        return detections
    
    def _estimate_dense_crowd(self, frame: np.ndarray, detections: List[Dict]) -> int:
        """
        Estimate additional people in dense crowd areas
        Uses head detection in regions where YOLO struggles
        """
        # Create mask of detected people
        h, w = frame.shape[:2]
        occupied_mask = np.zeros((h, w), dtype=np.uint8)
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            occupied_mask[y1:y2, x1:x2] = 255
        
        # Invert to get unoccupied regions
        unoccupied_mask = cv2.bitwise_not(occupied_mask)
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply mask
        gray_unoccupied = cv2.bitwise_and(gray, gray, mask=unoccupied_mask)
        
        # Detect circular regions (heads) using Hough circles
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
            # Cap at reasonable number
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
            
            col = min(cx // zone_w, 2)
            row = min(cy // zone_h, 2)
            
            zone_name = zone_map[row][col]
            zones[zone_name] += 1
        
        return zones
    
    def get_tracking_stats(self) -> Dict:
        """
        Get tracking statistics
        """
        return {
            'total_tracked': len(self.tracked_objects),
            'active_ids': list(self.tracked_objects.keys()),
            'next_id': self.next_id
        }


# Test
if __name__ == "__main__":
    detector = EnhancedCrowdDetector(enable_tracking=True)
    
    cap = cv2.VideoCapture(0)
    
    print("Testing enhanced crowd detector...")
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect
        result = detector.detect(frame, conf_threshold=0.3)
        
        # Visualize
        display = frame.copy()
        
        # Draw bounding boxes with IDs
        for det in result['detections']:
            bbox = det['bbox']
            x1, y1, x2, y2 = bbox
            
            # Color based on confidence
            conf = det['confidence']
            color = (0, int(255 * conf), int(255 * (1 - conf)))
            
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            
            # Show ID if tracked
            if 'id' in det:
                text = f"ID:{det['id']}"
                cv2.putText(display, text, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Show stats
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
        
        cv2.imshow("Enhanced Crowd Detection", display)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("\nFinal tracking stats:", detector.get_tracking_stats())