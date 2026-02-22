"""
detection/anomaly_detector.py
Detect anomalous crowd behavior patterns
"""

import numpy as np
from collections import deque
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import time


@dataclass
class AnomalyDetection:
    """Anomaly detection result"""
    detected: bool
    anomaly_type: Optional[str]
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    confidence: float
    description: str
    affected_zones: List[str]


class CrowdAnomalyDetector:
    """
    Detect anomalous patterns:
    - Sudden influx/dispersal
    - Panic movements
    - Zone imbalance
    - Flow disruption
    - Loitering
    - Counter-flow
    """
    
    def __init__(self, history_size=60):
        """
        Args:
            history_size: Number of frames to keep in history
        """
        self.history_size = history_size
        
        # Historical data
        self.count_history = deque(maxlen=history_size)
        self.zone_history = deque(maxlen=history_size)
        self.timestamp_history = deque(maxlen=history_size)
        self.velocity_history = deque(maxlen=30)  # Shorter for movement
        
        # Anomaly thresholds
        self.sudden_change_threshold = 0.4  # 40% change
        self.imbalance_threshold = 0.7  # 70% in one zone
        self.loitering_threshold = 50  # 50 frames (~5 seconds)
        
        print("[ANOMALY DETECTOR] Initialized")
    
    def detect(self, person_count: int, zones: Dict[str, int], 
               detections: List[Dict], timestamp: float) -> AnomalyDetection:
        """
        Detect anomalies in current frame
        """
        # Store history
        self.count_history.append(person_count)
        self.zone_history.append(zones.copy())
        self.timestamp_history.append(timestamp)
        
        # Need minimum history
        if len(self.count_history) < 10:
            return AnomalyDetection(
                detected=False,
                anomaly_type=None,
                severity='LOW',
                confidence=0.0,
                description="Gathering baseline data",
                affected_zones=[]
            )
        
        # Check for various anomalies
        anomalies = []
        
        # 1. Sudden influx/dispersal
        influx_anomaly = self._detect_sudden_change()
        if influx_anomaly:
            anomalies.append(influx_anomaly)
        
        # 2. Zone imbalance
        imbalance_anomaly = self._detect_zone_imbalance(zones, person_count)
        if imbalance_anomaly:
            anomalies.append(imbalance_anomaly)
        
        # 3. Abnormal density pattern
        density_anomaly = self._detect_density_anomaly()
        if density_anomaly:
            anomalies.append(density_anomaly)
        
        # 4. Movement pattern anomaly
        if len(detections) > 0:
            movement_anomaly = self._detect_movement_anomaly(detections)
            if movement_anomaly:
                anomalies.append(movement_anomaly)
        
        # Return highest severity anomaly
        if anomalies:
            # Sort by severity
            severity_order = {'LOW': 0, 'MEDIUM': 1, 'HIGH': 2, 'CRITICAL': 3}
            anomalies.sort(key=lambda x: severity_order[x.severity], reverse=True)
            return anomalies[0]
        
        return AnomalyDetection(
            detected=False,
            anomaly_type=None,
            severity='LOW',
            confidence=0.0,
            description="Normal behavior",
            affected_zones=[]
        )
    
    def _detect_sudden_change(self) -> Optional[AnomalyDetection]:
        """
        Detect sudden crowd influx or dispersal
        """
        if len(self.count_history) < 20:
            return None
        
        recent = list(self.count_history)[-10:]
        older = list(self.count_history)[-20:-10]
        
        recent_avg = np.mean(recent)
        older_avg = np.mean(older)
        
        if older_avg == 0:
            return None
        
        change_ratio = abs(recent_avg - older_avg) / older_avg
        
        if change_ratio > self.sudden_change_threshold:
            if recent_avg > older_avg:
                anomaly_type = "SUDDEN_INFLUX"
                description = f"Rapid crowd increase: {older_avg:.0f} → {recent_avg:.0f} people"
                severity = "HIGH" if change_ratio > 0.6 else "MEDIUM"
            else:
                anomaly_type = "SUDDEN_DISPERSAL"
                description = f"Rapid crowd decrease: {older_avg:.0f} → {recent_avg:.0f} people"
                severity = "MEDIUM"
            
            return AnomalyDetection(
                detected=True,
                anomaly_type=anomaly_type,
                severity=severity,
                confidence=min(0.95, change_ratio),
                description=description,
                affected_zones=[]
            )
        
        return None
    
    def _detect_zone_imbalance(self, zones: Dict[str, int], total_count: int) -> Optional[AnomalyDetection]:
        """
        Detect severe crowd concentration in one zone
        """
        if total_count < 10:
            return None
        
        max_zone = max(zones.items(), key=lambda x: x[1])
        max_zone_name, max_zone_count = max_zone
        
        concentration_ratio = max_zone_count / total_count
        
        if concentration_ratio > self.imbalance_threshold:
            severity = "CRITICAL" if concentration_ratio > 0.85 else "HIGH"
            
            return AnomalyDetection(
                detected=True,
                anomaly_type="ZONE_OVERCROWDING",
                severity=severity,
                confidence=concentration_ratio,
                description=f"{int(concentration_ratio*100)}% of crowd in {max_zone_name.replace('_', ' ').title()}",
                affected_zones=[max_zone_name]
            )
        
        return None
    
    def _detect_density_anomaly(self) -> Optional[AnomalyDetection]:
        """
        Detect unusual density patterns (e.g., oscillation, sustained high)
        """
        if len(self.count_history) < 30:
            return None
        
        recent_30 = list(self.count_history)[-30:]
        
        # Check for oscillation (could indicate panic)
        differences = np.diff(recent_30)
        sign_changes = np.sum(np.diff(np.sign(differences)) != 0)
        
        # Many sign changes = oscillation
        if sign_changes > 15 and np.mean(recent_30) > 20:
            return AnomalyDetection(
                detected=True,
                anomaly_type="CROWD_OSCILLATION",
                severity="HIGH",
                confidence=0.75,
                description="Unusual crowd movement pattern detected (possible panic)",
                affected_zones=[]
            )
        
        return None
    
    def _detect_movement_anomaly(self, detections: List[Dict]) -> Optional[AnomalyDetection]:
        """
        Detect anomalous movement patterns
        """
        # This would require tracking data
        # For now, basic implementation
        
        if len(detections) < 5:
            return None
        
        # Check if people are tracked
        tracked = [d for d in detections if 'id' in d]
        
        if len(tracked) < 5:
            return None
        
        # Could implement:
        # - Counter-flow detection
        # - Sudden stops
        # - Erratic movement
        # - Group formation/breaking
        
        # Placeholder for now
        return None
    
    def get_trend(self) -> str:
        """
        Get overall trend: INCREASING, DECREASING, STABLE
        """
        if len(self.count_history) < 10:
            return "STABLE"
        
        recent = list(self.count_history)[-10:]
        
        # Linear regression slope
        x = np.arange(len(recent))
        slope = np.polyfit(x, recent, 1)[0]
        
        if slope > 0.5:
            return "INCREASING"
        elif slope < -0.5:
            return "DECREASING"
        else:
            return "STABLE"
    
    def reset(self):
        """Reset all history"""
        self.count_history.clear()
        self.zone_history.clear()
        self.timestamp_history.clear()
        self.velocity_history.clear()


# Test
if __name__ == "__main__":
    detector = CrowdAnomalyDetector()
    
    print("Testing anomaly detector with simulated data...\n")
    
    # Simulate normal increase
    print("=== SCENARIO 1: Normal Gradual Increase ===")
    for i in range(30):
        count = 10 + i
        zones = {
            "top_left": count // 9, "top_center": count // 9, "top_right": count // 9,
            "mid_left": count // 9, "mid_center": count // 9, "mid_right": count // 9,
            "bot_left": count // 9, "bot_center": count // 9, "bot_right": count // 9
        }
        
        result = detector.detect(count, zones, [], time.time())
        
        if result.detected:
            print(f"Frame {i:2d}: {result.anomaly_type} - {result.description}")
    
    print(f"Trend: {detector.get_trend()}")
    
    # Reset and test sudden influx
    detector.reset()
    print("\n=== SCENARIO 2: Sudden Influx ===")
    
    for i in range(20):
        if i < 10:
            count = 20
        else:
            count = 60  # Sudden jump
        
        zones = {
            "top_left": count // 9, "top_center": count // 9, "top_right": count // 9,
            "mid_left": count // 9, "mid_center": count // 9, "mid_right": count // 9,
            "bot_left": count // 9, "bot_center": count // 9, "bot_right": count // 9
        }
        
        result = detector.detect(count, zones, [], time.time())
        
        if result.detected:
            print(f"Frame {i:2d}: {result.anomaly_type} - {result.description} (Severity: {result.severity})")
    
    # Test zone imbalance
    detector.reset()
    print("\n=== SCENARIO 3: Zone Overcrowding ===")
    
    for i in range(20):
        count = 50
        zones = {
            "top_left": 2, "top_center": 2, "top_right": 2,
            "mid_left": 3, "mid_center": 35, "mid_right": 2,  # 70% in center
            "bot_left": 2, "bot_center": 1, "bot_right": 1
        }
        
        result = detector.detect(count, zones, [], time.time())
        
        if result.detected:
            print(f"Frame {i:2d}: {result.anomaly_type} - {result.description} (Severity: {result.severity})")