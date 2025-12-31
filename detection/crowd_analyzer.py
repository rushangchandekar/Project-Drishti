"""
detection/crowd_analyzer.py
Advanced crowd analysis - trends, predictions, anomalies
"""

import time
from collections import deque
from dataclasses import dataclass
from typing import Optional, List
import numpy as np


@dataclass
class CrowdAnalysis:
    """Complete crowd analysis result"""
    current_count: int
    density_level: str
    trend: str  # INCREASING, DECREASING, STABLE
    rate_of_change: float  # persons per second
    predicted_count_1min: int
    anomaly_detected: bool
    anomaly_type: Optional[str]
    risk_score: float  # 0-100
    recommendation: str


class CrowdAnalyzer:
    """
    Analyzes crowd patterns over time
    Detects trends, anomalies, and predicts future states
    """
    
    def __init__(
        self,
        history_size: int = 30,  # Keep 30 samples
        sample_interval: float = 1.0,  # Expected seconds between samples
        sudden_change_threshold: float = 0.3  # 30% change is "sudden"
    ):
        self.history_size = history_size
        self.sample_interval = sample_interval
        self.sudden_change_threshold = sudden_change_threshold
        
        # Historical data
        self.count_history = deque(maxlen=history_size)
        self.timestamp_history = deque(maxlen=history_size)
        self.fire_history = deque(maxlen=history_size)
        
        print("[ANALYZER] Crowd analyzer initialized")
    
    def update(
        self,
        person_count: int,
        fire_detected: bool,
        timestamp: float,
        density_level: str,
        zones: dict
    ) -> CrowdAnalysis:
        """
        Update analyzer with new detection and return analysis
        """
        # Store in history
        self.count_history.append(person_count)
        self.timestamp_history.append(timestamp)
        self.fire_history.append(fire_detected)
        
        # Calculate trend
        trend, rate = self._calculate_trend()
        
        # Predict future count
        predicted_1min = self._predict_count(60)  # 60 seconds ahead
        
        # Detect anomalies
        anomaly_detected, anomaly_type = self._detect_anomalies(
            person_count, fire_detected, zones
        )
        
        # Calculate risk score
        risk_score = self._calculate_risk_score(
            person_count, density_level, fire_detected, 
            anomaly_detected, rate
        )
        
        # Generate recommendation
        recommendation = self._generate_recommendation(
            density_level, trend, fire_detected, 
            anomaly_detected, anomaly_type, risk_score
        )
        
        return CrowdAnalysis(
            current_count=person_count,
            density_level=density_level,
            trend=trend,
            rate_of_change=rate,
            predicted_count_1min=predicted_1min,
            anomaly_detected=anomaly_detected,
            anomaly_type=anomaly_type,
            risk_score=risk_score,
            recommendation=recommendation
        )
    
    def _calculate_trend(self) -> tuple:
        """Calculate if crowd is increasing, decreasing, or stable"""
        if len(self.count_history) < 5:
            return "STABLE", 0.0
        
        recent = list(self.count_history)[-10:]
        times = list(self.timestamp_history)[-10:]
        
        if len(recent) < 2:
            return "STABLE", 0.0
        
        # Calculate rate of change (persons per second)
        count_diff = recent[-1] - recent[0]
        time_diff = times[-1] - times[0]
        
        if time_diff == 0:
            return "STABLE", 0.0
        
        rate = count_diff / time_diff
        
        # Determine trend
        if rate > 0.5:
            trend = "INCREASING"
        elif rate < -0.5:
            trend = "DECREASING"
        else:
            trend = "STABLE"
        
        return trend, round(rate, 2)
    
    def _predict_count(self, seconds_ahead: int) -> int:
        """Simple linear prediction of future crowd count"""
        if len(self.count_history) < 5:
            return list(self.count_history)[-1] if self.count_history else 0
        
        _, rate = self._calculate_trend()
        current = list(self.count_history)[-1]
        
        predicted = current + (rate * seconds_ahead)
        return max(0, int(predicted))
    
    def _detect_anomalies(
        self,
        current_count: int,
        fire_detected: bool,
        zones: dict
    ) -> tuple:
        """Detect various anomalies in crowd behavior"""
        
        # Anomaly 1: Fire
        if fire_detected:
            return True, "FIRE_DETECTED"
        
        # Anomaly 2: Sudden crowd change
        if len(self.count_history) >= 5:
            recent = list(self.count_history)[-5:]
            avg = sum(recent[:-1]) / len(recent[:-1])
            
            if avg > 0:
                change_ratio = abs(current_count - avg) / avg
                if change_ratio > self.sudden_change_threshold:
                    if current_count > avg:
                        return True, "SUDDEN_INFLUX"
                    else:
                        return True, "SUDDEN_DISPERSAL"
        
        # Anomaly 3: Crowd concentration (one zone has >50% of people)
        total = sum(zones.values())
        if total > 10:  # Only check if meaningful crowd
            for zone_name, count in zones.items():
                if count / total > 0.5:
                    return True, f"OVERCROWDING_IN_{zone_name.upper()}"
        
        return False, None
    
    def _calculate_risk_score(
        self,
        count: int,
        density: str,
        fire: bool,
        anomaly: bool,
        rate: float
    ) -> float:
        """Calculate overall risk score 0-100"""
        score = 0.0
        
        # Base score from density
        density_scores = {
            "LOW": 10,
            "MEDIUM": 30,
            "HIGH": 60,
            "CRITICAL": 85
        }
        score += density_scores.get(density, 0)
        
        # Fire is critical
        if fire:
            score += 50
        
        # Anomaly adds risk
        if anomaly:
            score += 20
        
        # Rapid increase is risky
        if rate > 1.0:
            score += 10
        
        return min(100, score)
    
    def _generate_recommendation(
        self,
        density: str,
        trend: str,
        fire: bool,
        anomaly: bool,
        anomaly_type: str,
        risk_score: float
    ) -> str:
        """Generate actionable recommendation"""
        
        if fire:
            return "🚨 IMMEDIATE: Activate fire response. Initiate evacuation."
        
        if risk_score >= 80:
            return "🔴 CRITICAL: Deploy all response units. Consider evacuation."
        
        if density == "CRITICAL":
            return "🟠 WARNING: Open additional exits. Deploy crowd control."
        
        if anomaly and anomaly_type == "SUDDEN_INFLUX":
            return "⚠️ ALERT: Rapid crowd increase. Monitor closely. Prepare response teams."
        
        if anomaly and "OVERCROWDING" in (anomaly_type or ""):
            zone = anomaly_type.replace("OVERCROWDING_IN_", "").replace("_", " ")
            return f"⚠️ ALERT: Crowd concentrated in {zone}. Redirect to other areas."
        
        if density == "HIGH" and trend == "INCREASING":
            return "📈 CAUTION: Crowd growing. Pre-position response teams."
        
        if density == "HIGH":
            return "👁️ MONITOR: High density. Continue surveillance."
        
        return "✅ NORMAL: Situation under control."


# Test
if __name__ == "__main__":
    analyzer = CrowdAnalyzer()
    
    # Simulate increasing crowd
    import random
    base_count = 20
    
    for i in range(20):
        count = base_count + i * 2 + random.randint(-3, 3)
        fire = i == 15  # Fire at step 15
        
        zones = {
            "top_left": random.randint(0, 5),
            "top_center": random.randint(0, 5),
            "top_right": random.randint(0, 5),
            "mid_left": random.randint(0, 5),
            "mid_center": count // 2,  # Most people in center
            "mid_right": random.randint(0, 5),
            "bot_left": random.randint(0, 5),
            "bot_center": random.randint(0, 5),
            "bot_right": random.randint(0, 5),
        }
        
        density = "LOW" if count < 30 else "MEDIUM" if count < 50 else "HIGH" if count < 80 else "CRITICAL"
        
        result = analyzer.update(
            person_count=count,
            fire_detected=fire,
            timestamp=time.time(),
            density_level=density,
            zones=zones
        )
        
        print(f"\n--- Step {i+1} ---")
        print(f"Count: {result.current_count} | Trend: {result.trend} | Rate: {result.rate_of_change}/sec")
        print(f"Predicted (1min): {result.predicted_count_1min}")
        print(f"Risk Score: {result.risk_score}")
        print(f"Anomaly: {result.anomaly_type if result.anomaly_detected else 'None'}")
        print(f"Recommendation: {result.recommendation}")
        
        time.sleep(0.5)