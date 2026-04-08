"""
intelligence/context_analyzer.py
Builds comprehensive context from all detection sources
"""

from typing import Dict, Any
import time


class ContextAnalyzer:
    """
    Builds comprehensive context from multiple detection sources
    """
    
    def __init__(self):
        self.last_context = {}
        print("[CONTEXT ANALYZER] Initialized")
    
    def build_context(
        self,
        detection_result: Dict[str, Any],
        crowd_analysis: Dict[str, Any],
        density_metrics: Any,
        anomaly_detection: Any,
        fire_detection: Any = None
    ) -> Dict[str, Any]:
        """
        Build comprehensive context for agent system
        """
        # Basic crowd data
        context = {
            # Crowd metrics
            'person_count': crowd_analysis['person_count'],
            'density_level': density_metrics.level,
            'density_value': density_metrics.density_value,
            'is_dense_crowd': crowd_analysis.get('is_dense', False),
            
            # Zones
            'zones': crowd_analysis['zones'],
            
            # Trends
            'trend': crowd_analysis.get('trend', 'STABLE'),
            'rate_of_change': crowd_analysis.get('rate_of_change', 0),
            'predicted_count_1min': crowd_analysis.get('predicted_count', 0),
            
            # Risk
            'risk_score': crowd_analysis.get('risk_score', 0),
            'risk_factor': density_metrics.risk_factor,
            
            # Anomalies
            'anomaly_detected': anomaly_detection.detected,
            'anomaly_type': anomaly_detection.anomaly_type,
            'anomaly_confidence': anomaly_detection.confidence,
            'anomaly_severity': anomaly_detection.severity,
            'anomaly_zones': anomaly_detection.affected_zones,
            
            # Fire
            'fire_detected': False,
            'fire_confidence': 0.0,
            'fire_locations': [],
            
            # Metadata
            'timestamp': time.time(),
            'detection_time_ms': crowd_analysis.get('detection_time_ms', 0),
            
            # Frame (for further processing)
            'frame': detection_result.get('frame')
        }
        
        # Add fire detection if available
        if fire_detection:
            context['fire_detected'] = fire_detection.detected
            context['fire_confidence'] = fire_detection.confidence
            context['fire_locations'] = [
                self._bbox_to_zone(bbox, context['zones'])
                for bbox in fire_detection.bounding_boxes
            ]
        
        # Calculate derived metrics
        context['situation_severity'] = self._assess_severity(context)
        context['requires_immediate_action'] = self._needs_immediate_action(context)
        
        # Store for comparison
        self.last_context = context
        
        return context
    
    def _bbox_to_zone(self, bbox: list, zones: dict) -> str:
        """Convert bounding box to zone name"""
        # Simple implementation - could be enhanced
        cx = (bbox[0] + bbox[2]) // 2
        cy = (bbox[1] + bbox[3]) // 2
        
        # Assume frame dimensions (could be passed in)
        frame_w, frame_h = 1280, 720
        
        zone_w = frame_w // 3
        zone_h = frame_h // 3
        
        col = min(cx // zone_w, 2)
        row = min(cy // zone_h, 2)
        
        zone_map = [
            ["top_left", "top_center", "top_right"],
            ["mid_left", "mid_center", "mid_right"],
            ["bot_left", "bot_center", "bot_right"]
        ]
        
        return zone_map[row][col]
    
    def _assess_severity(self, context: Dict[str, Any]) -> str:
        """Assess overall situation severity"""
        if context['fire_detected']:
            return "CRITICAL"
        
        risk = context['risk_score']
        
        if risk >= 90:
            return "CRITICAL"
        elif risk >= 70:
            return "SEVERE"
        elif risk >= 50:
            return "HIGH"
        elif risk >= 30:
            return "MODERATE"
        else:
            return "LOW"
    
    def _needs_immediate_action(self, context: Dict[str, Any]) -> bool:
        """Determine if immediate action is required"""
        return (
            context['fire_detected'] or
            context['density_level'] in ['CRITICAL', 'VERY_HIGH'] or
            context['risk_score'] > 85 or
            context['anomaly_severity'] == 'CRITICAL'
        )