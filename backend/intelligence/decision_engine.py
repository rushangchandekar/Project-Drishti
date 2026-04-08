"""
intelligence/decision_engine.py
Hybrid decision intelligence (Rules + AI)
Provides strategic guidance based on detection context.
Agent orchestration is handled by n8n workflows.
"""

from typing import Dict, Any, List, Optional
from intelligence.gemini_integration import EnhancedGeminiAnalyzer
import time


class DecisionIntelligence:
    """
    Hybrid decision-making system combining:
    - Fast rule-based decisions (< 100ms)
    - AI-powered strategic analysis (1-3 sec)
    - Context-aware reasoning
    
    Note: Agent orchestration has been moved to n8n.
    This engine focuses on situation assessment and strategic guidance.
    """
    
    def __init__(self, gemini_api_key: Optional[str] = None):
        self.gemini = None
        
        if gemini_api_key:
            try:
                self.gemini = EnhancedGeminiAnalyzer(gemini_api_key)
            except Exception as e:
                print(f"[DECISION INTELLIGENCE] Gemini unavailable: {e}")
        
        # Decision history
        self.decision_history = []
        self.max_history = 100
        
        # Performance tracking
        self.rule_decisions = 0
        self.ai_decisions = 0
        
        # Rate limiting for Gemini calls (prevent quota exhaustion)
        self._last_ai_call_time = 0
        self._ai_cooldown = 120  # 2 minutes between AI calls in detection loop
        
        print("[DECISION INTELLIGENCE] Initialized")
    
    def make_decision(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make intelligent decision based on detection context.
        
        Returns:
            {
                'strategic_guidance': str,
                'reasoning': str,
                'confidence': float,
                'method': 'rules' | 'hybrid' | 'ai',
                'decision_time_ms': float
            }
        """
        start_time = time.time()
        
        # FAST PATH: Critical situations use rules only
        if self._is_critical(context):
            decision = self._rules_based_decision(context)
            decision['method'] = 'rules'
            self.rule_decisions += 1
        
        # HYBRID PATH: Use both rules and AI for complex situations
        elif self.gemini and self._needs_ai_analysis(context):
            decision = self._hybrid_decision(context)
            decision['method'] = 'hybrid'
            self.ai_decisions += 1
        
        # DEFAULT PATH: Rules only
        else:
            decision = self._rules_based_decision(context)
            decision['method'] = 'rules'
            self.rule_decisions += 1
        
        # Track decision time
        decision['decision_time_ms'] = (time.time() - start_time) * 1000
        
        # Log decision
        self._log_decision(context, decision)
        
        return decision
    
    def _is_critical(self, context: Dict[str, Any]) -> bool:
        """Check if situation is critical (needs immediate action)"""
        return (
            context.get('fire_detected', False) or
            context.get('risk_score', 0) > 90 or
            context.get('density_level') == 'CRITICAL'
        )
    
    def _needs_ai_analysis(self, context: Dict[str, Any]) -> bool:
        """Check if situation benefits from AI analysis (with rate limiting)"""
        # Rate limit: don't call Gemini more than once every 2 minutes
        if time.time() - self._last_ai_call_time < self._ai_cooldown:
            return False
        
        return (
            context.get('anomaly_detected', False) or
            context.get('risk_score', 0) > 60 or
            context.get('density_level') in ('HIGH', 'VERY_HIGH')
        )
    
    def _rules_based_decision(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fast rule-based decision making based on detection context.
        """
        guidance = self._generate_basic_guidance(context)
        risk = context.get('risk_score', 0)
        
        return {
            'strategic_guidance': guidance,
            'reasoning': f"Rule-based assessment: Risk={risk:.0f}, Density={context.get('density_level', 'UNKNOWN')}",
            'confidence': 0.95
        }
    
    def _hybrid_decision(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Hybrid decision using rules + AI analysis
        """
        # Start with rule-based assessment
        rule_result = self._rules_based_decision(context)
        
        # Get AI strategic analysis
        try:
            self._last_ai_call_time = time.time()  # Update cooldown timer
            
            # Build a summary for AI analysis
            situation_data = {
                'person_count': context.get('person_count', 0),
                'density_level': context.get('density_level', 'UNKNOWN'),
                'risk_score': context.get('risk_score', 0),
                'fire_detected': context.get('fire_detected', False),
                'anomaly_detected': context.get('anomaly_detected', False),
                'anomaly_type': context.get('anomaly_type'),
                'trend': context.get('trend', 'STABLE'),
                'zones': context.get('zones', {})
            }
            
            ai_analysis = self.gemini.analyze_decision_context(situation_data, [])
            
            # Combine insights
            strategic_guidance = (
                f"AI Assessment: {ai_analysis.get('strategic_assessment', 'Monitoring')}. "
                f"Actions: {', '.join(ai_analysis.get('priority_recommendations', [])[:2])}."
            )
            
            confidence = ai_analysis.get('confidence', 0.7)
            
        except Exception as e:
            print(f"[DECISION INTELLIGENCE] AI analysis failed: {e}")
            strategic_guidance = rule_result['strategic_guidance']
            confidence = rule_result['confidence']
        
        return {
            'strategic_guidance': strategic_guidance,
            'reasoning': f"Hybrid: Rule-based + AI strategic context",
            'confidence': confidence
        }
    
    def _generate_basic_guidance(self, context: Dict[str, Any]) -> str:
        """Generate basic strategic guidance without AI"""
        risk = context.get('risk_score', 0)
        density = context.get('density_level', 'UNKNOWN')
        trend = context.get('trend', 'STABLE')
        fire = context.get('fire_detected', False)
        
        if fire:
            return "🔥 FIRE DETECTED: Immediate evacuation required. All emergency protocols active."
        elif risk > 80:
            return "CRITICAL SITUATION: Deploy maximum resources. Consider evacuation."
        elif risk > 60:
            return "HIGH RISK: Increase monitoring and prepare for escalation."
        elif risk > 40:
            return "ELEVATED RISK: Proactive measures recommended."
        elif trend == 'INCREASING':
            return "GROWING CROWD: Monitor closely and prepare resources."
        else:
            return "NORMAL OPERATIONS: Continue routine surveillance."
    
    def _log_decision(self, context: Dict[str, Any], decision: Dict[str, Any]):
        """Log decision for analysis"""
        log_entry = {
            'timestamp': time.time(),
            'context_summary': {
                'person_count': context.get('person_count', 0),
                'risk_score': context.get('risk_score', 0),
                'density': context.get('density_level', 'UNKNOWN')
            },
            'method': decision['method'],
            'confidence': decision['confidence'],
            'decision_time_ms': decision.get('decision_time_ms', 0)
        }
        
        self.decision_history.append(log_entry)
        
        if len(self.decision_history) > self.max_history:
            self.decision_history.pop(0)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get decision intelligence statistics"""
        return {
            'total_decisions': len(self.decision_history),
            'rule_based': self.rule_decisions,
            'ai_enhanced': self.ai_decisions,
            'ai_usage_percentage': (self.ai_decisions / max(1, self.rule_decisions + self.ai_decisions)) * 100,
            'recent_decisions': self.decision_history[-10:] if self.decision_history else []
        }