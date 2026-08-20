"""
intelligence/gemini_integration.py
Enhanced Gemini integration for decision intelligence
"""

from google import genai
from typing import Dict, Any, Optional, List
import json
from backend.config import get_settings


class EnhancedGeminiAnalyzer:
    """
    Enhanced Gemini integration for Project Drishti
    
    Capabilities:
    - Situation analysis
    - Decision support
    - Natural language queries
    - Incident reports
    - Context-aware recommendations
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.settings = get_settings()
        self.api_key = api_key or self.settings.GEMINI_API_KEY
        self.openrouter_key = self.settings.OPENROUTER_API_KEY
        
        # Models
        self.gemini_model = "gemini-2.5-flash"
        self.openrouter_model = self.settings.OPENROUTER_MODEL or "meta-llama/llama-3.3-70b-instruct"
        self.model_name = self.openrouter_model if self.openrouter_key else self.gemini_model
        
        # Direct Gemini Client (for critical awareness, vision & fallback)
        self.client = None
        if self.api_key:
            try:
                self.client = genai.Client(api_key=self.api_key)
                print(f"[GEMINI INTELLIGENCE] Google Gemini client initialized ({self.gemini_model})")
            except Exception as e:
                print(f"[GEMINI INTELLIGENCE] Failed to init Gemini client: {e}")
            
        # Context memory
        self.conversation_history = []
        self.max_history = 10
        
        print(f"[GEMINI INTELLIGENCE] Analyzer ready (OpenRouter: {bool(self.openrouter_key)}, Direct Gemini: {bool(self.client)})")

    def _generate_content(self, prompt: str) -> str:
        """
        Generate text content.
        Uses OpenRouter for general assistant queries with seamless automatic fallback to direct Gemini API.
        """
        # Option 1: Try OpenRouter
        if self.openrouter_key:
            try:
                url = "https://openrouter.ai/api/v1/chat/completions"
                headers = {
                    "Authorization": f"Bearer {self.openrouter_key}",
                    "Content-Type": "application/json",
                }
                messages = [{"role": "user", "content": prompt}]
                payload = {
                    "model": self.openrouter_model,
                    "messages": messages,
                }
                import httpx
                with httpx.Client() as client:
                    response = client.post(url, headers=headers, json=payload, timeout=25.0)
                    if response.status_code == 200:
                        data = response.json()
                        return data['choices'][0]['message']['content'].strip()
                    else:
                        print(f"[GEMINI INTELLIGENCE WARN] OpenRouter returned HTTP {response.status_code}. Falling back to direct Gemini API...")
            except Exception as e:
                print(f"[GEMINI INTELLIGENCE WARN] OpenRouter call failed ({e}). Falling back to direct Gemini API...")

        # Option 2: Fallback to Direct Google Gemini API
        if self.client:
            try:
                response = self.client.models.generate_content(
                    model=self.gemini_model,
                    contents=prompt
                )
                return response.text.strip()
            except Exception as e:
                print(f"[GEMINI INTELLIGENCE ERROR] Direct Gemini API failed: {e}")
                raise

        raise ValueError("Neither OpenRouter nor Gemini API succeeded. Check your API keys and network connection.")
    
    def generate_situation_summary(
        self,
        person_count: int,
        density_level: str,
        trend: str,
        rate_of_change: float,
        predicted_count: int,
        risk_score: float,
        anomaly_type: Optional[str],
        zones: Dict[str, int]
    ) -> str:
        """
        Generate comprehensive situation summary
        """
        prompt = f"""You are Project Drishti, an AI crowd safety intelligence system.

Current Situation:
- Crowd Count: {person_count} people
- Density Level: {density_level}
- Trend: {trend} ({rate_of_change:+.1f} people/second)
- Predicted Count (1 min): {predicted_count}
- Risk Score: {risk_score}/100
- Anomaly: {anomaly_type or 'None'}
- Zone Distribution: {json.dumps(zones, indent=2)}

Generate a concise 2-3 sentence situation summary for the command center.
Focus on:
1. Current status
2. Trends and predictions
3. Immediate concerns
4. Recommended actions

Be clear, professional, and actionable."""

        try:
            return self._generate_content(prompt)
        except Exception as e:
            return f"Summary generation failed: {str(e)}"
    
    def analyze_decision_context(
        self,
        context: Dict[str, Any],
        agent_decisions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze context and agent decisions to provide strategic insights
        """
        prompt = f"""You are Project Drishti's strategic decision intelligence layer.

Current Context:
{json.dumps({k: v for k, v in context.items() if k != 'frame'}, indent=2)}

Agent Decisions Made:
{json.dumps(agent_decisions, indent=2)}

Analyze this situation and provide:
1. Strategic assessment of the decisions made
2. Potential risks or conflicts between actions
3. Missing considerations
4. Priority ranking of actions
5. Timeline for execution

Respond in JSON format:
{{
    "strategic_assessment": "brief assessment",
    "risks_identified": ["risk1", "risk2"],
    "missing_considerations": ["item1", "item2"],
    "priority_recommendations": ["action1", "action2", "action3"],
    "execution_timeline": "suggested timeline",
    "confidence": 0.0-1.0
}}"""

        try:
            text = self._generate_content(prompt)
            
            # Extract JSON
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            
            return json.loads(text)
        except Exception as e:
            return {
                "strategic_assessment": f"Analysis failed: {str(e)}",
                "risks_identified": [],
                "missing_considerations": [],
                "priority_recommendations": [],
                "execution_timeline": "unknown",
                "confidence": 0.0
            }
    
    def answer_query(self, question: str, context: Dict[str, Any]) -> str:
        """
        Answer natural language questions about the situation
        """
        # Build context summary
        context_summary = f"""Current System State:
- People: {context.get('person_count', 0)}
- Density: {context.get('density_level', 'UNKNOWN')}
- Trend: {context.get('trend', 'STABLE')}
- Risk Score: {context.get('risk_score', 0)}/100
- Fire: {'Yes' if context.get('fire_detected') else 'No'}
- Anomaly: {context.get('anomaly_type', 'None')}"""

        prompt = f"""You are Project Drishti's AI assistant.

{context_summary}

User Question: {question}

Provide a clear, concise answer based on the current data.
If you don't have enough information, say so and suggest what data would be needed."""

        try:
            return self._generate_content(prompt)
        except Exception as e:
            return f"Query processing failed: {str(e)}"
    
    def generate_incident_report(
        self,
        incident_type: str,
        timeline: List[Dict[str, Any]],
        actions_taken: List[str],
        outcome: str
    ) -> str:
        """
        Generate formal incident report
        """
        prompt = f"""Generate a formal incident report for Project Drishti.

Incident Type: {incident_type}

Timeline:
{json.dumps(timeline, indent=2)}

Actions Taken:
{json.dumps(actions_taken, indent=2)}

Outcome: {outcome}

Format as a professional incident report with:
1. Executive Summary
2. Incident Timeline
3. Response Actions
4. System Performance
5. Lessons Learned
6. Recommendations

Keep it concise but comprehensive."""

        try:
            return self._generate_content(prompt)
        except Exception as e:
            return f"Report generation failed: {str(e)}"
    
    def predict_escalation(
        self,
        current_context: Dict[str, Any],
        historical_trend: List[int]
    ) -> Dict[str, Any]:
        """
        Predict situation escalation
        """
        prompt = f"""Analyze this crowd safety situation and predict escalation potential.

Current State:
{json.dumps(current_context, indent=2)}

Historical Crowd Count (last 10 samples):
{historical_trend}

Predict:
1. Will situation escalate?
2. Timeframe for escalation
3. Potential triggers
4. Preventive measures

Respond in JSON:
{{
    "will_escalate": true/false,
    "confidence": 0.0-1.0,
    "timeframe_minutes": number,
    "triggers": ["trigger1", "trigger2"],
    "preventive_actions": ["action1", "action2"]
}}"""

        try:
            text = self._generate_content(prompt)
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            
            return json.loads(text)
        except Exception as e:
            return {
                "will_escalate": False,
                "confidence": 0.0,
                "timeframe_minutes": 0,
                "triggers": [],
                "preventive_actions": []
            }