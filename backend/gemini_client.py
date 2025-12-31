"""
backend/gemini_client.py
Google Gemini integration for intelligent analysis (Updated for new SDK)
"""

from google import genai
from typing import Dict, Any, Optional
import json
import os


class GeminiAnalyzer:
    """
    Uses Gemini for:
    - Situation summarization
    - Natural language queries
    - Complex multi-factor analysis
    - Incident report generation
    """
    
    def __init__(self, api_key: str):
        """Initialize Gemini client"""
        self.client = genai.Client(api_key=api_key)
        self.model_name = "gemini-2.5-flash-exp"
        
        print(f"[GEMINI] Initialized with model: {self.model_name}")
    
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
        Generate human-readable situation summary
        """
        prompt = f"""You are Project Drishti, an AI crowd safety system.

Current Situation Analysis:
- Current crowd count: {person_count} persons
- Density level: {density_level}
- Trend: {trend} ({rate_of_change:+.1f} persons/second)
- Predicted count (1 min): {predicted_count}
- Risk score: {risk_score}/100
- Anomaly: {anomaly_type or 'None'}
- Zone distribution: {json.dumps(zones, indent=2)}

Generate a concise 2-3 sentence situation summary for the command center.
Focus on actionable insights and any concerning patterns.
Use professional, clear language."""

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            return response.text.strip()
        except Exception as e:
            return f"Summary generation failed: {str(e)}"
    
    def answer_query(
        self,
        question: str,
        context: Dict[str, Any]
    ) -> str:
        """
        Answer natural language questions about current situation
        
        Example: "What's happening at Gate 3?"
        """
        prompt = f"""You are Project Drishti, an AI crowd safety assistant.

Current Context:
{json.dumps(context, indent=2)}

Command Center Question: {question}

Provide a clear, concise answer based on the current data.
If the data doesn't contain the answer, say so politely."""

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            return response.text.strip()
        except Exception as e:
            return f"Query failed: {str(e)}"
    
    def generate_incident_report(
        self,
        event_type: str,
        details: Dict[str, Any]
    ) -> str:
        """
        Generate formal incident report
        """
        prompt = f"""Generate a formal incident report for:

Event Type: {event_type}
Details: {json.dumps(details, indent=2)}

Format the report professionally with:
1. Incident Summary
2. Timeline
3. Actions Taken
4. Current Status
5. Recommendations

Keep it concise but comprehensive."""

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            return response.text.strip()
        except Exception as e:
            return f"Report generation failed: {str(e)}"
    
    def analyze_complex_situation(
        self,
        situation_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Deep analysis for complex scenarios
        Returns structured recommendations
        """
        prompt = f"""Analyze this crowd safety situation and provide structured recommendations:

Situation Data:
{json.dumps(situation_data, indent=2)}

Respond in JSON format with:
{{
    "severity": "low|medium|high|critical",
    "primary_concern": "brief description",
    "immediate_actions": ["action1", "action2"],
    "preventive_measures": ["measure1", "measure2"],
    "resource_allocation": "recommendation"
}}

Only return valid JSON, no other text."""

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            
            # Parse JSON response
            text = response.text.strip()
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            
            return json.loads(text)
        except Exception as e:
            # Fallback if JSON parsing fails
            return {
                "severity": "unknown",
                "primary_concern": f"Analysis failed: {str(e)}",
                "immediate_actions": [],
                "preventive_measures": [],
                "resource_allocation": "N/A"
            }


# Test
if __name__ == "__main__":
    from dotenv import load_dotenv
    
    # Load environment variables from parent directory
    load_dotenv("../.env")
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        print("ERROR: GEMINI_API_KEY not found in .env file")
        print(f"Current directory: {os.getcwd()}")
        print(f"Looking for .env in: {os.path.abspath('../.env')}")
        exit(1)
    
    print(f"API Key found: {api_key[:10]}...")
    
    # Initialize
    gemini = GeminiAnalyzer(api_key=api_key)
    
    # Test 1: Situation Summary
    print("\n=== TEST 1: Situation Summary ===")
    summary = gemini.generate_situation_summary(
        person_count=85,
        density_level="HIGH",
        trend="INCREASING",
        rate_of_change=1.5,
        predicted_count=120,
        risk_score=68,
        anomaly_type=None,
        zones={
            "top_left": 5, "top_center": 8, "top_right": 6,
            "mid_left": 12, "mid_center": 35, "mid_right": 10,
            "bot_left": 3, "bot_center": 4, "bot_right": 2
        }
    )
    print(summary)
    
    # Test 2: Natural Language Query
    print("\n=== TEST 2: NL Query ===")
    context = {
        "person_count": 85,
        "density": "HIGH",
        "trend": "INCREASING",
        "zones": {"mid_center": 35, "others": "distributed"}
    }
    answer = gemini.answer_query(
        question="Which area has the most people?",
        context=context
    )
    print(answer)
    
    print("\n✅ Gemini tests completed!")