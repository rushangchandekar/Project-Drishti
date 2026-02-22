import httpx
import time
from typing import Dict, Any, Optional
from backend.config import get_settings

settings = get_settings()

async def trigger_webhook(url: str, payload: Dict[str, Any]) -> Optional[Dict]:
    """Send webhook to n8n and return the response data"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, timeout=10.0)
            if response.status_code == 200:
                print(f"\n✅ Webhook sent: {url}")
                try:
                    return response.json()
                except Exception:
                    return {"response": response.text}
            else:
                print(f"\n⚠️  Webhook failed: {url} - Status {response.status_code}")
                return None
    except httpx.ConnectError:
        print(f"\n❌ Webhook connection refused: {url} - Is n8n running?")
        return None
    except httpx.TimeoutException:
        print(f"\n❌ Webhook timeout: {url}")
        return None
    except Exception as e:
        print(f"\n❌ Webhook error: {url} - {e}")
        return None


async def query_n8n_agent(question: str, context: Dict[str, Any]) -> Optional[str]:
    """Send a question to the n8n AI agent webhook and return the response"""
    webhook_url = f"{settings.N8N_WEBHOOK_BASE_URL}/chat"
    payload = {
        "question": question,
        "context": {k: v for k, v in context.items() if k != 'frame'},
        "timestamp": time.time()
    }
    
    result = await trigger_webhook(webhook_url, payload)
    
    if result:
        # n8n may return the answer in different fields depending on your workflow
        answer = (
            result.get('output') or
            result.get('answer') or
            result.get('response') or
            result.get('text') or
            result.get('message') or
            str(result)
        )
        return answer
    return None
