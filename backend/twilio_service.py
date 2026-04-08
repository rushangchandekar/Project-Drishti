from twilio.rest import Client
import time
from backend.config import get_settings

settings = get_settings()

def send_emergency_whatsapp(alert_type: str, details: str):
    """
    Send an emergency WhatsApp message using Twilio Sandbox.
    """
    if not settings.TWILIO_ACCOUNT_SID or not settings.TWILIO_AUTH_TOKEN:
        print("⚠️ Twilio credentials missing. SMS/WhatsApp will not be sent.")
        return False
        
    try:
        client = Client(settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN)
        
        message_body = (
            f"🚨 *DRISHTI EMERGENCY ALERT* 🚨\n\n"
            f"*Type:* {alert_type}\n"
            f"*Details:* {details}\n"
            f"*Time:* {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            f"Autonomous actions have been initiated."
        )
        
        message = client.messages.create(
            from_=settings.TWILIO_FROM_NUMBER,
            body=message_body,
            to=settings.TWILIO_TO_NUMBER
        )
        
        print(f"✅ Twilio WhatsApp sent successfully! SID: {message.sid}")
        return True
    except Exception as e:
        print(f"❌ Failed to send Twilio WhatsApp message: {e}")
        return False
