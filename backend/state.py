import threading

# ============================================================================
# PERFORMANCE SETTINGS
# ============================================================================

STREAM_QUALITY = 65          # JPEG quality (lower = faster, 50-80 recommended)
STREAM_FPS = 15              # Target streaming FPS
DETECTION_FRAME_SKIP = 3     # Process every Nth frame
MAX_STREAM_WIDTH = 800       # Max width for streaming
ENABLE_FRAME_RESIZE = True   # Resize frames for streaming

# ============================================================================
# GLOBAL STATE
# ============================================================================

# Detection components (Initialized in lifespan)
fire_detector = None
crowd_detector = None
density_calculator = None
anomaly_detector = None
crowd_analyzer = None

# Intelligence components
context_analyzer = None
decision_intelligence = None

# Agent orchestrator
agent_orchestrator = None
agent_statuses = {}  # Populated by agent_orchestrator._build_initial_agent_statuses()

# System state variables
video_capture = None
current_state = {}
recent_agent_actions = []
detection_task = None
frame_lock = threading.Lock()  # Thread-safe frame access

# Performance tracking
performance_metrics = {
    'total_detections': 0,
    'total_webhooks_sent': 0,
    'avg_detection_time_ms': 0,
    'frames_streamed': 0
}

# Track last webhook send times to avoid spamming n8n
_last_webhook_times = {}
