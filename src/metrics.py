from prometheus_client import Counter, Histogram, Gauge
import time

# --------------------------------------------------
# Application start time
# --------------------------------------------------

START_TIME = time.time()

# --------------------------------------------------
# Request count
# --------------------------------------------------

REQUEST_COUNT = Counter(
    "api_requests_total",
    "Total number of API requests",
    ["method", "endpoint"]
)

# --------------------------------------------------
# Request latency
# --------------------------------------------------

REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds",
    "API request latency in seconds",
    ["method", "endpoint"]
)

# --------------------------------------------------
# Uptime
# --------------------------------------------------

UPTIME = Gauge(
    "api_uptime_seconds",
    "API application uptime in seconds"
)

# --------------------------------------------------
# Inference FPS
# --------------------------------------------------

INFERENCE_FPS = Gauge(
    "inference_fps",
    "YOLO inference FPS"
)


def update_uptime():
    """Update application uptime."""
    UPTIME.set(time.time() - START_TIME)


def record_request(method, endpoint, latency):
    """Record API request count and latency."""

    REQUEST_COUNT.labels(
        method=method,
        endpoint=endpoint
    ).inc()

    REQUEST_LATENCY.labels(
        method=method,
        endpoint=endpoint
    ).observe(latency)

    update_uptime()


def record_inference_fps(fps):
    """Record YOLO inference FPS."""
    INFERENCE_FPS.set(fps)