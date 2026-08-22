import multiprocessing
import os

# Binding
bind = os.getenv("GUNICORN_BIND", "0.0.0.0:5000")

# Worker configuration
# For ML workloads, gthread is often preferred if thread-safe, or sync with multiple workers
worker_class = os.getenv("GUNICORN_WORKER_CLASS", "sync")
workers = int(os.getenv("GUNICORN_WORKERS", 2)) # Lower default to prevent OOM
threads = int(os.getenv("GUNICORN_THREADS", 2))
preload_app = True # Uses Copy-on-Write to share ML models across workers

# Graceful restarts to prevent memory leaks over time
max_requests = 1000
max_requests_jitter = 50

# Timeout
# ML inference might take time, setting a reasonable timeout
timeout = int(os.getenv("GUNICORN_TIMEOUT", 120))

# Logging
accesslog = "-"
errorlog = "-"
loglevel = os.getenv("GUNICORN_LOGLEVEL", "info")

# Security
limit_request_line = int(os.getenv("GUNICORN_LIMIT_REQUEST_LINE", 4094))
limit_request_fields = int(os.getenv("GUNICORN_LIMIT_REQUEST_FIELDS", 100))
limit_request_field_size = int(os.getenv("GUNICORN_LIMIT_REQUEST_FIELD_SIZE", 8190))
