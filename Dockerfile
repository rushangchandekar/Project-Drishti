# Backend Dockerfile for Project Drishti (FastAPI + YOLOv11 + OpenCV)
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies required for OpenCV and GL libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy entire directory first or requirements specifically
COPY . /app/

# Install python dependencies
RUN if [ -f /app/requirements.txt ]; then \
        pip install --no-cache-dir -r /app/requirements.txt; \
    elif [ -f /app/backend/requirements.txt ]; then \
        pip install --no-cache-dir -r /app/backend/requirements.txt; \
    else \
        echo "Error: requirements.txt not found!" && exit 1; \
    fi

# Expose FastAPI port
EXPOSE 8000

# Set PYTHONPATH so 'backend.main' or 'main' can be imported seamlessly
ENV PYTHONPATH="/app:/app/backend"

# Run FastAPI backend using uvicorn
CMD ["sh", "-c", "if [ -f /app/backend/main.py ]; then uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}; else uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}; fi"]
