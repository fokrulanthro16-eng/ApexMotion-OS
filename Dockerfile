# Multi-stage production-hardened edge container for ApexMotion OS v3.0
# Optimized for Intel Core Ultra / Xeon Edge Gateways and Robotics IPCs

FROM python:3.11-slim as builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Final minimal deployment image
FROM python:3.11-slim

LABEL org.opencontainers.image.title="ApexMotion OS"
LABEL org.opencontainers.image.description="Physical AI & Edge Robotics Infrastructure Platform"
LABEL org.opencontainers.image.version="3.0.0"
LABEL org.opencontainers.image.vendor="ApexMotion Systems"
LABEL org.opencontainers.image.licenses="MIT"

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Security: Create non-root application user
RUN groupadd -g 1001 appgroup && \
    useradd -u 1001 -g appgroup -m -s /bin/bash appuser

WORKDIR /app

# Copy dependencies from builder
COPY --from=builder /root/.local /home/appuser/.local
ENV PATH=/home/appuser/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV HARDWARE_ACCELERATION=INTEL_NPU

# Copy application source files
COPY --chown=appuser:appgroup main.py planner.py vision.py tts_engine.py audio_bank.py test_system.py ./
COPY --chown=appuser:appgroup static/ ./static/

USER appuser

EXPOSE 8000

# Edge Node Healthcheck
HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "main.py"]
