FROM python:3.10-slim

WORKDIR /app

# Cài đặt các gói cần thiết, bao gồm cả build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Cài đặt các thư viện Python, sau đó dọn dẹp build tools
RUN pip install --no-cache-dir -r requirements.txt \
    && apt-get purge -y --auto-remove build-essential

# Copy application code
COPY src/ ./src/
COPY run.py .

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "run.py"]