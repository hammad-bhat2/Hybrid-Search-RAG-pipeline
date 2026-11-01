# Stage 1: Build Frontend Assets
FROM node:20-alpine AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci --silent
COPY frontend/ ./
RUN npm run build

# Stage 2: Python Backend Runtime
FROM python:3.11-slim AS runtime
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and config
COPY config/ config/
COPY src/ src/
COPY pyproject.toml .

# Copy built frontend dist from Stage 1
COPY --from=frontend-builder /app/frontend/dist frontend/dist

# Expose backend API and web interface port
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    QDRANT_HOST=qdrant \
    QDRANT_PORT=6333

# Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Start Uvicorn production server
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
