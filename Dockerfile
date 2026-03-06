# Stage 1: Build frontend
FROM node:20-slim AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci
COPY frontend/ .
RUN npm run build

# Stage 2: Python application
FROM python:3.11-slim

WORKDIR /app

# Install system deps for Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy source and install (pip needs src/ for the package build)
COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-cache-dir ".[server]"

# Copy frontend build from stage 1
COPY --from=frontend-builder /app/frontend/dist frontend/dist

EXPOSE 8000

CMD ["uvicorn", "weatherwise.server:app", "--host", "0.0.0.0", "--port", "8000"]
