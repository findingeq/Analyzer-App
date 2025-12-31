# Multi-stage build for React + FastAPI
# Stage 1: Build React frontend
FROM node:20-alpine AS frontend-builder
WORKDIR /app/web
COPY web/package*.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

# Stage 2: Python backend with built frontend
FROM python:3.11-slim
WORKDIR /app

# Install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY api/ ./api/

# Copy built frontend from stage 1
COPY --from=frontend-builder /app/web/dist ./web/dist

# Expose port
EXPOSE 8000

# Start the server
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
