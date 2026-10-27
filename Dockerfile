# syntax=docker/dockerfile:1
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir -r requirements.txt

COPY card ./card
COPY app.py ./
COPY models ./models
COPY reports/evaluation ./reports/evaluation

EXPOSE 8501

HEALTHCHECK CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", "--server.headless", "true", "--server.port", "8501", "--server.address", "0.0.0.0"]
