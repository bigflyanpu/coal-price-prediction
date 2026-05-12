FROM python:3.11-slim

WORKDIR /app/core
RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake && rm -rf /var/lib/apt/lists/*
COPY core/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV FAST_MODE=1
ENV LIVE_TEXT_SOURCES=0
ENV REFRESH_CACHE=1
ENV STRICT_REAL_DATA=1
ENV COAL_CPP_CORE=0
RUN STRICT_REAL_DATA=0 python train.py
EXPOSE 7860
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:7860", "--timeout", "120"]
