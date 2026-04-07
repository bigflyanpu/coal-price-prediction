FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV FAST_MODE=1
ENV LIVE_TEXT_SOURCES=0
ENV REFRESH_CACHE=1
RUN python train.py
EXPOSE 7860
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:7860", "--timeout", "180"]
