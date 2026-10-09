FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir fastapi==0.115.0 uvicorn==0.30.6 python-multipart==0.0.20

COPY app.py .

RUN mkdir -p uploads exports

EXPOSE 5000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000"]
