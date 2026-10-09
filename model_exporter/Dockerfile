FROM python:3.10-slim
WORKDIR /app
COPY model_exporter/requirements.txt ./service-requirements.txt
RUN pip install --no-cache-dir torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r service-requirements.txt
COPY model_exporter/ ./model_exporter/
COPY fedcore/ ./fedcore/
EXPOSE 5000
# The container is reached through the explicitly local host port mapping.
# The entry point refuses to start without FEDCORE_API_TOKEN.
CMD ["python", "-m", "model_exporter.api_server", "--host", "0.0.0.0", "--storage-root", "results/service"]
