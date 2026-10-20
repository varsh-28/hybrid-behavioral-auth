# ---- build stage: train artifacts from the bundled datasets ----
FROM python:3.11-slim AS builder
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY scripts ./scripts
COPY data ./data
RUN python scripts/run_pipeline.py --fast

# ---- runtime stage: API only ----
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY --from=builder /app/models ./models
ENV PYTHONPATH=/app/src PYTHONUNBUFFERED=1
RUN useradd -m app && chown -R app /app
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1
CMD ["uvicorn", "hba.api:app", "--host", "0.0.0.0", "--port", "8000"]
