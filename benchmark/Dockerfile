FROM mcr.microsoft.com/devcontainers/python:3.12

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY benchmark/requirements.txt ./requirements-benchmark.txt

RUN pip install --no-cache-dir -e ".[cache]" && \
    pip install --no-cache-dir -r requirements-benchmark.txt

COPY prism/ ./prism/
COPY benchmark/ ./benchmark/

ENV PYTHONUNBUFFERED=1
ENV APP_HOST=0.0.0.0
ENV APP_PORT=8000
ENV PRISM_BENCHMARK_SMOKE=0

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=3s --start-period=20s \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "-m", "benchmark.app.main"]
