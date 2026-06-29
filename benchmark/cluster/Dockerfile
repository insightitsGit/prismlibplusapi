FROM python:3.11-slim

WORKDIR /app

# System deps for psutil
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Disable pip's Unicode progress bar / color: the az-cli build-log streamer on
# Windows (colorama) crashes trying to encode the U+2501 bar chars to cp1252.
ENV PIP_PROGRESS_BAR=off PIP_NO_COLOR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN pip install --no-cache-dir --progress-bar off -r requirements.txt

COPY node_app.py .

EXPOSE 8080

CMD ["python", "-u", "node_app.py"]
