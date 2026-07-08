FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PYTHONPATH=/app/src
ENV HF_HOME=/cache/huggingface
ENV LOG_TRIAGE_ADAPTER_DIR=/app/adapter

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
        python3 \
        python3-dev \
        python3-pip \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

ENV CC=/usr/bin/gcc
ENV CXX=/usr/bin/g++

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

RUN python -m pip install --upgrade pip setuptools wheel

COPY requirements-docker.txt /app/requirements-docker.txt

RUN python -m pip install \
        --index-url https://download.pytorch.org/whl/cu126 \
        torch \
    && python -m pip install -r /app/requirements-docker.txt

COPY src /app/src

EXPOSE 7860

CMD ["uvicorn", "log_triage.api:app", "--host", "0.0.0.0", "--port", "7860"]
