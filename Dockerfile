FROM python:3.12-slim

WORKDIR /app

# build-essential: some wheels; make: entrypoint; libgomp1: OpenMP runtime that
# faiss-cpu and onnxruntime link against (missing from slim, import fails without it).
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    make \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p indexes results

ENV PYTHONPATH=/app/src
# Keep BLAS/OMP thread counts predictable for reproducible latency.
ENV OMP_NUM_THREADS=4

# Default: run the benchmark. If indexes/ wasn't copied in, run `make index` first.
CMD ["make", "eval"]
