FROM --platform=linux/amd64 python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN python -m pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir torch torchvision \
       --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir seqeval==1.2.2 \
    && pip install --no-cache-dir --no-build-isolation .

ENV PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
ENV OMP_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1

EXPOSE 8501

CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501"]