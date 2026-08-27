FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip install --no-cache-dir uv \
    && pip install --no-cache-dir torch torchvision \
    && pip install --no-cache-dir seqeval==1.2.2 \
    && pip install --no-cache-dir --no-build-isolation .

EXPOSE 8501

CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501"]