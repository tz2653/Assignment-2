# Use a lightweight Python base image
FROM python:3.11-slim

# Avoid .pyc files and show logs unbuffered
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Workdir inside the container
WORKDIR /app

# System libs needed by Pillow/torch to handle images
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg62-turbo-dev libpng-dev && \
    rm -rf /var/lib/apt/lists/*

# 1) Install normal Python deps from PyPI
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 2) Install PyTorch family from the official CPU index (only these three)
RUN pip install --no-cache-dir torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cpu

# Copy source code and checkpoints into the image
COPY app /app/app
COPY helper_lib /app/helper_lib
COPY checkpoints /app/checkpoints

# Expose FastAPI port
EXPOSE 8000

# Launch API
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
