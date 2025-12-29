FROM python:3.10-slim

# Install build dependencies for dlib
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    wget \
    bzip2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for layer caching
COPY requirements.txt .
# Replace dlib-bin with dlib for Linux (dlib-bin is Windows-only)
RUN sed -i 's/dlib-bin/dlib/' requirements.txt && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY emotion_analyzer.py .
COPY gui_app.py .
COPY main.py .

# Download shape predictor
RUN wget -q http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 \
    && bunzip2 shape_predictor_68_face_landmarks.dat.bz2

# Default command
CMD ["python", "main.py", "--help"]
