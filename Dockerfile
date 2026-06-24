# Use an official lightweight Python runtime (pinned to bookworm for stable apt packages)
FROM python:3.10-slim-bookworm

# Set environment variables to prevent Python from writing .pyc files and to ensure output is unbuffered
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=5000
ENV MALLOC_ARENA_MAX=2

# Install system dependencies required by OpenCV (cv2)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# CRITICAL FOR RAILWAY: 
# Install the CPU-only version of PyTorch first. 
# The default PyTorch includes 2.5GB+ of NVIDIA CUDA binaries which will crash Railway builds and waste memory.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install the rest of the dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install Gunicorn, the production WSGI HTTP server
RUN pip install --no-cache-dir gunicorn werkzeug

# Copy the rest of the application code
COPY . .

# Expose the port Railway provides
EXPOSE $PORT

# Command to run the application using Gunicorn
# Timeout set to 120 seconds to allow for initial model loading
CMD gunicorn app:app -b 0.0.0.0:$PORT --timeout 120 --workers 1 --threads 2
