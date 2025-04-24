# Use official Python image
FROM python:3.13-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    espeak \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy app files
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torchvision flask torch pandas opencv-python pillow pyttsx3

# Create uploads folder if not exists
RUN mkdir -p static/uploads

# Expose the default Flask port
EXPOSE 5000

# Command to run the app
CMD ["python", "app.py"]