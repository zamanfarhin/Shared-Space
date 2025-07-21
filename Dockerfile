# Use official lightweight Python image
FROM python:3.10-slim

# Avoid interactive prompts during install
ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory inside container
WORKDIR /app

# Copy backend requirements and install dependencies
COPY backend/requirements.txt .
RUN pip install --upgrade pip
RUN pip install --prefer-binary -r requirements.txt

# Copy all backend source code into the container
COPY backend/ .

# Run your app
CMD ["python", "app.py"]
