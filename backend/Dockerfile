# Use official lightweight Python image
FROM python:3.10-slim

# Avoid interactive prompts during install
ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --prefer-binary -r requirements.txt

# Copy all your code into the container
COPY . .

# Run your app (replace with your actual startup file if different)
CMD ["python", "app.py"]
