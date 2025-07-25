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
COPY backend/requirements.txt .
RUN pip install --upgrade pip
RUN pip install --prefer-binary --only-binary=tokenizers -r requirements.txt

# ✅ Copy backend files
COPY backend/ .

# ✅ Explicitly copy aesthetic_profiles folder
COPY backend/aesthetic_profiles ./aesthetic_profiles

# Expose port for Render to bind to
EXPOSE 10000

# Run your app
CMD ["python", "app.py"]


