# Base image with Python
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set working directory inside the container
WORKDIR /app

# Copy project files
COPY . .

# Install OS dependencies (optional: for scipy/audio)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --upgrade pip

# Use requirements.txt or pyproject.toml
COPY requirement.txt .  # Or use pyproject.toml instead
RUN pip install -r requirement.txt

# Expose the Gradio default port
EXPOSE 7860

# Run the Gradio app
CMD ["python", "main.py"]
