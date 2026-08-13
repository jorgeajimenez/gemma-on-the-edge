# Use a lightweight Python base image
FROM python:3.11-slim

# Install system build dependencies required for compiling llama-cpp
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency specifications
COPY pyproject.toml requirements.txt* ./

# Install python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir flask huggingface_hub llama-cpp-python

# Copy server code
COPY server/server.py ./server.py

# Expose port 5002
EXPOSE 5002

# Run the server on container startup
CMD ["python", "server.py"]
