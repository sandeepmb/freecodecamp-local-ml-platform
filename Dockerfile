FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ src/
COPY feature_repo/ feature_repo/

# Create directories for generated artifacts
RUN mkdir -p data models

# Generate training data and train model during build
RUN python src/generate_data.py && \
    python src/train_naive.py

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.serve_validated:app", "--host", "0.0.0.0", "--port", "8000"]
