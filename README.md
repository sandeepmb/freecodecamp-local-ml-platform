# FreeCodeCamp Local ML Platform

A complete, production-ready ML platform built locally with experiment tracking, feature store, data validation, monitoring, and CI/CD. This project demonstrates how to build a fraud detection system from scratch with all the MLOps infrastructure needed for production.

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Chapter-by-Chapter Guide](#chapter-by-chapter-guide)
- [Running the Complete Pipeline](#running-the-complete-pipeline)
- [Testing](#testing)
- [Docker Deployment](#docker-deployment)
- [Troubleshooting](#troubleshooting)

## 🎯 Overview

This project implements a **fraud detection ML system** with production-grade infrastructure:

| Component | Tool | Purpose |
|-----------|------|---------|
| **Experiment Tracking** | MLflow | Track experiments, compare runs, reproduce results |
| **Model Registry** | MLflow | Version models, manage deployment stages |
| **Feature Store** | Feast | Consistent features between training and serving |
| **Data Validation** | Great Expectations | Reject invalid input data |
| **Monitoring** | Evidently | Detect data drift and model decay |
| **API Serving** | FastAPI | High-performance prediction API |
| **Containerization** | Docker | Environment consistency |
| **CI/CD** | GitHub Actions | Automated testing and deployment |

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Training Pipeline                         │
├─────────────────────────────────────────────────────────────────┤
│  Raw Data → Great Expectations → Feast Feature Store            │
│                    ↓                                             │
│            Training Script (MLflow tracking)                     │
│                    ↓                                             │
│            MLflow Model Registry (versioned models)              │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                        Serving Pipeline                          │
├─────────────────────────────────────────────────────────────────┤
│  Client Request → FastAPI → Data Validation                     │
│                    ↓                                             │
│            Feast Online Store (features)                         │
│                    ↓                                             │
│            MLflow Production Model                               │
│                    ↓                                             │
│            Prediction Response                                   │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                      Monitoring & CI/CD                          │
├─────────────────────────────────────────────────────────────────┤
│  Evidently (drift detection) → Alerts → Retrain                 │
│  GitHub Actions → Tests → Docker Build → Deploy                 │
└─────────────────────────────────────────────────────────────────┘
```

## 📦 Prerequisites

- **Python 3.9+** installed
- **Docker Desktop** installed and running
- **Git** for version control
- **8GB RAM** minimum (16GB recommended)
- **5GB disk space** for dependencies and data

## 🚀 Quick Start

### 1. Clone and Setup

```bash
# Navigate to project directory
cd freecodecamp-local-ml-platform

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Data and Train Model

```bash
# Generate synthetic fraud detection dataset
python src/generate_data.py

# Train the naive model (baseline)
python src/train_naive.py
```

### 3. Start the API

```bash
# Run the API server
uvicorn src.serve_naive:app --reload --host 0.0.0.0 --port 8000
```

Visit `http://localhost:8000/docs` for interactive API documentation.

### 4. Test the API

```bash
# In a new terminal
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"amount": 150.0, "hour": 14, "day_of_week": 3, "merchant_category": "online"}'
```

## 📁 Project Structure

```
freecodecamp-local-ml-platform/
├── data/                          # Training and test datasets
│   ├── train.csv
│   ├── test.csv
│   ├── merchant_features.parquet  # Feast features
│   ├── registry.db                # Feast registry
│   └── online_store.db            # Feast online store
├── models/                        # Saved model files
│   └── model.pkl
├── src/                           # Source code
│   ├── generate_data.py           # Chapter 1: Data generation
│   ├── train_naive.py             # Chapter 1: Naive training
│   ├── serve_naive.py             # Chapter 1: Naive API
│   ├── test_bad_data.py           # Chapter 2: Testing bad inputs
│   ├── train_mlflow.py            # Chapter 3: MLflow training
│   ├── serve_mlflow.py            # Chapter 3: MLflow API
│   ├── prepare_feast_features.py  # Chapter 4: Feast setup
│   ├── feast_features.py          # Chapter 4: Feature retrieval
│   ├── data_validation.py         # Chapter 5: Validation logic
│   ├── serve_validated.py         # Chapter 5: Validated API
│   └── monitoring.py              # Chapter 6: Drift detection
├── tests/                         # Test suite
│   ├── test_data_and_model.py     # Data and model tests
│   └── test_api.py                # API integration tests
├── feature_repo/                  # Feast feature repository
│   ├── feature_store.yaml         # Feast configuration
│   └── features.py                # Feature definitions
├── .github/workflows/             # CI/CD configuration
│   └── ci.yml                     # GitHub Actions workflow
├── Dockerfile                     # Container definition
├── .dockerignore                  # Docker ignore patterns
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 📚 Chapter-by-Chapter Guide

### Chapter 1: Build a Simple Model and API (Naive Approach)

**Goal:** Create a basic ML system to understand what's missing.

```bash
# Generate data
python src/generate_data.py

# Train model
python src/train_naive.py

# Start API
uvicorn src.serve_naive:app --reload --port 8000

# Test with bad data (shows the problem)
python src/test_bad_data.py
```

**What you'll see:** The API accepts garbage data and returns predictions without warnings.

### Chapter 2: Where the Naive Approach Breaks

**Problems identified:**
- ❌ No experiment tracking (can't reproduce results)
- ❌ No model versioning (can't roll back)
- ❌ No data validation (garbage in, garbage out)
- ❌ No monitoring (drift goes unnoticed)
- ❌ No CI/CD (risky deployments)

### Chapter 3: Add Experiment Tracking with MLflow

**Goal:** Track experiments and version models.

```bash
# Terminal 1: Start MLflow server
mkdir -p mlruns
mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlruns

# Terminal 2: Run experiment sweep
python src/train_mlflow.py

# View experiments at http://localhost:5000
```

**In MLflow UI:**
1. Compare runs side-by-side
2. Find the best model (highest test_f1)
3. Promote it to "Production" stage

```bash
# Start API with MLflow model
uvicorn src.serve_mlflow:app --reload --port 8000
```

### Chapter 4: Ensure Feature Consistency with Feast

**Goal:** Eliminate training-serving skew with a feature store.

```bash
# Initialize Feast (if not already done)
cd feature_repo
feast init . --minimal
cd ..

# Prepare features
python src/prepare_feast_features.py

# Test feature retrieval
python src/feast_features.py
```

**What you get:**
- ✅ Same feature definitions for training and serving
- ✅ Point-in-time correct joins (no data leakage)
- ✅ Low-latency online serving

### Chapter 5: Add Data Validation with Great Expectations

**Goal:** Reject invalid input data before making predictions.

```bash
# Test validation logic
python src/data_validation.py

# Start validated API
uvicorn src.serve_validated:app --reload --port 8000

# Test with bad data (now returns HTTP 400)
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"amount": -500, "hour": 25, "day_of_week": 10, "merchant_category": "fake"}'
```

**Response:**
```json
{
  "detail": {
    "message": "Validation failed",
    "errors": [
      "amount must be positive",
      "hour must be between 0 and 23 (got 25)",
      "day_of_week must be between 0 (Monday) and 6 (Sunday) (got 10)",
      "merchant_category must be one of ['grocery', 'restaurant', 'retail', 'online', 'travel'] (got 'fake')"
    ]
  }
}
```

### Chapter 6: Monitor Model Performance and Data Drift

**Goal:** Detect when model performance degrades.

```bash
# Run drift detection simulation
python src/monitoring.py

# Open drift_report.html in browser to see visualizations
```

**Scenarios tested:**
- ✅ Similar data (test set) → No drift
- ⚠️ Fraud spike (10% fraud) → Drift detected
- ⚠️ Amount inflation (2x) → Drift detected
- ⚠️ Time shift (late night) → Drift detected

### Chapter 7: Automate Testing and Deployment with CI/CD

**Goal:** Automated testing and safe deployments.

```bash
# Run tests locally
pytest tests/test_data_and_model.py -v
pytest tests/test_api.py -v  # Requires API running

# Build Docker image
docker build -t fraud-detection-api .

# Run container
docker run -p 8000:8000 fraud-detection-api

# Test health check
curl http://localhost:8000/health
```

## 🔄 Running the Complete Pipeline

### Full End-to-End Workflow

```bash
# 1. Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Generate data
python src/generate_data.py

# 3. Prepare features (Feast)
python src/prepare_feast_features.py

# 4. Start MLflow server (Terminal 1)
mlflow server --host 0.0.0.0 --port 5000 \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlruns

# 5. Train models with experiment tracking (Terminal 2)
python src/train_mlflow.py

# 6. Promote best model to Production in MLflow UI
# Visit http://localhost:5000, go to Models tab, select best version, set stage to "Production"

# 7. Start validated API (Terminal 2)
uvicorn src.serve_validated:app --reload --port 8000

# 8. Test API (Terminal 3)
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"amount": 150.0, "hour": 14, "day_of_week": 3, "merchant_category": "online"}'

# 9. Run drift monitoring
python src/monitoring.py

# 10. Run tests
pytest tests/ -v
```

## 🧪 Testing

### Run All Tests

```bash
# Data quality and model performance tests
pytest tests/test_data_and_model.py -v

# API tests (requires API running on port 8000)
pytest tests/test_api.py -v

# Run all tests with coverage
pytest tests/ -v --cov=src --cov-report=html
```

### Manual Testing

```bash
# Valid transaction (low risk)
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"amount": 50.0, "hour": 14, "day_of_week": 3, "merchant_category": "grocery"}'

# High-risk transaction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"amount": 500.0, "hour": 3, "day_of_week": 1, "merchant_category": "online"}'

# Invalid transaction (should return 400)
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"amount": -100.0, "hour": 25, "day_of_week": 10, "merchant_category": "fake"}'
```

## 🐳 Docker Deployment

### Build and Run

```bash
# Build image
docker build -t fraud-detection-api .

# Run container
docker run -d -p 8000:8000 --name fraud-api fraud-detection-api

# Check logs
docker logs fraud-api

# Stop container
docker stop fraud-api
docker rm fraud-api
```

### Docker Compose (Optional)

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  mlflow:
    image: ghcr.io/mlflow/mlflow:latest
    ports:
      - "5000:5000"
    command: >
      mlflow server
      --host 0.0.0.0
      --port 5000
      --backend-store-uri sqlite:///mlflow/mlflow.db
      --default-artifact-root /mlflow/artifacts
    volumes:
      - mlflow_data:/mlflow

  api:
    build: .
    ports:
      - "8000:8000"
    depends_on:
      - mlflow
    environment:
      - MLFLOW_TRACKING_URI=http://mlflow:5000

volumes:
  mlflow_data:
```

Run with: `docker-compose up`

## 🔧 Troubleshooting

### Common Issues

**Issue: MLflow model not found**
```
Error: No model found with name 'fraud-detection-model' in stage 'Production'
```
**Solution:** Promote a model to Production stage in MLflow UI (http://localhost:5000)

**Issue: Feast features not found**
```
Error: Feature view 'merchant_stats' not found
```
**Solution:** Run `python src/prepare_feast_features.py` to materialize features

**Issue: API tests fail with connection error**
```
httpx.ConnectError: [Errno 61] Connection refused
```
**Solution:** Make sure the API is running on port 8000 before running tests

**Issue: Docker build fails**
```
ERROR: failed to solve: failed to compute cache key
```
**Solution:** Ensure `models/model.pkl` exists by running `python src/train_naive.py` first

**Issue: Import errors in tests**
```
ModuleNotFoundError: No module named 'src'
```
**Solution:** Run tests from project root directory, not from `tests/` directory

### Performance Tips

1. **Speed up training:** Reduce `n_estimators` in training scripts
2. **Reduce memory usage:** Use smaller dataset in `generate_data.py`
3. **Faster API startup:** Use `--workers 1` flag with uvicorn
4. **Docker optimization:** Use multi-stage builds for smaller images

## 📊 Key Metrics

After running the complete pipeline, you should see:

| Metric | Expected Value | Notes |
|--------|---------------|-------|
| **Accuracy** | ~98% | High due to class imbalance |
| **F1-Score** | 0.5-0.7 | More meaningful for imbalanced data |
| **Precision** | 0.6-0.8 | Legitimate transactions flagged as fraud |
| **Recall** | 0.5-0.7 | Fraud transactions caught |
| **API Latency** | <50ms | p95 response time |
| **Drift Detection** | <10% | On similar data |

## 🎓 Learning Outcomes

By completing this tutorial, you will:

✅ Understand the gap between notebook ML and production ML  
✅ Track experiments and compare models systematically  
✅ Version models with proper governance  
✅ Eliminate training-serving skew with feature stores  
✅ Validate data before making predictions  
✅ Detect model drift before it causes problems  
✅ Deploy ML systems with Docker  
✅ Automate testing with CI/CD  

## 🚀 Next Steps: Scaling to Production

To scale this to production:

1. **Replace SQLite with PostgreSQL** for MLflow backend
2. **Use Redis or DynamoDB** for Feast online store
3. **Deploy to Kubernetes** with KServe or Seldon
4. **Add Prometheus + Grafana** for real-time monitoring
5. **Implement canary deployments** for safe rollouts
6. **Set up PagerDuty/Slack alerts** for drift detection
7. **Add A/B testing** to compare model versions
8. **Implement model retraining pipelines** (Airflow, Prefect)

## 📚 Additional Resources

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Feast Documentation](https://docs.feast.dev/)
- [Great Expectations Documentation](https://docs.greatexpectations.io/)
- [Evidently Documentation](https://docs.evidentlyai.com/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

## 📝 License

This project is for educational purposes. Feel free to use and modify for your own learning.

## 🤝 Contributing

This is a tutorial project. If you find issues or have improvements, feel free to create a pull request.

---

**Built with ❤️ for the ML community**
