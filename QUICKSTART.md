# Quick Start Guide

Get the ML platform running in 10 minutes!

## Prerequisites Check

```bash
# Check Python version (need 3.9+)
python --version

# Check Docker
docker --version

# Check available disk space (need 5GB+)
df -h .
```

## Step 1: Setup (2 minutes)

```bash
cd freecodecamp-local-ml-platform

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Step 2: Generate Data and Train Model (2 minutes)

```bash
# Create directories for generated artifacts (scripts will also create these)
mkdir -p data models

# Generate synthetic fraud detection data
python src/generate_data.py

# Train baseline model
python src/train_naive.py
```

**Expected output:**
- `data/train.csv` (8,000 transactions)
- `data/test.csv` (2,000 transactions)
- `models/model.pkl` (trained model)
- Accuracy: ~98%, F1-score: ~0.5-0.7

## Step 3: Start the API (1 minute)

```bash
# Start the validated API with data validation
uvicorn src.serve_validated:app --reload --host 0.0.0.0 --port 8000
```

**Verify:** Open http://localhost:8000/docs in your browser

## Step 4: Test the API (1 minute)

```bash
# In a new terminal - Test valid transaction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"amount": 150.0, "hour": 14, "day_of_week": 3, "merchant_category": "online"}'

# Expected response:
# {"is_fraud": false, "fraud_probability": 0.15, "validation_passed": true}

# Test invalid transaction (should return 400 error)
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"amount": -100, "hour": 25, "day_of_week": 10, "merchant_category": "fake"}'

# Expected: HTTP 400 with detailed error messages
```

## Step 5: Run MLflow Experiments (Optional, 3 minutes)

```bash
# Terminal 1: Start MLflow server
mlflow server --host 0.0.0.0 --port 5000 \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlruns

# Terminal 2: Run experiment sweep
python src/train_mlflow.py

# Open http://localhost:5000 to view experiments
```

## Step 6: Run Drift Monitoring (Optional, 1 minute)

```bash
python src/monitoring.py

# Open drift_report.html in your browser
```

## Common Commands

```bash
# Run all tests
pytest tests/ -v

# Build Docker image
docker build -t fraud-detection-api .

# Run Docker container
docker run -p 8000:8000 fraud-detection-api

# Check API health
curl http://localhost:8000/health

# View API documentation
open http://localhost:8000/docs
```

## What's Next?

1. **Explore the code:** Check `src/` directory for implementations
2. **Read the full README:** Detailed chapter-by-chapter guide
3. **Experiment:** Modify hyperparameters in training scripts
4. **Deploy:** Use Docker for containerized deployment
5. **Scale:** Follow production scaling guide in README

## Troubleshooting

**API won't start:**
```bash
# Check if port 8000 is already in use
lsof -i :8000
# Kill the process if needed
kill -9 <PID>
```

**Model not found:**
```bash
# Retrain the model
python src/train_naive.py
```

**Import errors:**
```bash
# Make sure you're in the project root
pwd  # Should end with ml-platform-tutorial
# Reinstall dependencies
pip install -r requirements.txt
```

**MLflow connection error:**
```bash
# Make sure MLflow server is running on port 5000
curl http://localhost:5000/health
```

## Success Criteria

✅ API responds to health check  
✅ Valid predictions return 200 status  
✅ Invalid inputs return 400 with error details  
✅ Model accuracy > 90%  
✅ F1-score > 0.3  
✅ All tests pass  

---

**Time to complete:** ~10 minutes  
**Difficulty:** Beginner-friendly  
**Support:** Check README.md for detailed documentation
