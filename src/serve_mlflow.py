"""
Serve fraud detection model from MLflow Model Registry.

This version loads the Production model from MLflow, which means:
- Always serves the latest Production model
- Can roll back by changing the Production stage
- No manual file copying needed
"""
import mlflow
import mlflow.sklearn
import pickle
import os
from fastapi import FastAPI
from pydantic import BaseModel, Field

mlflow.set_tracking_uri("http://localhost:5000")

print("Loading model from MLflow Model Registry...")

try:
    # Try loading with 'champion' alias (modern approach)
    model = mlflow.sklearn.load_model("models:/fraud-detection-model@champion")
    print("Successfully loaded 'champion' model from MLflow!")
except Exception as e:
    print(f"Could not load 'champion' alias: {e}")
    print("Trying to load latest version...")
    try:
        # Fallback to latest version
        model = mlflow.sklearn.load_model("models:/fraud-detection-model/latest")
        print("Successfully loaded latest model version from MLflow!")
    except Exception as e2:
        print(f"Error loading from MLflow: {e2}")
        print("Make sure you have registered a model in MLflow UI")
        print("You can set an alias with: mlflow.register_model(...)")
        raise

with open("encoder.pkl", "rb") as f:
    encoder = pickle.load(f)
print("Encoder loaded successfully!")

app = FastAPI(
    title="Fraud Detection API (MLflow)",
    description="""
    Fraud detection API that loads models from MLflow Model Registry.
    
    This version serves the model with 'champion' alias (or latest version).
    To update the model:
    1. Train a new model with train_mlflow.py
    2. Compare metrics in MLflow UI
    3. Set 'champion' alias on the best model version
    4. Restart this API
    
    To roll back: Move the 'champion' alias to a previous version in MLflow UI.
    """,
    version="2.0.0"
)

class Transaction(BaseModel):
    amount: float = Field(..., description="Transaction amount in dollars", example=150.00)
    hour: int = Field(..., description="Hour of the day (0-23)", example=14)
    day_of_week: int = Field(..., description="Day of week (0=Monday, 6=Sunday)", example=3)
    merchant_category: str = Field(..., description="Type of merchant", example="online")

class PredictionResponse(BaseModel):
    is_fraud: bool
    fraud_probability: float
    model_source: str = "MLflow Registry"

@app.post("/predict", response_model=PredictionResponse)
def predict(tx: Transaction):
    data = tx.dict()
    
    try:
        data["merchant_encoded"] = encoder.transform([data["merchant_category"]])[0]
    except ValueError:
        data["merchant_encoded"] = 0
    
    X = [[data["amount"], data["hour"], data["day_of_week"], data["merchant_encoded"]]]
    
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0][1]
    
    return PredictionResponse(
        is_fraud=bool(pred),
        fraud_probability=round(float(prob), 4),
        model_source="MLflow Registry"
    )

@app.get("/health")
def health():
    return {"status": "healthy", "model_source": "MLflow Registry"}

@app.get("/model-info")
def model_info():
    return {
        "registry": "MLflow",
        "model_name": "fraud-detection-model",
        "alias": "champion (or latest)",
        "tracking_uri": "http://localhost:5000"
    }
