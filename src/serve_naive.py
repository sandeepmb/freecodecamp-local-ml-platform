"""
Serve fraud detection model as a REST API - NAIVE VERSION.

This is a simple API that:
1. Loads the trained model at startup
2. Accepts transaction data via POST request
3. Returns fraud prediction

We'll improve this with validation, monitoring, and better
model loading in later sections.
"""
import pickle
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Optional

print("Loading model...")
with open("models/model.pkl", "rb") as f:
    model, encoder = pickle.load(f)
print("Model loaded successfully!")

app = FastAPI(
    title="Fraud Detection API",
    description="""
    Predict whether a credit card transaction is fraudulent.
    
    This API accepts transaction details and returns:
    - Whether the transaction is predicted to be fraud
    - The probability of fraud (0.0 to 1.0)
    
    **Note:** This is the naive version without validation or monitoring.
    """,
    version="1.0.0"
)

class Transaction(BaseModel):
    amount: float = Field(
        ..., 
        description="Transaction amount in dollars",
        example=150.00
    )
    hour: int = Field(
        ..., 
        description="Hour of the day (0-23)",
        example=14
    )
    day_of_week: int = Field(
        ..., 
        description="Day of week (0=Monday, 6=Sunday)",
        example=3
    )
    merchant_category: str = Field(
        ..., 
        description="Type of merchant",
        example="online"
    )

class PredictionResponse(BaseModel):
    is_fraud: bool = Field(description="Whether the transaction is predicted as fraud")
    fraud_probability: float = Field(description="Probability of fraud (0.0 to 1.0)")
    
@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: Transaction):
    """
    Predict whether a transaction is fraudulent.
    
    Takes transaction details and returns a fraud prediction
    along with the probability score.
    """
    data = transaction.dict()
    
    try:
        data["merchant_encoded"] = encoder.transform([data["merchant_category"]])[0]
    except ValueError:
        data["merchant_encoded"] = 0
    
    X = [[
        data["amount"],
        data["hour"],
        data["day_of_week"],
        data["merchant_encoded"]
    ]]
    
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0][1]
    
    return PredictionResponse(
        is_fraud=bool(prediction),
        fraud_probability=round(float(probability), 4)
    )

@app.get("/health")
def health_check():
    """
    Health check endpoint.
    
    Returns the status of the API. Useful for:
    - Load balancer health checks
    - Kubernetes liveness probes
    - Monitoring systems
    """
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

@app.get("/")
def root():
    return {
        "message": "Fraud Detection API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }
