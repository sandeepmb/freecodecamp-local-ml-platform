"""
Serve fraud detection model with input validation.

This version adds data validation BEFORE making predictions:
- Invalid inputs are rejected with HTTP 400 and clear error messages
- Valid inputs are processed and predictions returned

This is much safer than the naive version which accepted garbage.
"""
import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.data_validation import validate_transaction

with open("models/model.pkl", "rb") as f:
    model, encoder = pickle.load(f)

app = FastAPI(
    title="Fraud Detection API (Validated)",
    description="""
    Fraud detection API with input validation.
    
    All inputs are validated before prediction:
    - amount: Must be positive and below $50,000
    - hour: Must be 0-23
    - day_of_week: Must be 0-6
    - merchant_category: Must be one of: grocery, restaurant, retail, online, travel
    
    Invalid inputs return HTTP 400 with detailed error messages.
    """,
    version="3.0.0"
)

class Transaction(BaseModel):
    amount: float = Field(..., description="Transaction amount (must be positive)", example=150.00)
    hour: int = Field(..., description="Hour of day (0-23)", example=14)
    day_of_week: int = Field(..., description="Day of week (0=Mon, 6=Sun)", example=3)
    merchant_category: str = Field(..., description="Merchant type", example="online")

class PredictionResponse(BaseModel):
    is_fraud: bool
    fraud_probability: float
    validation_passed: bool = True

class ValidationErrorResponse(BaseModel):
    detail: dict

@app.post("/predict", response_model=PredictionResponse, responses={400: {"model": ValidationErrorResponse}})
def predict(tx: Transaction):
    """
    Predict whether a transaction is fraudulent.
    
    Input is validated before prediction. Invalid inputs return HTTP 400.
    """
    data = tx.dict()
    
    validation = validate_transaction(data)
    
    if not validation["valid"]:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Validation failed",
                "errors": validation["errors"],
                "input": data
            }
        )
    
    data["merchant_encoded"] = encoder.transform([data["merchant_category"]])[0]
    X = [[data["amount"], data["hour"], data["day_of_week"], data["merchant_encoded"]]]
    
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0][1]
    
    return PredictionResponse(
        is_fraud=bool(pred),
        fraud_probability=round(float(prob), 4),
        validation_passed=True
    )

@app.get("/health")
def health():
    return {"status": "healthy", "validation": "enabled"}
