"""
Tests for data quality and model performance.

These tests run in CI/CD to ensure:
1. Data meets quality requirements
2. Model meets performance thresholds
3. No regressions are introduced

Run with: pytest tests/test_data_and_model.py -v
"""
import pandas as pd
import pickle
import pytest
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

class TestDataQuality:
    
    @pytest.fixture
    def train_data(self):
        return pd.read_csv("data/train.csv")
    
    @pytest.fixture
    def test_data(self):
        return pd.read_csv("data/test.csv")
    
    def test_train_data_has_expected_columns(self, train_data):
        required_columns = {"amount", "hour", "day_of_week", "merchant_category", "is_fraud"}
        actual_columns = set(train_data.columns)
        missing = required_columns - actual_columns
        assert not missing, f"Missing columns: {missing}"
    
    def test_train_data_not_empty(self, train_data):
        assert len(train_data) > 0, "Training data is empty"
        assert len(train_data) >= 1000, f"Training data too small: {len(train_data)} rows"
    
    def test_no_negative_amounts(self, train_data):
        negative_count = (train_data["amount"] < 0).sum()
        assert negative_count == 0, f"Found {negative_count} negative amounts"
    
    def test_amounts_reasonable(self, train_data):
        max_amount = train_data["amount"].max()
        assert max_amount <= 100000, f"Max amount {max_amount} exceeds reasonable limit"
    
    def test_hours_valid(self, train_data):
        invalid = train_data[(train_data["hour"] < 0) | (train_data["hour"] > 23)]
        assert len(invalid) == 0, f"Found {len(invalid)} invalid hours"
    
    def test_days_valid(self, train_data):
        invalid = train_data[(train_data["day_of_week"] < 0) | (train_data["day_of_week"] > 6)]
        assert len(invalid) == 0, f"Found {len(invalid)} invalid days"
    
    def test_merchant_categories_valid(self, train_data):
        valid_categories = {"grocery", "restaurant", "retail", "online", "travel"}
        actual_categories = set(train_data["merchant_category"].unique())
        invalid = actual_categories - valid_categories
        assert not invalid, f"Invalid merchant categories: {invalid}"
    
    def test_fraud_ratio_reasonable(self, train_data):
        fraud_ratio = train_data["is_fraud"].mean()
        assert 0.001 <= fraud_ratio <= 0.5, f"Fraud ratio {fraud_ratio:.2%} is unrealistic"
    
    def test_no_nulls_in_critical_columns(self, train_data):
        critical = ["amount", "hour", "day_of_week", "merchant_category", "is_fraud"]
        for col in critical:
            null_count = train_data[col].isnull().sum()
            assert null_count == 0, f"Column {col} has {null_count} null values"


class TestModelPerformance:
    
    @pytest.fixture
    def model_and_encoder(self):
        with open("models/model.pkl", "rb") as f:
            return pickle.load(f)
    
    @pytest.fixture
    def test_data(self):
        return pd.read_csv("data/test.csv")
    
    def test_model_loads_successfully(self, model_and_encoder):
        model, encoder = model_and_encoder
        assert model is not None, "Model is None"
        assert encoder is not None, "Encoder is None"
    
    def test_model_can_predict(self, model_and_encoder, test_data):
        model, encoder = model_and_encoder
        test_data["merchant_encoded"] = encoder.transform(test_data["merchant_category"])
        X = test_data[["amount", "hour", "day_of_week", "merchant_encoded"]]
        predictions = model.predict(X)
        assert len(predictions) == len(X), "Prediction count mismatch"
    
    def test_accuracy_threshold(self, model_and_encoder, test_data):
        model, encoder = model_and_encoder
        test_data["merchant_encoded"] = encoder.transform(test_data["merchant_category"])
        X = test_data[["amount", "hour", "day_of_week", "merchant_encoded"]]
        y = test_data["is_fraud"]
        accuracy = model.score(X, y)
        assert accuracy >= 0.90, f"Accuracy {accuracy:.2%} below 90% threshold"
    
    def test_f1_threshold(self, model_and_encoder, test_data):
        model, encoder = model_and_encoder
        test_data["merchant_encoded"] = encoder.transform(test_data["merchant_category"])
        X = test_data[["amount", "hour", "day_of_week", "merchant_encoded"]]
        y = test_data["is_fraud"]
        y_pred = model.predict(X)
        f1 = f1_score(y, y_pred)
        assert f1 >= 0.3, f"F1-score {f1:.2f} below 0.3 threshold"
    
    def test_precision_not_zero(self, model_and_encoder, test_data):
        model, encoder = model_and_encoder
        test_data["merchant_encoded"] = encoder.transform(test_data["merchant_category"])
        X = test_data[["amount", "hour", "day_of_week", "merchant_encoded"]]
        y = test_data["is_fraud"]
        y_pred = model.predict(X)
        precision = precision_score(y, y_pred, zero_division=0)
        assert precision > 0, "Model has zero precision (predicts no fraud)"
    
    def test_recall_not_zero(self, model_and_encoder, test_data):
        model, encoder = model_and_encoder
        test_data["merchant_encoded"] = encoder.transform(test_data["merchant_category"])
        X = test_data[["amount", "hour", "day_of_week", "merchant_encoded"]]
        y = test_data["is_fraud"]
        y_pred = model.predict(X)
        recall = recall_score(y, y_pred, zero_division=0)
        assert recall > 0, "Model has zero recall (misses all fraud)"
