"""
Data validation for fraud detection.

This module provides functions to validate input data BEFORE making predictions.
Invalid data is rejected with clear error messages.

The key insight: It's better to reject bad input than to make garbage predictions.
"""
import pandas as pd
from typing import Dict, List, Any, Optional

VALID_CATEGORIES = ["grocery", "restaurant", "retail", "online", "travel"]

def validate_transaction(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate a single transaction for fraud prediction.
    
    Checks all business rules and data quality requirements.
    Returns a dictionary with 'valid' (bool) and 'errors' (list).
    
    Args:
        data: Dictionary with transaction fields
        
    Returns:
        {"valid": bool, "errors": list of error messages}
        
    Example:
        >>> validate_transaction({"amount": -100, "hour": 25, ...})
        {"valid": False, "errors": ["amount must be positive", "hour must be 0-23"]}
    """
    errors = []
    
    amount = data.get("amount")
    if amount is None:
        errors.append("amount is required")
    elif not isinstance(amount, (int, float)):
        errors.append(f"amount must be a number (got {type(amount).__name__})")
    elif amount <= 0:
        errors.append("amount must be positive")
    elif amount > 50000:
        errors.append(f"amount exceeds maximum allowed value of $50,000 (got ${amount:,.2f})")
    
    hour = data.get("hour")
    if hour is None:
        errors.append("hour is required")
    elif not isinstance(hour, int):
        errors.append(f"hour must be an integer (got {type(hour).__name__})")
    elif not (0 <= hour <= 23):
        errors.append(f"hour must be between 0 and 23 (got {hour})")
    
    day = data.get("day_of_week")
    if day is None:
        errors.append("day_of_week is required")
    elif not isinstance(day, int):
        errors.append(f"day_of_week must be an integer (got {type(day).__name__})")
    elif not (0 <= day <= 6):
        errors.append(f"day_of_week must be between 0 (Monday) and 6 (Sunday) (got {day})")
    
    category = data.get("merchant_category")
    if category is None:
        errors.append("merchant_category is required")
    elif not isinstance(category, str):
        errors.append(f"merchant_category must be a string (got {type(category).__name__})")
    elif category not in VALID_CATEGORIES:
        errors.append(
            f"merchant_category must be one of {VALID_CATEGORIES} (got '{category}')"
        )
    
    return {
        "valid": len(errors) == 0,
        "errors": errors
    }

def validate_batch(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate a batch of transactions using Great Expectations.
    
    This is useful for validating training data or batch prediction requests.
    Uses Great Expectations for more sophisticated validation.
    
    Args:
        df: DataFrame with transaction data
        
    Returns:
        Dictionary with validation results
    """
    import great_expectations as gx
    
    ge_df = gx.from_pandas(df)
    
    results = []
    
    r = ge_df.expect_column_values_to_be_between(
        'amount', min_value=0.01, max_value=50000, mostly=0.99
    )
    results.append(('amount_range', r.success, r.result))
    
    r = ge_df.expect_column_values_to_be_between(
        'hour', min_value=0, max_value=23
    )
    results.append(('hour_range', r.success, r.result))
    
    r = ge_df.expect_column_values_to_be_between(
        'day_of_week', min_value=0, max_value=6
    )
    results.append(('day_range', r.success, r.result))
    
    r = ge_df.expect_column_values_to_be_in_set(
        'merchant_category', VALID_CATEGORIES
    )
    results.append(('category_valid', r.success, r.result))
    
    for col in ['amount', 'hour', 'day_of_week', 'merchant_category']:
        r = ge_df.expect_column_values_to_not_be_null(col)
        results.append((f'{col}_not_null', r.success, r.result))
    
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    return {
        'success': passed == total,
        'passed': passed,
        'total': total,
        'pass_rate': passed / total,
        'details': {name: {'passed': success, 'result': result} 
                   for name, success, result in results}
    }

if __name__ == "__main__":
    print("="*60)
    print("TESTING DATA VALIDATION")
    print("="*60)
    
    print("\n1. Single Transaction Validation")
    print("-"*40)
    
    test_cases = [
        {
            "name": "Valid transaction",
            "data": {"amount": 50.0, "hour": 14, "day_of_week": 3, "merchant_category": "grocery"}
        },
        {
            "name": "Negative amount",
            "data": {"amount": -100.0, "hour": 14, "day_of_week": 3, "merchant_category": "grocery"}
        },
        {
            "name": "Invalid hour",
            "data": {"amount": 50.0, "hour": 25, "day_of_week": 3, "merchant_category": "grocery"}
        },
        {
            "name": "Unknown merchant",
            "data": {"amount": 50.0, "hour": 14, "day_of_week": 3, "merchant_category": "unknown"}
        },
        {
            "name": "Everything wrong",
            "data": {"amount": -999, "hour": 99, "day_of_week": 15, "merchant_category": "fake"}
        },
    ]
    
    for tc in test_cases:
        result = validate_transaction(tc["data"])
        status = "PASS" if result["valid"] else "FAIL"
        print(f"\n{tc['name']}: {status}")
        if result["errors"]:
            for error in result["errors"]:
                print(f"  - {error}")
    
    print("\n\n2. Batch Validation with Great Expectations")
    print("-"*40)
    
    train_df = pd.read_csv('data/train.csv')
    results = validate_batch(train_df)
    
    print(f"\nTraining data validation: {results['passed']}/{results['total']} checks passed")
    print(f"Pass rate: {results['pass_rate']:.1%}")
    
    if not results['success']:
        print("\nFailed checks:")
        for name, detail in results['details'].items():
            if not detail['passed']:
                print(f"  - {name}")
