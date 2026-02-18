"""
Generate synthetic fraud detection dataset.

This script creates realistic-looking transaction data where fraudulent
transactions have different patterns than legitimate ones:
- Fraud tends to have higher amounts
- Fraud tends to occur late at night
- Fraud is more common for online and travel merchants
"""
import pandas as pd
import numpy as np
from pathlib import Path

def generate_transactions(n_samples=10000, fraud_ratio=0.02, seed=42):
    """
    Generate synthetic fraud detection dataset.
    
    Args:
        n_samples: Total number of transactions to generate
        fraud_ratio: Proportion of fraudulent transactions (default 2%)
        seed: Random seed for reproducibility
    
    Returns:
        DataFrame with transaction features and fraud labels
    
    Fraud transactions have different patterns:
    - Higher amounts (mean $245 vs $33 for legit)
    - Late night hours (0-5, 23)
    - More likely to be online or travel merchants
    """
    np.random.seed(seed)
    n_fraud = int(n_samples * fraud_ratio)
    n_legit = n_samples - n_fraud

    legit = pd.DataFrame({
        "amount": np.random.lognormal(mean=3.5, sigma=1.2, size=n_legit),
        "hour": np.random.randint(0, 24, size=n_legit),
        "day_of_week": np.random.randint(0, 7, size=n_legit),
        "merchant_category": np.random.choice(
            ["grocery", "restaurant", "retail", "online", "travel"],
            size=n_legit,
            p=[0.30, 0.25, 0.25, 0.15, 0.05]
        ),
        "is_fraud": 0
    })
    
    fraud = pd.DataFrame({
        "amount": np.random.lognormal(mean=5.5, sigma=1.5, size=n_fraud),
        "hour": np.random.choice([0, 1, 2, 3, 4, 5, 23], size=n_fraud),
        "day_of_week": np.random.randint(0, 7, size=n_fraud),
        "merchant_category": np.random.choice(
            ["grocery", "restaurant", "retail", "online", "travel"],
            size=n_fraud,
            p=[0.05, 0.05, 0.10, 0.60, 0.20]
        ),
        "is_fraud": 1
    })
    
    df = pd.concat([legit, fraud], ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    return df

if __name__ == "__main__":
    print("Generating synthetic fraud detection dataset...")
    df = generate_transactions(n_samples=10000, fraud_ratio=0.02)
    
    train_df = df.sample(frac=0.8, random_state=42)
    test_df = df.drop(train_df.index)
    
    # Create data directory if it doesn't exist
    Path("data").mkdir(parents=True, exist_ok=True)
    
    train_df.to_csv("data/train.csv", index=False)
    test_df.to_csv("data/test.csv", index=False)
    
    print(f"\nDataset generated successfully!")
    print(f"Training set: {len(train_df):,} transactions")
    print(f"Test set: {len(test_df):,} transactions")
    print(f"Overall fraud ratio: {df['is_fraud'].mean():.2%}")
    print(f"\nLegitimate transactions - Average amount: ${df[df['is_fraud']==0]['amount'].mean():.2f}")
    print(f"Fraudulent transactions - Average amount: ${df[df['is_fraud']==1]['amount'].mean():.2f}")
    print(f"\nMerchant category distribution (fraud):")
    print(df[df['is_fraud']==1]['merchant_category'].value_counts(normalize=True))
