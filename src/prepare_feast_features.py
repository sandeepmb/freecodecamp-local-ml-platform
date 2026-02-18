"""
Prepare feature data for Feast.

This script:
1. Computes aggregated merchant features from training data
2. Saves them in Parquet format (Feast's offline store format)
3. Applies Feast feature definitions
4. Materializes features to the online store for low-latency serving

Run this whenever your training data changes or you want to refresh features.
"""
import pandas as pd
import numpy as np
from datetime import datetime
import subprocess
import os

def compute_merchant_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute aggregated features by merchant category.
    
    THIS IS THE SINGLE SOURCE OF TRUTH FOR FEATURE COMPUTATION.
    
    Both training and serving will use features computed by this exact logic.
    Any change here automatically applies everywhere.
    
    Args:
        df: Transaction DataFrame with columns: amount, merchant_category, is_fraud
        
    Returns:
        DataFrame with computed features per merchant category
    """
    print("Computing merchant-level features...")
    
    stats = df.groupby('merchant_category').agg({
        'amount': ['mean', 'count'],
        'is_fraud': 'mean'
    }).reset_index()
    
    stats.columns = ['merchant_category', 'avg_amount', 'transaction_count', 'fraud_rate']
    
    stats['event_timestamp'] = datetime.now()
    
    stats['avg_amount'] = stats['avg_amount'].astype('float32')
    stats['transaction_count'] = stats['transaction_count'].astype('int64')
    stats['fraud_rate'] = stats['fraud_rate'].astype('float32')
    
    return stats

def main():
    print("="*60)
    print("FEAST FEATURE PREPARATION")
    print("="*60)
    
    print("\n1. Loading training data...")
    train_df = pd.read_csv('data/train.csv')
    print(f"   Loaded {len(train_df):,} transactions")
    
    print("\n2. Computing merchant features...")
    merchant_features = compute_merchant_features(train_df)
    
    print("\n   Computed features:")
    print(merchant_features.to_string(index=False))
    
    print("\n3. Saving features to Parquet...")
    os.makedirs('data', exist_ok=True)
    output_path = 'data/merchant_features.parquet'
    merchant_features.to_parquet(output_path, index=False)
    print(f"   Saved to {output_path}")
    
    print("\n4. Applying Feast feature definitions...")
    try:
        result = subprocess.run(
            ['feast', 'apply'],
            cwd='feature_repo',
            capture_output=True,
            text=True,
            check=True
        )
        print("   Feature definitions applied successfully!")
        if result.stdout:
            print(f"   {result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"   Error applying Feast: {e.stderr}")
        raise
    
    print("\n5. Materializing features to online store...")
    try:
        result = subprocess.run(
            ['feast', 'materialize-incremental', datetime.now().isoformat()],
            cwd='feature_repo',
            capture_output=True,
            text=True,
            check=True
        )
        print("   Features materialized successfully!")
        if result.stdout:
            print(f"   {result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"   Error materializing: {e.stderr}")
        raise
    
    print("\n" + "="*60)
    print("FEAST FEATURE PREPARATION COMPLETE!")
    print("="*60)
    print("\nYou can now:")
    print("  - Retrieve features for training: get_training_features()")
    print("  - Retrieve features for serving: get_online_features()")
    print("  - View feature stats: feast feature-views list")

if __name__ == "__main__":
    main()
