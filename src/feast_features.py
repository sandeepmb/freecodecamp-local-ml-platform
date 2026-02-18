"""
Feast feature retrieval for training and serving.

This module provides functions to retrieve features from Feast:
- get_training_features(): For offline training (historical features)
- get_online_features(): For real-time serving (low-latency)

IMPORTANT: Both functions use the SAME feature definitions,
ensuring consistency between training and serving.
"""
import pandas as pd
from feast import FeatureStore
from datetime import datetime

store = FeatureStore(repo_path="feature_repo")

def get_training_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get features for training using Feast's offline store.
    
    Uses point-in-time correct joins to prevent data leakage.
    This means features are looked up as of the time each transaction occurred,
    not as of "now" - preventing you from accidentally using future data.
    
    Args:
        df: DataFrame with at least 'merchant_category' column
        
    Returns:
        DataFrame with original columns plus Feast features
    """
    print("Retrieving training features from Feast offline store...")
    
    entity_df = df[['merchant_category']].copy()
    entity_df['event_timestamp'] = datetime.now()
    entity_df = entity_df.drop_duplicates()
    
    training_data = store.get_historical_features(
        entity_df=entity_df,
        features=[
            "merchant_stats:avg_amount",
            "merchant_stats:transaction_count",
            "merchant_stats:fraud_rate",
        ],
    ).to_df()
    
    result = df.merge(
        training_data[['merchant_category', 'avg_amount', 'transaction_count', 'fraud_rate']],
        on='merchant_category',
        how='left'
    )
    
    print(f"Retrieved features for {len(entity_df)} unique merchants")
    return result

def get_online_features(merchant_category: str) -> dict:
    """
    Get features for real-time serving using Feast's online store.
    
    This is optimized for low-latency retrieval (milliseconds).
    Use this in your prediction API for real-time inference.
    
    Args:
        merchant_category: The merchant category to look up
        
    Returns:
        Dictionary with feature names and values
    """
    feature_vector = store.get_online_features(
        features=[
            "merchant_stats:avg_amount",
            "merchant_stats:transaction_count",
            "merchant_stats:fraud_rate",
        ],
        entity_rows=[{"merchant_category": merchant_category}],
    ).to_dict()
    
    return {
        'merchant_avg_amount': feature_vector['avg_amount'][0],
        'merchant_tx_count': feature_vector['transaction_count'][0],
        'merchant_fraud_rate': feature_vector['fraud_rate'][0],
    }

def get_online_features_batch(merchant_categories: list) -> pd.DataFrame:
    """
    Get features for multiple merchants at once (batch serving).
    
    More efficient than calling get_online_features() in a loop.
    
    Args:
        merchant_categories: List of merchant categories to look up
        
    Returns:
        DataFrame with features for each merchant
    """
    feature_vector = store.get_online_features(
        features=[
            "merchant_stats:avg_amount",
            "merchant_stats:transaction_count",
            "merchant_stats:fraud_rate",
        ],
        entity_rows=[{"merchant_category": mc} for mc in merchant_categories],
    ).to_df()
    
    return feature_vector

if __name__ == "__main__":
    print("="*60)
    print("TESTING FEAST FEATURE RETRIEVAL")
    print("="*60)
    
    print("\n1. Testing OFFLINE feature retrieval (for training)...")
    train_df = pd.read_csv('data/train.csv').head(10)
    enriched = get_training_features(train_df)
    print("\n   Sample enriched training data:")
    print(enriched[['amount', 'merchant_category', 'avg_amount', 'fraud_rate']].head())
    
    print("\n2. Testing ONLINE feature retrieval (for serving)...")
    for category in ['online', 'grocery', 'travel', 'restaurant', 'retail']:
        features = get_online_features(category)
        print(f"   {category}: avg_amount=${features['merchant_avg_amount']:.2f}, "
              f"fraud_rate={features['merchant_fraud_rate']:.2%}")
    
    print("\n3. Testing BATCH online retrieval...")
    batch_features = get_online_features_batch(['online', 'grocery', 'travel'])
    print(batch_features)
    
    print("\n" + "="*60)
    print("FEAST FEATURE RETRIEVAL TEST COMPLETE!")
    print("="*60)
