# Location-Based Forecasting Model for POWERGRID Inventory

import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

def train_location_based_model(features_df, target='quantity', test_size=0.2, random_state=42):
    """
    Train a location-based forecasting model using LightGBM

    Args:
        features_df: DataFrame with features including location, tower_type, substation_type
        target: Target column name (default: 'quantity')
        test_size: Fraction of data for testing
        random_state: Random state for reproducibility

    Returns:
        dict: Trained model and evaluation metrics
    """

    print("🏗️ Training Location-Based Forecasting Model...")
    print(f"📊 Dataset shape: {features_df.shape}")
    print(f"🎯 Target: {target}")

    # Define location-based features
    location_features = [
        # Location features
        'location_demand_multiplier', 'infrastructure_density_index',
        'monsoon_sensitive_location',

        # Project type features
        'tower_complexity_factor', 'substation_material_factor',
        'project_complexity_score',

        # Budget features
        'budget_category_small', 'budget_category_medium', 'budget_category_large',
        'budget_category_mega', 'budget_category_ultra_mega',
        'budget_utilization_rate', 'budget_intensity', 'budget_allocation_efficiency',

        # Categorical features (will be handled by LightGBM)
        'location', 'tower_type', 'substation_type',

        # Temporal features
        'month', 'quarter', 'seasonal_multiplier', 'monsoon_flag',

        # Material features
        'material_id',

        # Lag features (if available)
        'quantity_lag_1', 'quantity_lag_7', 'quantity_lag_30',
        'rolling_mean_7', 'rolling_mean_30', 'rolling_std_7'
    ]

    # Filter to available features
    available_features = [col for col in location_features if col in features_df.columns]
    print(f"✅ Using {len(available_features)} location-based features")

    # Prepare data
    X = features_df[available_features].copy()
    y = features_df[target].copy()

    # Handle categorical features for LightGBM
    categorical_features = ['location', 'tower_type', 'substation_type', 'material_id']
    categorical_features = [col for col in categorical_features if col in X.columns]

    print(f"📋 Categorical features: {categorical_features}")

    # Convert categorical columns to category dtype
    for col in categorical_features:
        X[col] = X[col].astype('category')

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=True
    )

    print(f"📊 Train set: {X_train.shape}, Test set: {X_test.shape}")

    # Create LightGBM datasets
    train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_features)
    test_data = lgb.Dataset(X_test, label=y_test, categorical_feature=categorical_features, reference=train_data)

    # Model parameters optimized for location-based forecasting
    params = {
        'objective': 'regression',
        'metric': 'mae',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'random_state': random_state,
        'n_estimators': 1000,
        'early_stopping_rounds': 50,
        'categorical_feature': categorical_features
    }

    print("🚀 Training LightGBM model...")

    # Train model with early stopping
    model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, test_data],
        valid_names=['train', 'valid'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=100)
        ]
    )

    # Make predictions
    y_pred_train = model.predict(X_train, num_iteration=model.best_iteration)
    y_pred_test = model.predict(X_test, num_iteration=model.best_iteration)

    # Calculate metrics
    metrics = {
        'train': {
            'mae': mean_absolute_error(y_train, y_pred_train),
            'rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'r2': r2_score(y_train, y_pred_train)
        },
        'test': {
            'mae': mean_absolute_error(y_test, y_pred_test),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'r2': r2_score(y_test, y_pred_test)
        }
    }

    print("📊 Model Performance:")
    print(f"  Train MAE: {metrics['train']['mae']:.3f}")
    print(f"  Train RMSE: {metrics['train']['rmse']:.3f}")
    print(f"  Train R²: {metrics['train']['r2']:.3f}")
    print("📊 Test Performance:")
    print(f"  Test MAE: {metrics['test']['mae']:.3f}")
    print(f"  Test RMSE: {metrics['test']['rmse']:.3f}")
    print(f"  Test R²: {metrics['test']['r2']:.3f}")
    # Feature importance
    feature_importance = dict(zip(model.feature_name(), model.feature_importance('gain')))
    feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))

    print("🎯 Top 10 Important Features:")
    for i, (feature, importance) in enumerate(list(feature_importance.items())[:10]):
        print(f"  {i+1}. {feature}: {importance:.3f}")
    # Save model and metadata
    model_results = save_location_model(model, metrics, feature_importance, available_features, categorical_features)

    print("✅ Location-based model training complete!")
    return model_results

def save_location_model(model, metrics, feature_importance, features_used, categorical_features):
    """
    Save the trained location model and metadata

    Args:
        model: Trained LightGBM model
        metrics: Performance metrics
        feature_importance: Feature importance dictionary
        features_used: List of features used
        categorical_features: List of categorical features

    Returns:
        dict: Model information
    """

    # Create model directory
    model_dir = Path("ml/models/location")
    model_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = model_dir / "location_model.pkl"
    joblib.dump(model, model_path)

    # Save model info
    model_info = {
        'model_type': 'lightgbm_location_based',
        'target': 'quantity',
        'features_used': features_used,
        'categorical_features': categorical_features,
        'metrics': metrics,
        'feature_importance': feature_importance,
        'training_date': pd.Timestamp.now().isoformat(),
        'model_path': str(model_path)
    }

    info_path = model_dir / "model_info.json"
    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=2, default=str)

    # Save feature names
    feature_names_path = model_dir / "feature_names.json"
    with open(feature_names_path, 'w') as f:
        json.dump(features_used, f, indent=2)

    # Save categorical columns
    cat_cols_path = model_dir / "categorical_columns.json"
    with open(cat_cols_path, 'w') as f:
        json.dump(categorical_features, f, indent=2)

    print(f"💾 Model saved to: {model_dir}")

    return {
        'model': model,
        'metrics': metrics,
        'feature_importance': feature_importance,
        'model_path': model_path,
        'model_info': model_info
    }

def load_location_model():
    """
    Load the trained location model

    Returns:
        dict: Model and metadata
    """
    model_dir = Path("ml/models/location")

    # Load model
    model_path = model_dir / "location_model.pkl"
    model = joblib.load(model_path)

    # Load metadata
    info_path = model_dir / "model_info.json"
    with open(info_path, 'r') as f:
        model_info = json.load(f)

    return {
        'model': model,
        'model_info': model_info
    }

def predict_with_location_model(features_df):
    """
    Make predictions using the trained location model

    Args:
        features_df: DataFrame with features

    Returns:
        np.array: Predictions
    """
    model_data = load_location_model()
    model = model_data['model']

    # Ensure categorical features are properly formatted
    categorical_features = model_data['model_info']['categorical_features']
    for col in categorical_features:
        if col in features_df.columns:
            features_df[col] = features_df[col].astype('category')

    # Use only the features the model was trained on
    features_used = model_data['model_info']['features_used']
    available_features = [col for col in features_used if col in features_df.columns]

    X = features_df[available_features]

    predictions = model.predict(X, num_iteration=model.best_iteration)

    return predictions

if __name__ == "__main__":
    print("Location-Based Forecasting Model")
    print("Use train_location.py to train the model")