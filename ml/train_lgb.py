import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import pickle
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_features():
    """Load feature data for training"""
    print("📥 Loading feature data...")

    # Try to load from parquet first, fallback to CSV
    features_path_parquet = Path("data/features/features.parquet")
    features_path_csv = Path("data/features/features.csv")

    if features_path_parquet.exists():
        df = pd.read_parquet(features_path_parquet)
        print(f"✅ Loaded {len(df)} records from parquet")
    elif features_path_csv.exists():
        df = pd.read_csv(features_path_csv)
        print(f"✅ Loaded {len(df)} records from CSV")
    else:
        raise FileNotFoundError("Features file not found in data/features/")

    return df

def prepare_data_for_lgb(df):
    """Prepare data for LightGBM training"""
    print("🔧 Preparing data for LightGBM...")

    # Convert date to datetime if not already
    df['date'] = pd.to_datetime(df['date'])

    # Sort by date for time series split
    df = df.sort_values('date').reset_index(drop=True)

    # Define target and features
    target_col = 'quantity'

    # Identify categorical and numeric columns
    exclude_cols = ['quantity', 'project_id', 'material_id', 'date']
    all_cols = [col for col in df.columns if col not in exclude_cols]

    # Separate categorical and numeric features
    categorical_cols = []
    numeric_cols = []

    for col in all_cols:
        if df[col].dtype == 'object':
            categorical_cols.append(col)
        elif df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
            numeric_cols.append(col)

    print(f"🎯 Target: {target_col}")
    print(f"🔢 Numeric features: {len(numeric_cols)}")
    print(f"📊 Categorical features: {len(categorical_cols)} ({', '.join(categorical_cols)})")

    # Label encode categorical features
    df_encoded = df.copy()
    label_encoders = {}

    for col in categorical_cols:
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        label_encoders[col] = le
        print(f"✅ Encoded {col}: {len(le.classes_)} unique values")

    # Prepare X and y
    feature_cols = numeric_cols + categorical_cols
    X = df_encoded[feature_cols]
    y = df_encoded[target_col]

    # Add material and date info for grouping predictions
    metadata = df[['material_id', 'date', 'quantity']].copy()

    return X, y, metadata, feature_cols, categorical_cols, label_encoders

def create_time_based_split(X, y, metadata, train_ratio=0.8):
    """Create time-based train/validation split"""
    print("⏰ Creating time-based train/validation split...")

    # Sort by date to ensure temporal order
    sorted_indices = metadata['date'].argsort()

    # Split point
    split_idx = int(len(sorted_indices) * train_ratio)

    train_indices = sorted_indices[:split_idx]
    val_indices = sorted_indices[split_idx:]

    X_train = X.iloc[train_indices]
    X_val = X.iloc[val_indices]
    y_train = y.iloc[train_indices]
    y_val = y.iloc[val_indices]

    metadata_train = metadata.iloc[train_indices]
    metadata_val = metadata.iloc[val_indices]

    print(f"📊 Train set: {len(X_train)} samples")
    print(f"📊 Validation set: {len(X_val)} samples")
    print(f"📅 Train date range: {metadata_train['date'].min()} to {metadata_train['date'].max()}")
    print(f"📅 Val date range: {metadata_val['date'].min()} to {metadata_val['date'].max()}")

    return X_train, X_val, y_train, y_val, metadata_train, metadata_val

def train_lightgbm_model(X_train, y_train, X_val, y_val, feature_cols, categorical_cols):
    """Train LightGBM model"""
    print("🚀 Training LightGBM model...")

    # Create LightGBM datasets
    train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_cols)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, categorical_feature=categorical_cols)

    # LightGBM parameters optimized for time series forecasting
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'random_state': 42,
        'n_estimators': 1000,
        'early_stopping_rounds': 50,
        'min_child_samples': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8
    }

    # Train model
    model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'valid'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=100)
        ]
    )

    print(f"✅ Model trained with {model.best_iteration} iterations")

    return model

def generate_predictions(model, X_val, metadata_val):
    """Generate predictions for validation set"""
    print("🔮 Generating predictions...")

    # Make predictions
    y_pred = model.predict(X_val, num_iteration=model.best_iteration)

    # Create predictions dataframe
    predictions_df = metadata_val.copy()
    predictions_df['predicted_quantity'] = y_pred
    predictions_df['prediction_error'] = predictions_df['predicted_quantity'] - predictions_df['quantity']
    predictions_df['absolute_error'] = np.abs(predictions_df['prediction_error'])
    predictions_df['percentage_error'] = (predictions_df['absolute_error'] / predictions_df['quantity'].replace(0, 1)) * 100

    print(f"✅ Generated {len(predictions_df)} predictions")

    return predictions_df

def calculate_metrics(predictions_df):
    """Calculate evaluation metrics"""
    print("📊 Calculating evaluation metrics...")

    # Remove zero actual values for MAPE calculation
    valid_predictions = predictions_df[predictions_df['quantity'] > 0].copy()

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(valid_predictions['quantity'], valid_predictions['predicted_quantity']))
    mape = mean_absolute_percentage_error(valid_predictions['quantity'], valid_predictions['predicted_quantity']) * 100

    # Additional metrics
    mae = valid_predictions['absolute_error'].mean()
    median_ae = valid_predictions['absolute_error'].median()

    # Accuracy within different thresholds
    accuracy_10pct = (valid_predictions['percentage_error'] <= 10).mean() * 100
    accuracy_20pct = (valid_predictions['percentage_error'] <= 20).mean() * 100

    metrics = {
        'rmse': rmse,
        'mape': mape,
        'mae': mae,
        'median_absolute_error': median_ae,
        'accuracy_within_10pct': accuracy_10pct,
        'accuracy_within_20pct': accuracy_20pct,
        'total_predictions': len(predictions_df),
        'valid_predictions': len(valid_predictions)
    }

    return metrics

def print_metrics(metrics):
    """Print evaluation metrics to console"""
    print("\n" + "="*60)
    print("LIGHTGBM MODEL EVALUATION METRICS")
    print("="*60)
    print(f"RMSE: {metrics['rmse']:.2f}")
    print(f"MAPE: {metrics['mape']:.2f}%")
    print(f"MAE: {metrics['mae']:.2f}")
    print(f"Median Absolute Error: {metrics['median_absolute_error']:.2f}")
    print(f"Accuracy within 10%: {metrics['accuracy_within_10pct']:.1f}%")
    print(f"Accuracy within 20%: {metrics['accuracy_within_20pct']:.1f}%")
    print(f"Total Predictions: {metrics['total_predictions']}")
    print(f"Valid Predictions: {metrics['valid_predictions']}")
    print("="*60)

def save_model_artifacts(model, feature_cols, categorical_cols, label_encoders, metrics):
    """Save model artifacts to disk"""
    print("💾 Saving model artifacts...")

    # Create output directory
    model_dir = Path("ml/models/lgb")
    model_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = model_dir / "lightgbm_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"✅ Model saved to {model_path}")

    # Save feature names
    features_path = model_dir / "feature_names.json"
    with open(features_path, 'w') as f:
        json.dump(feature_cols, f, indent=2)
    print(f"✅ Feature names saved to {features_path}")

    # Save categorical columns
    categorical_path = model_dir / "categorical_columns.json"
    with open(categorical_path, 'w') as f:
        json.dump(categorical_cols, f, indent=2)
    print(f"✅ Categorical columns saved to {categorical_path}")

    # Save label encoders
    encoders_path = model_dir / "label_encoders.pkl"
    with open(encoders_path, 'wb') as f:
        pickle.dump(label_encoders, f)
    print(f"✅ Label encoders saved to {encoders_path}")

    # Save metrics
    metrics_path = model_dir / "evaluation_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"✅ Metrics saved to {metrics_path}")

    # Save model info
    model_info = {
        'model_type': 'LightGBM Regressor',
        'target_variable': 'quantity',
        'best_iteration': model.best_iteration,
        'feature_importance': dict(zip(feature_cols, model.feature_importance().tolist())),
        'categorical_features': categorical_cols,
        'training_params': model.params
    }

    info_path = model_dir / "model_info.json"
    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=2, default=str)
    print(f"✅ Model info saved to {info_path}")

def save_predictions(predictions_df):
    """Save predictions to CSV"""
    print("💾 Saving predictions...")

    # Create output directory
    output_dir = Path("ml/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save predictions
    predictions_path = output_dir / "lgb_predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)
    print(f"✅ Predictions saved to {predictions_path}")

    # Save summary by material
    material_summary = predictions_df.groupby('material_id').agg({
        'quantity': ['count', 'mean'],
        'predicted_quantity': 'mean',
        'absolute_error': 'mean',
        'percentage_error': 'mean'
    }).round(2)

    material_summary.columns = ['count', 'actual_mean', 'predicted_mean', 'mae', 'mape']
    material_summary = material_summary.reset_index()

    summary_path = output_dir / "lgb_material_summary.csv"
    material_summary.to_csv(summary_path, index=False)
    print(f"✅ Material summary saved to {summary_path}")

def main():
    """Main function to train LightGBM model"""
    print("🚀 Starting LightGBM Training Pipeline")
    print("=" * 50)

    try:
        # Load and prepare data
        df = load_features()
        X, y, metadata, feature_cols, categorical_cols, label_encoders = prepare_data_for_lgb(df)

        # Create time-based split
        X_train, X_val, y_train, y_val, metadata_train, metadata_val = create_time_based_split(
            X, y, metadata, train_ratio=0.8
        )

        # Train model
        model = train_lightgbm_model(X_train, y_train, X_val, y_val, feature_cols, categorical_cols)

        # Generate predictions
        predictions_df = generate_predictions(model, X_val, metadata_val)

        # Calculate and print metrics
        metrics = calculate_metrics(predictions_df)
        print_metrics(metrics)

        # Save artifacts
        save_model_artifacts(model, feature_cols, categorical_cols, label_encoders, metrics)
        save_predictions(predictions_df)

        print("\n" + "=" * 50)
        print("✅ LightGBM training pipeline completed!")
        print("=" * 50)

    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()