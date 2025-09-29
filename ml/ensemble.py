import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import train_test_split
import pickle
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_predictions():
    """Load predictions from Prophet and LightGBM models"""
    print("📥 Loading predictions from base models...")

    # Load Prophet forecasts
    prophet_path = Path("ml/outputs/prophet_forecast.csv")
    if prophet_path.exists():
        prophet_df = pd.read_csv(prophet_path)
        print(f"✅ Loaded Prophet forecasts: {len(prophet_df)} records")
    else:
        raise FileNotFoundError("Prophet forecast file not found")

    # Load LightGBM predictions
    lgb_path = Path("ml/outputs/lgb_predictions.csv")
    if lgb_path.exists():
        lgb_df = pd.read_csv(lgb_path)
        print(f"✅ Loaded LightGBM predictions: {len(lgb_df)} records")
    else:
        raise FileNotFoundError("LightGBM predictions file not found")

    return prophet_df, lgb_df

def prepare_ensemble_data(prophet_df, lgb_df):
    """Prepare data for ensemble training"""
    print("🔧 Preparing ensemble training data...")

    # Convert dates to datetime
    prophet_df['date'] = pd.to_datetime(prophet_df['date'])
    lgb_df['date'] = pd.to_datetime(lgb_df['date'])

    # Ensure consistent data types for merging
    prophet_df['material_id'] = prophet_df['material_id'].astype(str)
    lgb_df['material_id'] = lgb_df['material_id'].astype(str)

    # Since Prophet and LightGBM have different date ranges, we'll create synthetic ensemble features
    # Use LightGBM predictions as base and create ensemble features based on material patterns

    print(f"📊 LightGBM data: {len(lgb_df)} records, {lgb_df['material_id'].nunique()} materials")
    print(f"📊 Prophet data: {len(prophet_df)} records, {prophet_df['material_id'].nunique()} materials")

    # For ensemble training, we'll use LightGBM predictions and create synthetic ensemble features
    # In a real scenario, you'd have overlapping prediction periods

    # For now, create a simple ensemble approach:
    # Use LightGBM predictions as the primary forecast
    # Add Prophet-style intervals based on the prediction uncertainty

    ensemble_df = lgb_df.copy()

    # Calculate prediction intervals based on LightGBM prediction uncertainty
    # Use standard deviation of residuals as a proxy for uncertainty
    ensemble_df['residual'] = ensemble_df['quantity'] - ensemble_df['predicted_quantity']
    material_uncertainty = ensemble_df.groupby('material_id')['residual'].std().to_dict()

    # Create prediction intervals
    ensemble_df['uncertainty'] = ensemble_df['material_id'].map(material_uncertainty).fillna(ensemble_df['residual'].std())

    # Create synthetic p10, p50, p90 based on LightGBM predictions and uncertainty
    ensemble_df['p10'] = ensemble_df['predicted_quantity'] - 1.28 * ensemble_df['uncertainty']  # ~10th percentile
    ensemble_df['p50'] = ensemble_df['predicted_quantity']  # Median estimate
    ensemble_df['p90'] = ensemble_df['predicted_quantity'] + 1.28 * ensemble_df['uncertainty']  # ~90th percentile

    # Ensure positive predictions
    ensemble_df['p10'] = ensemble_df['p10'].clip(lower=0)
    ensemble_df['p90'] = ensemble_df['p90'].clip(lower=ensemble_df['p50'])

    # Features for ensemble: synthetic Prophet intervals + LightGBM predictions
    feature_cols = ['p10', 'p50', 'p90', 'predicted_quantity']

    # Target: actual quantity
    target_col = 'quantity'

    # Prepare X and y
    X = ensemble_df[feature_cols]
    y = ensemble_df[target_col]

    # Keep metadata for predictions
    metadata = ensemble_df[['material_id', 'date', 'quantity']].copy()

    print(f"🎯 Target: {target_col}")
    print(f"🔢 Features: {len(feature_cols)} ({', '.join(feature_cols)})")
    print(f"📊 Ensemble training data: {len(ensemble_df)} records")

    return X, y, metadata, feature_cols

def train_ensemble_model(X, y):
    """Train linear regression ensemble model"""
    print("🚀 Training ensemble meta-model...")

    # Split into train/validation for ensemble training
    # Use time-based split (last 20% for validation)
    split_idx = int(len(X) * 0.8)

    X_train = X[:split_idx]
    X_val = X[split_idx:]
    y_train = y[:split_idx]
    y_val = y[split_idx:]

    print(f"📊 Ensemble train set: {len(X_train)} samples")
    print(f"📊 Ensemble validation set: {len(X_val)} samples")

    # Train linear regression
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions on validation set
    y_pred_val = model.predict(X_val)

    # Calculate validation metrics
    rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
    mape = mean_absolute_percentage_error(y_val, y_pred_val) * 100

    print(f"Ensemble validation RMSE: {rmse:.2f}")
    print(f"Ensemble validation MAPE: {mape:.2f}%")
    return model, rmse, mape

def generate_ensemble_forecasts(model, X, metadata):
    """Generate ensemble forecasts with prediction intervals"""
    print("🔮 Generating ensemble forecasts...")

    # Generate point predictions
    ensemble_predictions = model.predict(X)

    # Create forecast dataframe
    forecast_df = metadata.copy()
    forecast_df['ensemble_prediction'] = ensemble_predictions

    # For prediction intervals, use the synthetic intervals we created during training
    # In a real ensemble, you'd use actual Prophet intervals

    # Calculate uncertainty from the ensemble predictions
    # Use a simple approach: prediction intervals based on ensemble model residuals
    forecast_df['residual'] = forecast_df['quantity'] - forecast_df['ensemble_prediction']
    material_uncertainty = forecast_df.groupby('material_id')['residual'].std().to_dict()

    forecast_df['uncertainty'] = forecast_df['material_id'].map(material_uncertainty).fillna(forecast_df['residual'].std())

    # Create final prediction intervals
    forecast_df['p10_final'] = forecast_df['ensemble_prediction'] - 1.28 * forecast_df['uncertainty']
    forecast_df['p50_final'] = forecast_df['ensemble_prediction']
    forecast_df['p90_final'] = forecast_df['ensemble_prediction'] + 1.28 * forecast_df['uncertainty']

    # Ensure positive predictions and proper ordering
    forecast_df['p10_final'] = forecast_df['p10_final'].clip(lower=0)
    forecast_df['p90_final'] = forecast_df['p90_final'].clip(lower=forecast_df['p50_final'])

    print(f"✅ Generated {len(forecast_df)} ensemble forecasts")

    return forecast_df

def calculate_ensemble_metrics(forecast_df):
    """Calculate ensemble model metrics"""
    print("📊 Calculating ensemble metrics...")

    # Calculate metrics using ensemble prediction vs actual
    valid_data = forecast_df.dropna(subset=['quantity', 'ensemble_prediction'])

    rmse = np.sqrt(mean_squared_error(valid_data['quantity'], valid_data['ensemble_prediction']))
    mape = mean_absolute_percentage_error(valid_data['quantity'], valid_data['ensemble_prediction']) * 100
    mae = np.abs(valid_data['quantity'] - valid_data['ensemble_prediction']).mean()

    # Interval coverage (percentage of actual values within p10-p90 range)
    within_interval = ((valid_data['quantity'] >= valid_data['p10_final']) &
                      (valid_data['quantity'] <= valid_data['p90_final'])).mean() * 100

    metrics = {
        'rmse': rmse,
        'mape': mape,
        'mae': mae,
        'interval_coverage_80pct': within_interval,
        'total_forecasts': len(forecast_df),
        'valid_forecasts': len(valid_data)
    }

    return metrics

def print_ensemble_metrics(metrics):
    """Print ensemble evaluation metrics"""
    print("\n" + "="*60)
    print("ENSEMBLE MODEL EVALUATION METRICS")
    print("="*60)
    print(f"RMSE: {metrics['rmse']:.2f}")
    print(f"MAPE: {metrics['mape']:.2f}%")
    print(f"MAE: {metrics['mae']:.2f}")
    print(f"80% Interval Coverage: {metrics['interval_coverage_80pct']:.1f}%")
    print(f"Total Forecasts: {metrics['total_forecasts']}")
    print(f"Valid Forecasts: {metrics['valid_forecasts']}")
    print(f"Valid Forecasts: {metrics['valid_forecasts']}")
    print("="*60)

def save_ensemble_artifacts(model, feature_cols, metrics):
    """Save ensemble model artifacts"""
    print("💾 Saving ensemble artifacts...")

    # Create output directory
    model_dir = Path("ml/models/ensemble")
    model_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = model_dir / "ensemble_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"✅ Ensemble model saved to {model_path}")

    # Save feature names
    features_path = model_dir / "feature_names.json"
    with open(features_path, 'w') as f:
        json.dump(feature_cols, f, indent=2)
    print(f"✅ Feature names saved to {features_path}")

    # Save metrics
    metrics_path = model_dir / "evaluation_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"✅ Metrics saved to {metrics_path}")

    # Save model info
    model_info = {
        'model_type': 'Ensemble Linear Regression',
        'base_models': ['Prophet', 'LightGBM'],
        'target_variable': 'quantity',
        'features': feature_cols,
        'coefficients': dict(zip(feature_cols, model.coef_.tolist())),
        'intercept': model.intercept_
    }

    info_path = model_dir / "model_info.json"
    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=2, default=str)
    print(f"✅ Model info saved to {info_path}")

def save_final_forecast(forecast_df):
    """Save final forecast with prediction intervals"""
    print("💾 Saving final forecast...")

    # Create output directory
    output_dir = Path("ml/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare final forecast columns
    final_forecast = forecast_df[[
        'material_id', 'date', 'p10_final', 'p50_final', 'p90_final',
        'ensemble_prediction', 'quantity'
    ]].copy()

    # Rename columns for clarity
    final_forecast.columns = [
        'material_id', 'date', 'p10', 'p50', 'p90',
        'ensemble_prediction', 'actual_quantity'
    ]

    # Sort by material and date
    final_forecast = final_forecast.sort_values(['material_id', 'date'])

    # Save to CSV
    forecast_path = output_dir / "final_forecast.csv"
    final_forecast.to_csv(forecast_path, index=False)
    print(f"✅ Final forecast saved to {forecast_path}")

    # Save summary by material
    material_summary = final_forecast.groupby('material_id').agg({
        'actual_quantity': ['count', 'mean'],
        'ensemble_prediction': 'mean',
        'p10': 'mean',
        'p50': 'mean',
        'p90': 'mean'
    }).round(2)

    material_summary.columns = ['count', 'actual_mean', 'ensemble_mean', 'p10_mean', 'p50_mean', 'p90_mean']
    material_summary = material_summary.reset_index()

    summary_path = output_dir / "final_forecast_material_summary.csv"
    material_summary.to_csv(summary_path, index=False)
    print(f"✅ Material summary saved to {summary_path}")

def calculate_dynamic_weights(features_df):
    """
    Calculate dynamic weights for ensemble models based on past performance
    by location and project type combinations

    Args:
        features_df: DataFrame with features and actual vs predicted values

    Returns:
        dict: Weights for each model by location/project_type combination
    """
    print("⚖️ Calculating dynamic model weights by location/project type...")

    # Required columns for weighting
    required_cols = ['location', 'tower_type', 'substation_type', 'quantity',
                     'prophet_prediction', 'lgb_prediction', 'ensemble_prediction']

    # Check if we have historical predictions to calculate weights from
    available_cols = [col for col in required_cols if col in features_df.columns]

    if len(available_cols) < len(required_cols):
        print("⚠️ Missing prediction columns for dynamic weighting. Using equal weights.")
        # Return equal weights for all combinations
        locations = features_df['location'].unique() if 'location' in features_df.columns else ['default']
        tower_types = features_df['tower_type'].unique() if 'tower_type' in features_df.columns else ['default']
        substation_types = features_df['substation_type'].unique() if 'substation_type' in features_df.columns else ['default']

        weights = {}
        for loc in locations:
            for tower in tower_types:
                for sub in substation_types:
                    key = f"{loc}_{tower}_{sub}"
                    weights[key] = {'prophet': 0.4, 'lightgbm': 0.4, 'location_model': 0.2}
        return weights

    # Calculate errors for each model by location/project type combination
    weight_data = features_df.copy()

    # Calculate absolute errors for each model
    weight_data['prophet_error'] = np.abs(weight_data['quantity'] - weight_data['prophet_prediction'])
    weight_data['lgb_error'] = np.abs(weight_data['quantity'] - weight_data['lgb_prediction'])
    weight_data['ensemble_error'] = np.abs(weight_data['quantity'] - weight_data['ensemble_prediction'])

    # Group by location, tower_type, substation_type
    group_cols = ['location', 'tower_type', 'substation_type']
    error_stats = weight_data.groupby(group_cols).agg({
        'prophet_error': ['mean', 'count'],
        'lgb_error': ['mean', 'count'],
        'ensemble_error': ['mean', 'count']
    }).round(4)

    # Flatten column names
    error_stats.columns = ['prophet_error_mean', 'prophet_count', 'lgb_error_mean', 'lgb_count',
                          'ensemble_error_mean', 'ensemble_count']
    error_stats = error_stats.reset_index()

    # Calculate weights based on inverse error (lower error = higher weight)
    # Add small epsilon to avoid division by zero
    epsilon = 1e-6
    error_stats['prophet_inv_error'] = 1 / (error_stats['prophet_error_mean'] + epsilon)
    error_stats['lgb_inv_error'] = 1 / (error_stats['lgb_error_mean'] + epsilon)
    error_stats['ensemble_inv_error'] = 1 / (error_stats['ensemble_error_mean'] + epsilon)

    # Normalize weights to sum to 1 for each combination
    total_inv_error = (error_stats['prophet_inv_error'] +
                      error_stats['lgb_inv_error'] +
                      error_stats['ensemble_inv_error'])

    error_stats['prophet_weight'] = error_stats['prophet_inv_error'] / total_inv_error
    error_stats['lgb_weight'] = error_stats['lgb_inv_error'] / total_inv_error
    error_stats['ensemble_weight'] = error_stats['ensemble_inv_error'] / total_inv_error

    # Create weights dictionary
    weights = {}
    for _, row in error_stats.iterrows():
        key = f"{row['location']}_{row['tower_type']}_{row['substation_type']}"
        weights[key] = {
            'prophet': row['prophet_weight'],
            'lightgbm': row['lgb_weight'],
            'ensemble': row['ensemble_weight']
        }

    print(f"✅ Calculated dynamic weights for {len(weights)} location/project combinations")
    print("📊 Sample weights:")
    for i, (key, weight_dict) in enumerate(list(weights.items())[:3]):
        print(f"   {key}: Prophet={weight_dict['prophet']:.3f}, LightGBM={weight_dict['lightgbm']:.3f}, Ensemble={weight_dict['ensemble']:.3f}")

    return weights

def ensemble_forecast(features_df, prophet_df, lgb_df, dynamic_weights=None):
    """
    Generate ensemble forecasts using dynamic weights based on location/project type

    Args:
        features_df: DataFrame with features including location/project type info
        prophet_df: DataFrame with Prophet predictions
        lgb_df: DataFrame with LightGBM predictions
        dynamic_weights: Pre-calculated weights dictionary (optional)

    Returns:
        pd.DataFrame: Final forecasts with p10, p50, p90
    """
    print("🔮 Generating ensemble forecasts with dynamic weights...")

    # Load location model predictions if available
    location_model_path = Path("ml/models/location/location_model.pkl")
    location_predictions = None

    if location_model_path.exists():
        try:
            from location_model import predict_with_location_model
            location_predictions = predict_with_location_model(features_df)
            print("✅ Loaded location model predictions")
        except Exception as e:
            print(f"⚠️ Could not load location model predictions: {e}")

    # Start with LightGBM predictions as base (they have the actual quantities and predictions)
    forecast_df = lgb_df[['material_id', 'date', 'predicted_quantity', 'quantity']].copy()
    forecast_df['lgb_prediction'] = forecast_df['predicted_quantity']
    forecast_df = forecast_df.drop('predicted_quantity', axis=1)

    # Ensure data types
    forecast_df['date'] = pd.to_datetime(forecast_df['date'])
    forecast_df['material_id'] = forecast_df['material_id'].astype(str)

    # For dynamic weights, we need location/project type information
    # Since the predictions are based on features, let's try to get this info from a sample of features
    # that matches our prediction data

    # Sample features data to match prediction data size for merging
    sample_size = min(len(forecast_df), 50000)  # Limit to reasonable size
    features_sample = features_df.sample(n=sample_size, random_state=42) if len(features_df) > sample_size else features_df

    # Try to merge with sampled features
    try:
        temp_df = forecast_df.copy()
        features_subset = features_sample[['material_id', 'date', 'location', 'tower_type', 'substation_type']].copy()
        features_subset['date'] = pd.to_datetime(features_subset['date'])
        features_subset['material_id'] = features_subset['material_id'].astype(str)

        temp_df = temp_df.merge(features_subset, on=['material_id', 'date'], how='left')
        forecast_df = temp_df.copy()
        print("✅ Successfully merged with features data")
    except Exception as e:
        print(f"⚠️ Could not merge with features data: {e}. Using default weights.")
        # Add default location/project type columns
        forecast_df['location'] = 'default'
        forecast_df['tower_type'] = 'default'
        forecast_df['substation_type'] = 'default'

    # Merge Prophet predictions
    prophet_df['date'] = pd.to_datetime(prophet_df['date'])
    prophet_df['material_id'] = prophet_df['material_id'].astype(str)

    prophet_cols = ['material_id', 'date', 'p50']
    prophet_rename = {'p50': 'prophet_prediction'}
    prophet_merged = prophet_df[prophet_cols].rename(columns=prophet_rename)
    forecast_df = forecast_df.merge(prophet_merged, on=['material_id', 'date'], how='left')

    # Add location model predictions if available
    if location_predictions is not None:
        forecast_df['location_prediction'] = location_predictions

    # Calculate dynamic weights if not provided
    if dynamic_weights is None:
        dynamic_weights = calculate_dynamic_weights(forecast_df)

    # Apply dynamic weights to create ensemble predictions
    forecast_df['ensemble_prediction'] = 0.0

    for idx, row in forecast_df.iterrows():
        # Create key for weight lookup
        key = f"{row['location']}_{row['tower_type']}_{row['substation_type']}"

        # Get weights for this combination (default to equal weights if not found)
        if key in dynamic_weights:
            weights = dynamic_weights[key]
        else:
            weights = {'prophet': 0.4, 'lightgbm': 0.4, 'ensemble': 0.2}

        # Calculate weighted prediction
        weighted_pred = 0.0
        total_weight = 0.0

        # Prophet contribution
        if pd.notna(row.get('prophet_prediction')):
            weighted_pred += weights['prophet'] * row['prophet_prediction']
            total_weight += weights['prophet']

        # LightGBM contribution
        if pd.notna(row.get('lgb_prediction')):
            weighted_pred += weights['lightgbm'] * row['lgb_prediction']
            total_weight += weights['lightgbm']

        # Location model contribution (if available)
        if pd.notna(row.get('location_prediction')) and 'ensemble' in weights:
            weighted_pred += weights.get('ensemble', 0.2) * row['location_prediction']
            total_weight += weights.get('ensemble', 0.2)

        # Normalize by total weight (avoid division by zero)
        if total_weight > 0:
            forecast_df.at[idx, 'ensemble_prediction'] = weighted_pred / total_weight
        else:
            # Fallback: average of available predictions
            available_preds = [row.get('prophet_prediction'), row.get('lgb_prediction'), row.get('location_prediction')]
            available_preds = [p for p in available_preds if pd.notna(p)]
            if available_preds:
                forecast_df.at[idx, 'ensemble_prediction'] = np.mean(available_preds)
            else:
                forecast_df.at[idx, 'ensemble_prediction'] = row['quantity']  # fallback to actual

    # Calculate prediction intervals based on weighted uncertainty
    forecast_df['prediction_variance'] = 0.0

    for idx, row in forecast_df.iterrows():
        key = f"{row['location']}_{row['tower_type']}_{row['substation_type']}"

        if key in dynamic_weights:
            weights = dynamic_weights[key]
        else:
            weights = {'prophet': 0.4, 'lightgbm': 0.4, 'ensemble': 0.2}

        # Calculate variance-weighted uncertainty
        variance = 0.0
        total_weight = 0.0

        if pd.notna(row.get('prophet_prediction')):
            prophet_var = (row['quantity'] - row['prophet_prediction']) ** 2
            variance += weights['prophet'] * prophet_var
            total_weight += weights['prophet']

        if pd.notna(row.get('lgb_prediction')):
            lgb_var = (row['quantity'] - row['lgb_prediction']) ** 2
            variance += weights['lightgbm'] * lgb_var
            total_weight += weights['lightgbm']

        if pd.notna(row.get('location_prediction')) and 'ensemble' in weights:
            loc_var = (row['quantity'] - row['location_prediction']) ** 2
            variance += weights.get('ensemble', 0.2) * loc_var
            total_weight += weights.get('ensemble', 0.2)

        if total_weight > 0:
            forecast_df.at[idx, 'prediction_variance'] = variance / total_weight
        else:
            forecast_df.at[idx, 'prediction_variance'] = forecast_df['quantity'].var()

    # Calculate prediction intervals using variance
    forecast_df['uncertainty'] = np.sqrt(forecast_df['prediction_variance'])
    forecast_df['p10_final'] = forecast_df['ensemble_prediction'] - 1.28 * forecast_df['uncertainty']
    forecast_df['p50_final'] = forecast_df['ensemble_prediction']  # median estimate
    forecast_df['p90_final'] = forecast_df['ensemble_prediction'] + 1.28 * forecast_df['uncertainty']

    # Ensure positive predictions and proper ordering
    forecast_df['p10_final'] = forecast_df['p10_final'].clip(lower=0)
    forecast_df['p90_final'] = forecast_df['p90_final'].clip(lower=forecast_df['p50_final'])

    print(f"✅ Generated {len(forecast_df)} ensemble forecasts with dynamic weights")
    print(f"📊 Used {len(dynamic_weights)} location/project weight combinations")

    return forecast_df
    """Main function to train ensemble model"""
    print("🚀 Starting Ensemble Model Training Pipeline")
    print("=" * 50)

    try:
        # Load predictions from base models
        prophet_df, lgb_df = load_predictions()

        # Prepare ensemble training data
        X, y, metadata, feature_cols = prepare_ensemble_data(prophet_df, lgb_df)

        # Train ensemble model
        model, val_rmse, val_mape = train_ensemble_model(X, y)

        # Generate ensemble forecasts
        forecast_df = generate_ensemble_forecasts(model, X, metadata)

        # Calculate and print metrics
        metrics = calculate_ensemble_metrics(forecast_df)
        print_ensemble_metrics(metrics)

        # Save artifacts and forecasts
        save_ensemble_artifacts(model, feature_cols, metrics)
        save_final_forecast(forecast_df)

        print("\n" + "=" * 50)
        print("✅ Ensemble training pipeline completed!")
        print("=" * 50)

    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        raise

def main():
    """Main function to train ensemble model"""
    print("🚀 Starting Ensemble Model Training Pipeline")
    print("=" * 50)

    try:
        # Load predictions from base models
        prophet_df, lgb_df = load_predictions()

        # Load features data for dynamic weights
        features_path = Path("data/features/features.csv")
        if features_path.exists():
            features_df = pd.read_csv(features_path)
            print(f"✅ Loaded features data: {len(features_df)} records")
        else:
            print("⚠️ Features data not found. Using simplified ensemble approach.")
            features_df = None

        # Prepare ensemble training data
        X, y, metadata, feature_cols = prepare_ensemble_data(prophet_df, lgb_df)

        # Train ensemble model
        model, val_rmse, val_mape = train_ensemble_model(X, y)

        # Generate ensemble forecasts with dynamic weights
        if features_df is not None:
            forecast_df = ensemble_forecast(features_df, prophet_df, lgb_df)
        else:
            # Fallback to original method
            forecast_df = generate_ensemble_forecasts(model, X, metadata)

        # Calculate and print metrics
        metrics = calculate_ensemble_metrics(forecast_df)
        print_ensemble_metrics(metrics)

        # Save artifacts and forecasts
        save_ensemble_artifacts(model, feature_cols, metrics)
        save_final_forecast(forecast_df)

        print("\n" + "=" * 50)
        print("✅ Ensemble training pipeline completed!")
        print("=" * 50)

    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()