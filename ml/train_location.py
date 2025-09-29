# Location Model Training Script

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ml.location_model import train_location_based_model

def load_feature_data():
    """
    Load feature data for training

    Returns:
        pd.DataFrame: Feature dataframe
    """
    print("📥 Loading feature data...")

    # Try to load from features.csv
    features_path = Path("../data/features/features.csv")
    if not features_path.exists():
        print("❌ Features file not found. Please run feature engineering first.")
        print("   Run: cd ml && python features.py")
        sys.exit(1)

    try:
        features_df = pd.read_csv(features_path)
        print(f"✅ Loaded {len(features_df)} feature records with {len(features_df.columns)} features")
        return features_df
    except Exception as e:
        print(f"❌ Error loading features: {e}")
        sys.exit(1)

def validate_data(features_df):
    """
    Validate that required columns are present

    Args:
        features_df: Feature dataframe

    Returns:
        bool: True if validation passes
    """
    required_columns = ['quantity', 'location', 'tower_type', 'substation_type']

    missing_columns = [col for col in required_columns if col not in features_df.columns]
    if missing_columns:
        print(f"❌ Missing required columns: {missing_columns}")
        return False

    # Check for location-based features
    location_features = ['location_demand_multiplier', 'infrastructure_density_index']
    available_location_features = [col for col in location_features if col in features_df.columns]

    if not available_location_features:
        print("⚠️ No location-specific features found. Model may not perform optimally.")
    else:
        print(f"✅ Found {len(available_location_features)} location features: {available_location_features}")

    return True

def save_training_report(model_results, features_df):
    """
    Save comprehensive training report

    Args:
        model_results: Results from model training
        features_df: Feature dataframe
    """
    report_dir = Path("ml/models/location")
    report_dir.mkdir(parents=True, exist_ok=True)

    report = {
        'training_summary': {
            'timestamp': datetime.now().isoformat(),
            'dataset_size': len(features_df),
            'features_count': len(features_df.columns),
            'target_variable': 'quantity'
        },
        'model_performance': model_results['metrics'],
        'feature_importance': dict(list(model_results['feature_importance'].items())[:20]),  # Top 20
        'model_path': str(model_results['model_path']),
        'data_characteristics': {
            'unique_locations': features_df['location'].nunique() if 'location' in features_df.columns else 0,
            'unique_tower_types': features_df['tower_type'].nunique() if 'tower_type' in features_df.columns else 0,
            'unique_substation_types': features_df['substation_type'].nunique() if 'substation_type' in features_df.columns else 0,
            'date_range': {
                'start': features_df['date'].min() if 'date' in features_df.columns else None,
                'end': features_df['date'].max() if 'date' in features_df.columns else None
            }
        }
    }

    report_path = report_dir / "training_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    print(f"📄 Training report saved to: {report_path}")

def print_training_summary(model_results, features_df):
    """
    Print comprehensive training summary

    Args:
        model_results: Results from model training
        features_df: Feature dataframe
    """
    print("\n" + "="*60)
    print("🎯 LOCATION MODEL TRAINING SUMMARY")
    print("="*60)

    print("📊 Dataset Overview:")
    print(f"   • Total records: {len(features_df):,}")
    print(f"   • Features used: {len(features_df.columns)}")
    print(f"   • Unique locations: {features_df['location'].nunique() if 'location' in features_df.columns else 'N/A'}")
    print(f"   • Tower types: {features_df['tower_type'].nunique() if 'tower_type' in features_df.columns else 'N/A'}")
    print(f"   • Substation types: {features_df['substation_type'].nunique() if 'substation_type' in features_df.columns else 'N/A'}")

    print("\n📈 Model Performance:")
    metrics = model_results['metrics']
    print("   Train Set:")
    print(f"     • MAE: {metrics['train']['mae']:.3f}")
    print(f"     • RMSE: {metrics['train']['rmse']:.3f}")
    print(f"     • R²: {metrics['train']['r2']:.3f}")
    print("   Test Set:")
    print(f"     • MAE: {metrics['test']['mae']:.3f}")
    print(f"     • RMSE: {metrics['test']['rmse']:.3f}")
    print(f"     • R²: {metrics['test']['r2']:.3f}")

    print("\n🎯 Top 5 Important Features:")
    for i, (feature, importance) in enumerate(list(model_results['feature_importance'].items())[:5]):
        print(f"     {i+1}. {feature}: {importance:.3f}")

    print("\n💾 Model Artifacts:")
    print(f"   • Model saved: {model_results['model_path']}")
    print("   • Metadata files: model_info.json, feature_names.json, categorical_columns.json")
    print("   • Training report: training_report.json")

    print("\n✅ Training completed successfully!")
    print("="*60)

def main():
    """
    Main training function
    """
    print("🚀 Starting Location-Based Model Training")
    print("="*50)

    # Load data
    features_df = load_feature_data()

    # Validate data
    if not validate_data(features_df):
        sys.exit(1)

    # Train model
    print("\n🏗️ Training location-based forecasting model...")
    model_results = train_location_based_model(
        features_df,
        target='quantity',
        test_size=0.2,
        random_state=42
    )

    # Save training report
    save_training_report(model_results, features_df)

    # Print summary
    print_training_summary(model_results, features_df)

    print("\n🎉 Location model training completed!")
    print("💡 Next steps:")
    print("   • Evaluate model: python ml/train_location.py --evaluate")
    print("   • Make predictions: Use predict_with_location_model() in location_model.py")
    print("   • Compare with other models: Check ensemble performance")

if __name__ == "__main__":
    main()