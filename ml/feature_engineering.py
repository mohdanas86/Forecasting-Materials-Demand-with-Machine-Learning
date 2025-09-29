# Feature Engineering Module for POWERGRID Inventory Forecasting

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def build_features(df):
    """
    Main feature engineering pipeline that calls all feature creation functions

    Args:
        df: Input dataframe with raw data

    Returns:
        df: Dataframe with all engineered features
    """
    print("🔧 Starting comprehensive feature engineering...")

    # Create location-based features
    df = create_location_features(df)

    # Create project type features
    df = create_project_type_features(df)

    # Create budget-related features
    df = create_budget_features(df)

    print("✅ All new features created successfully")
    return df

def create_location_features(df):
    """
    Create location-based features including one-hot encoding and geographic buckets

    Args:
        df: Input dataframe with location data

    Returns:
        df: Dataframe with location features added
    """
    print("📍 Creating location-based features...")

    # One-hot encoding for Indian states
    indian_states = [
        'Maharashtra', 'Gujarat', 'Rajasthan', 'Madhya Pradesh', 'Karnataka',
        'Tamil Nadu', 'Andhra Pradesh', 'Telangana', 'Uttar Pradesh', 'Bihar',
        'West Bengal', 'Odisha', 'Chhattisgarh', 'Jharkhand', 'Punjab',
        'Haryana', 'Kerala', 'Assam', 'Delhi', 'Goa'
    ]

    # Create one-hot encoded columns for states
    for state in indian_states:
        df[f'location_{state.lower().replace(" ", "_")}'] = (df['location'] == state).astype(int)

    # Geographic region buckets (based on POWERGRID zones)
    region_mapping = {
        # Northern Region
        'Delhi': 'north', 'Punjab': 'north', 'Haryana': 'north', 'Uttar Pradesh': 'north',
        'Uttarakhand': 'north', 'Himachal Pradesh': 'north', 'Jammu and Kashmir': 'north',

        # Western Region
        'Maharashtra': 'west', 'Gujarat': 'west', 'Rajasthan': 'west', 'Goa': 'west',

        # Southern Region
        'Karnataka': 'south', 'Tamil Nadu': 'south', 'Kerala': 'south', 'Andhra Pradesh': 'south',
        'Telangana': 'south', 'Puducherry': 'south',

        # Eastern Region
        'West Bengal': 'east', 'Bihar': 'east', 'Odisha': 'east', 'Jharkhand': 'east',
        'Assam': 'east', 'Sikkim': 'east', 'Arunachal Pradesh': 'east',

        # North-Eastern Region
        'Nagaland': 'northeast', 'Manipur': 'northeast', 'Mizoram': 'northeast',
        'Tripura': 'northeast', 'Meghalaya': 'northeast',

        # Central Region
        'Madhya Pradesh': 'central', 'Chhattisgarh': 'central'
    }

    df['geo_region'] = df['location'].map(region_mapping).fillna('other')

    # One-hot encoding for regions
    regions = ['north', 'south', 'east', 'west', 'central', 'northeast', 'other']
    for region in regions:
        df[f'geo_region_{region}'] = (df['geo_region'] == region).astype(int)

    # Geographic demand multipliers based on infrastructure density
    location_demand_multipliers = {
        'Maharashtra': 1.3, 'Gujarat': 1.2, 'Rajasthan': 0.9, 'Madhya Pradesh': 0.8,
        'Karnataka': 1.1, 'Tamil Nadu': 0.95, 'Andhra Pradesh': 1.0, 'Telangana': 1.05,
        'Uttar Pradesh': 0.85, 'Bihar': 0.75, 'West Bengal': 0.9, 'Odisha': 0.8,
        'Chhattisgarh': 0.7, 'Jharkhand': 0.65, 'Punjab': 0.8, 'Haryana': 0.85,
        'Kerala': 0.9, 'Assam': 0.6, 'Delhi': 1.4, 'Goa': 0.5
    }

    df['location_demand_multiplier'] = df['location'].map(location_demand_multipliers).fillna(1.0)

    # Infrastructure development index (proxy for transmission line density)
    infrastructure_index = {
        'Maharashtra': 0.85, 'Gujarat': 0.82, 'Rajasthan': 0.65, 'Madhya Pradesh': 0.58,
        'Karnataka': 0.75, 'Tamil Nadu': 0.72, 'Andhra Pradesh': 0.68, 'Telangana': 0.70,
        'Uttar Pradesh': 0.55, 'Bihar': 0.45, 'West Bengal': 0.60, 'Odisha': 0.52,
        'Chhattisgarh': 0.48, 'Jharkhand': 0.42, 'Punjab': 0.62, 'Haryana': 0.65,
        'Kerala': 0.58, 'Assam': 0.35, 'Delhi': 0.90, 'Goa': 0.40
    }

    df['infrastructure_density_index'] = df['location'].map(infrastructure_index).fillna(0.5)

    # Seasonal location factors (some regions have different monsoon patterns)
    monsoon_impacted_states = ['Maharashtra', 'Gujarat', 'Karnataka', 'Kerala', 'Goa']
    df['monsoon_sensitive_location'] = df['location'].isin(monsoon_impacted_states).astype(int)

    print(f"✅ Created {len([col for col in df.columns if 'location_' in col or 'geo_' in col])} location features")
    return df

def create_project_type_features(df):
    """
    Create project type features based on tower and substation types

    Args:
        df: Input dataframe with project type data

    Returns:
        df: Dataframe with project type features added
    """
    print("🏗️ Creating project type features...")

    # Tower type encodings
    tower_types = ['Lattice Tower', 'Tubular Tower', 'Guyed Tower', 'Monopole Tower']

    # One-hot encoding for tower types
    for tower_type in tower_types:
        df[f'tower_type_{tower_type.lower().replace(" ", "_")}'] = (df['tower_type'] == tower_type).astype(int)

    # Tower complexity factors (material requirements)
    tower_complexity_factors = {
        'Lattice Tower': 1.0,      # Baseline
        'Tubular Tower': 1.2,      # More complex, higher material needs
        'Guyed Tower': 0.8,        # Simpler design, less materials
        'Monopole Tower': 0.9      # Moderate complexity
    }

    df['tower_complexity_factor'] = df['tower_type'].map(tower_complexity_factors).fillna(1.0)

    # Tower material intensity (steel requirements per km)
    tower_material_intensity = {
        'Lattice Tower': 1.0,      # Standard steel requirements
        'Tubular Tower': 1.3,      # Higher steel content
        'Guyed Tower': 0.7,        # Less steel, more concrete/foundations
        'Monopole Tower': 0.9      # Moderate steel requirements
    }

    df['tower_material_intensity'] = df['tower_type'].map(tower_material_intensity).fillna(1.0)

    # Substation type encodings
    substation_types = ['AIS Substation', 'GIS Substation', 'Hybrid Substation']

    # One-hot encoding for substation types
    for substation_type in substation_types:
        df[f'substation_type_{substation_type.lower().replace(" ", "_")}'] = (df['substation_type'] == substation_type).astype(int)

    # Substation material factors
    substation_material_factors = {
        'AIS Substation': 1.0,     # Air Insulated Switchgear - baseline
        'GIS Substation': 1.4,     # Gas Insulated Switchgear - higher material intensity
        'Hybrid Substation': 1.2   # Mixed technology - moderate increase
    }

    df['substation_material_factor'] = df['substation_type'].map(substation_material_factors).fillna(1.0)

    # Substation equipment complexity
    substation_equipment_factors = {
        'AIS Substation': 1.0,     # Standard equipment
        'GIS Substation': 1.6,     # More sophisticated equipment
        'Hybrid Substation': 1.3   # Mixed complexity
    }

    df['substation_equipment_factor'] = df['substation_type'].map(substation_equipment_factors).fillna(1.0)

    # Combined project complexity score
    df['project_complexity_score'] = df['tower_complexity_factor'] * df['substation_material_factor']

    # Project type categories
    df['project_category'] = 'transmission'  # All projects are transmission for POWERGRID

    # Material demand patterns by project type
    df['conductor_demand_factor'] = df['tower_complexity_factor'] * 1.1  # Towers need conductors
    df['transformer_demand_factor'] = df['substation_material_factor'] * 1.2  # Substations need transformers
    df['insulator_demand_factor'] = df['tower_complexity_factor'] * 0.9  # Towers need insulators

    print(f"✅ Created {len([col for col in df.columns if 'tower_' in col or 'substation_' in col or 'project_' in col])} project type features")
    return df

def create_budget_features(df):
    """
    Create budget-related features including utilization and intensity metrics

    Args:
        df: Input dataframe with budget data

    Returns:
        df: Dataframe with budget features added
    """
    print("💰 Creating budget-related features...")

    # Budget categories based on POWERGRID project scales
    budget_bins = [0, 50000000, 200000000, 500000000, 1000000000, float('inf')]
    budget_labels = ['small', 'medium', 'large', 'mega', 'ultra_mega']

    df['budget_category'] = pd.cut(df['budget'], bins=budget_bins, labels=budget_labels, include_lowest=True)

    # One-hot encoding for budget categories
    for category in budget_labels:
        df[f'budget_category_{category}'] = (df['budget_category'] == category).astype(int)

    # Budget utilization rate (estimated based on project phase and time elapsed)
    # This is a simplified estimation - in real scenarios, this would come from project management data
    current_date = pd.Timestamp.now()

    # Calculate project duration and elapsed time
    df['project_duration_days'] = (pd.to_datetime(df['end_date']) - pd.to_datetime(df['start_date'])).dt.days
    df['elapsed_days'] = (current_date - pd.to_datetime(df['start_date'])).dt.days
    df['project_progress_ratio'] = (df['elapsed_days'] / df['project_duration_days']).clip(0, 1)

    # Estimate budget utilization based on project progress (S-curve pattern)
    df['budget_utilization_rate'] = df['project_progress_ratio'].apply(
        lambda x: 0.1 + 0.8 * (1 - np.exp(-3 * x)) if x < 0.8 else 0.9 + 0.1 * (x - 0.8) / 0.2
    )

    # Budget intensity (material cost per unit budget)
    df['budget_intensity'] = df['quantity'] / df['budget'].replace(0, 1)  # Avoid division by zero

    # Budget allocation efficiency (higher values = more efficient material usage)
    df['budget_allocation_efficiency'] = 1 / (df['budget_intensity'] + 0.001)  # Inverse relationship

    # Material cost ratio (estimated material cost as percentage of budget)
    material_cost_ratios = {
        'small': 0.6,      # Small projects have higher material cost ratio
        'medium': 0.55,
        'large': 0.5,
        'mega': 0.45,      # Large projects have more overhead
        'ultra_mega': 0.4
    }

    df['estimated_material_cost_ratio'] = df['budget_category'].map(material_cost_ratios)

    # Project scale factor (larger projects may have different procurement patterns)
    scale_factors = {
        'small': 0.8,
        'medium': 1.0,
        'large': 1.2,
        'mega': 1.4,
        'ultra_mega': 1.6
    }

    df['project_scale_factor'] = df['budget_category'].map(scale_factors)

    # Budget pressure indicator (projects behind schedule may accelerate spending)
    df['budget_pressure_indicator'] = ((df['elapsed_days'] > df['project_duration_days'] * 0.7) &
                                      (df['project_progress_ratio'] < 0.6)).astype(int)

    # Procurement urgency factor
    df['procurement_urgency_factor'] = df['budget_pressure_indicator'] * 1.3 + 1.0

    print(f"✅ Created {len([col for col in df.columns if 'budget_' in col])} budget features")
    return df

# Utility functions for feature engineering
def validate_features(df, required_columns=None):
    """
    Validate that required features are present and valid

    Args:
        df: Dataframe to validate
        required_columns: List of required column names

    Returns:
        bool: True if validation passes
    """
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"⚠️ Missing required columns: {missing_columns}")
            return False

    # Check for NaN values in numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    nan_counts = df[numeric_cols].isnull().sum()
    if nan_counts.sum() > 0:
        print(f"⚠️ Found NaN values in columns: {nan_counts[nan_counts > 0].to_dict()}")

    return True

def get_feature_importance_ranking(df, target_col='quantity'):
    """
    Calculate feature importance ranking using correlation

    Args:
        df: Dataframe with features
        target_col: Target column name

    Returns:
        pd.Series: Feature importance rankings
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlations = df[numeric_cols].corr()[target_col].abs().sort_values(ascending=False)
    return correlations

if __name__ == "__main__":
    # Example usage
    print("Feature Engineering Module for POWERGRID Inventory Forecasting")
    print("Run this module through the main features.py pipeline")