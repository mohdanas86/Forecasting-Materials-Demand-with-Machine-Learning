import pandas as pd
import sys
import os
from pathlib import Path

def test_synthetic_data():
    """Test that synthetic data can be loaded and validated"""

    # Add the backend directory to Python path
    backend_path = Path(__file__).parent.parent / "backend"
    sys.path.insert(0, str(backend_path))

    try:
        # Import our models
        from models import Projects, Materials, HistoricalDemand, Inventory

        print("Testing synthetic data loading...")

        # Load synthetic data
        data_dir = Path("data/synthetic")

        projects_df = pd.read_csv(data_dir / "projects.csv")
        materials_df = pd.read_csv(data_dir / "materials.csv")
        demand_df = pd.read_csv(data_dir / "historical_demand.csv")
        inventory_df = pd.read_csv(data_dir / "inventory.csv")

        print(f"✓ Loaded {len(projects_df)} projects")
        print(f"✓ Loaded {len(materials_df)} materials")
        print(f"✓ Loaded {len(demand_df)} demand records")
        print(f"✓ Loaded {len(inventory_df)} inventory records")

        # Validate data types and ranges
        print("\nValidating data integrity...")

        # Check project budgets are reasonable (50M to 2B rupees)
        assert projects_df['budget'].min() > 50000000, "Project budget too low"
        assert projects_df['budget'].max() < 2000000000, "Project budget too high"
        print("✓ Project budgets are within reasonable range")

        # Check dates are valid
        projects_df['start_date'] = pd.to_datetime(projects_df['start_date'])
        projects_df['end_date'] = pd.to_datetime(projects_df['end_date'])
        assert (projects_df['end_date'] > projects_df['start_date']).all(), "End date before start date"
        print("✓ Project dates are valid")

        # Check demand quantities are positive
        assert (demand_df['quantity'] > 0).all(), "Negative demand quantities found"
        print("✓ Demand quantities are positive")

        # Check inventory levels are reasonable
        assert (inventory_df['stock_level'] > 0).all(), "Negative inventory levels found"
        print("✓ Inventory levels are positive")

        # Test data compatibility with Pydantic models
        print("\nTesting Pydantic model compatibility...")

        # Test a sample project
        sample_project = projects_df.iloc[0].to_dict()
        project_model = Projects(**sample_project)
        print("✓ Project model validation passed")

        # Test a sample material
        sample_material = materials_df.iloc[0].to_dict()
        material_model = Materials(**sample_material)
        print("✓ Material model validation passed")

        # Test a sample demand record
        sample_demand = demand_df.iloc[0].to_dict()
        sample_demand['date'] = pd.to_datetime(sample_demand['date'])
        demand_model = HistoricalDemand(**sample_demand)
        print("✓ HistoricalDemand model validation passed")

        # Test a sample inventory record
        sample_inventory = inventory_df.iloc[0].to_dict()
        sample_inventory['date'] = pd.to_datetime(sample_inventory['date'])
        inventory_model = Inventory(**sample_inventory)
        print("✓ Inventory model validation passed")

        print("\n" + "="*50)
        print("ALL TESTS PASSED! 🎉")
        print("Synthetic data is ready for use with the FastAPI backend.")
        print("="*50)

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_synthetic_data()
    sys.exit(0 if success else 1)