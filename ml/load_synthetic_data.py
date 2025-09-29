import pandas as pd
import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
from pathlib import Path
import sys

# Add backend to path
backend_path = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_path))

from models import Projects, Materials, HistoricalDemand, Inventory, PurchaseOrders, Pipeline

async def load_synthetic_data():
    """Load synthetic data into MongoDB"""

    # MongoDB connection
    MONGODB_URL = "mongodb+srv://anas:anas@food.t6wubmw.mongodb.net/inverntory"
    DATABASE_NAME = "inventory_forecasting"

    client = AsyncIOMotorClient(MONGODB_URL)
    db = client[DATABASE_NAME]

    try:
        print("Connecting to MongoDB...")
        # Test connection
        await client.admin.command('ping')
        print("✓ Connected to MongoDB")

        # Load data files
        data_dir = Path("data/synthetic")

        print("\nLoading synthetic data...")

        # Load projects
        projects_df = pd.read_csv(data_dir / "projects.csv")
        projects_df['start_date'] = pd.to_datetime(projects_df['start_date'])
        projects_df['end_date'] = pd.to_datetime(projects_df['end_date'])

        projects_data = []
        for _, row in projects_df.iterrows():
            project = Projects(**row.to_dict())
            projects_data.append(project.dict())

        await db.projects.insert_many(projects_data)
        print(f"✓ Loaded {len(projects_data)} projects")

        # Load materials
        materials_df = pd.read_csv(data_dir / "materials.csv")
        materials_data = []
        for _, row in materials_df.iterrows():
            material = Materials(**row.to_dict())
            materials_data.append(material.dict())

        await db.materials.insert_many(materials_data)
        print(f"✓ Loaded {len(materials_data)} materials")

        # Load historical demand
        demand_df = pd.read_csv(data_dir / "historical_demand.csv")
        demand_df['date'] = pd.to_datetime(demand_df['date'])

        demand_data = []
        for _, row in demand_df.iterrows():
            demand = HistoricalDemand(**row.to_dict())
            demand_data.append(demand.dict())

        await db.historical_demand.insert_many(demand_data)
        print(f"✓ Loaded {len(demand_data)} historical demand records")

        # Load inventory
        inventory_df = pd.read_csv(data_dir / "inventory.csv")
        inventory_df['date'] = pd.to_datetime(inventory_df['date'])

        inventory_data = []
        for _, row in inventory_df.iterrows():
            inventory = Inventory(**row.to_dict())
            inventory_data.append(inventory.dict())

        await db.inventory.insert_many(inventory_data)
        print(f"✓ Loaded {len(inventory_data)} inventory records")

        # Load purchase orders
        po_df = pd.read_csv(data_dir / "purchase_orders.csv")
        po_df['order_date'] = pd.to_datetime(po_df['order_date'])
        po_df['expected_delivery'] = pd.to_datetime(po_df['expected_delivery'])

        po_data = []
        for _, row in po_df.iterrows():
            po = PurchaseOrders(**row.to_dict())
            po_data.append(po.dict())

        await db.purchase_orders.insert_many(po_data)
        print(f"✓ Loaded {len(po_data)} purchase orders")

        # Load pipeline
        pipeline_df = pd.read_csv(data_dir / "pipeline.csv")
        pipeline_df['delivery_date'] = pd.to_datetime(pipeline_df['delivery_date'])

        pipeline_data = []
        for _, row in pipeline_df.iterrows():
            pipeline = Pipeline(**row.to_dict())
            pipeline_data.append(pipeline.dict())

        await db.pipeline.insert_many(pipeline_data)
        print(f"✓ Loaded {len(pipeline_data)} pipeline records")

        print("\n" + "="*50)
        print("SYNTHETIC DATA LOADED SUCCESSFULLY! 🎉")
        print("="*50)
        print(f"Database: {DATABASE_NAME}")
        print(f"Collections populated: projects, materials, historical_demand,")
        print(f"                      inventory, purchase_orders, pipeline")
        print("\nYou can now test the FastAPI endpoints with real data!")

    except Exception as e:
        print(f"❌ Error loading data: {e}")
        raise
    finally:
        client.close()

if __name__ == "__main__":
    asyncio.run(load_synthetic_data())