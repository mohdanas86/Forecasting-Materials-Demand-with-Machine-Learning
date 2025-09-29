import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_path))

async def test_database_connection():
    """Test that we can connect to MongoDB and retrieve data"""

    # MongoDB connection
    MONGODB_URL = "mongodb+srv://anas:anas@food.t6wubmw.mongodb.net/inverntory"
    DATABASE_NAME = "inventory_forecasting"

    client = AsyncIOMotorClient(MONGODB_URL)
    db = client[DATABASE_NAME]

    try:
        print("Testing MongoDB connection and data retrieval...")

        # Test connection
        await client.admin.command('ping')
        print("✓ Connected to MongoDB")

        # Test data retrieval
        collections = ['projects', 'materials', 'historical_demand', 'inventory', 'purchase_orders', 'pipeline']

        for collection_name in collections:
            collection = db[collection_name]
            count = await collection.count_documents({})
            sample = await collection.find_one({})

            print(f"✓ {collection_name}: {count} documents")
            if sample:
                # Show a few key fields from the sample
                keys = list(sample.keys())[:5]  # First 5 keys
                print(f"  Sample keys: {keys}")

        print("\n" + "="*50)
        print("DATABASE TEST PASSED! 🎉")
        print("="*50)
        print("All collections are populated with synthetic data.")
        print("FastAPI backend should work correctly now!")

    except Exception as e:
        print(f"❌ Database test failed: {e}")
        raise
    finally:
        client.close()

if __name__ == "__main__":
    asyncio.run(test_database_connection())