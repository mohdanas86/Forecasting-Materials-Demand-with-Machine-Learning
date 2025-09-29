from fastapi import FastAPI
from motor.motor_asyncio import AsyncIOMotorClient
from fastapi.middleware.cors import CORSMiddleware
import os
from routes.upload import router as upload_router
from routes.alerts import router as alerts_router
from routes.forecast import router as forecast_router

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# MongoDB connection
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb+srv://anas:anas@food.t6wubmw.mongodb.net/inverntory")
try:
    client = AsyncIOMotorClient(MONGODB_URL)
    database = client.forecasting_db
    print(f"Connected to MongoDB at {MONGODB_URL}")
except Exception as e:
    print(f"Warning: Could not connect to MongoDB: {e}")
    database = None

# Make database available to routes
app.state.database = database

app.include_router(upload_router)
app.include_router(alerts_router, tags=["alerts"])
app.include_router(forecast_router, tags=["forecast"])

@app.get("/")
def read_root():
    return {"message": "Hello Forecasting"}

# MongoDB connection
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb+srv://anas:anas@food.t6wubmw.mongodb.net/inverntory")
try:
    client = AsyncIOMotorClient(MONGODB_URL)
    database = client.forecasting_db
    print(f"Connected to MongoDB at {MONGODB_URL}")
except Exception as e:
    print(f"Warning: Could not connect to MongoDB: {e}")
    database = None

# Make database available to routes
app.state.database = database

app.include_router(upload_router)
app.include_router(alerts_router, tags=["alerts"])
app.include_router(forecast_router, tags=["forecast"])

@app.get("/")
def read_root():
    return {"message": "Hello Forecasting"}