from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from typing import List, Dict, Any
import csv
import json
import io
from datetime import datetime
from models import Projects, Materials, HistoricalDemand, Inventory
import pandas as pd
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from ml.data_cleaning import clean_historical_demand_data

router = APIRouter()

def parse_file(file: UploadFile) -> List[Dict[str, Any]]:
    """Parse uploaded file (CSV or JSON) into list of dictionaries"""
    content = file.file.read().decode('utf-8')
    file.file.seek(0)  # Reset file pointer

    if file.filename.endswith('.csv'):
        # Parse CSV
        reader = csv.DictReader(io.StringIO(content))
        return list(reader)
    elif file.filename.endswith('.json'):
        # Parse JSON
        data = json.loads(content)
        if isinstance(data, list):
            return data
        else:
            raise HTTPException(status_code=400, detail="JSON file must contain an array of objects")
    else:
        raise HTTPException(status_code=400, detail="File must be CSV or JSON")

def validate_and_create_models(data: List[Dict[str, Any]], model_class):
    """Validate data and create model instances"""
    valid_models = []
    failed_rows = []

    for i, row in enumerate(data):
        try:
            # Convert string dates to datetime objects if needed
            if model_class == Projects:
                if 'start_date' in row and isinstance(row['start_date'], str):
                    row['start_date'] = datetime.fromisoformat(row['start_date'])
                if 'end_date' in row and isinstance(row['end_date'], str):
                    row['end_date'] = datetime.fromisoformat(row['end_date'])
                if 'budget' in row:
                    row['budget'] = float(row['budget'])
            elif model_class in [HistoricalDemand, Inventory]:
                if 'date' in row and isinstance(row['date'], str):
                    row['date'] = datetime.fromisoformat(row['date'])
                if 'quantity' in row:
                    row['quantity'] = float(row['quantity'])
                if 'stock_level' in row:
                    row['stock_level'] = float(row['stock_level'])
            elif model_class == Materials:
                if 'lead_time_days' in row:
                    row['lead_time_days'] = int(row['lead_time_days'])
                if 'tax_rate' in row:
                    row['tax_rate'] = float(row['tax_rate'])

            model = model_class(**row)
            valid_models.append(model)
        except Exception as e:
            failed_rows.append({"row": i + 1, "error": str(e), "data": row})

    return valid_models, failed_rows

@router.post("/upload/projects")
async def upload_projects(request: Request, file: UploadFile = File(...)):
    try:
        # Test database connection
        await request.app.state.database.command("ping")
    except Exception:
        raise HTTPException(status_code=503, detail="Database not available")
    
    data = parse_file(file)
    valid_models, failed_rows = validate_and_create_models(data, Projects)

    # Insert valid models using Motor
    inserted_count = 0
    if valid_models:
        try:
            collection = request.app.state.database.projects
            documents = [model.dict() for model in valid_models]
            result = await collection.insert_many(documents)
            inserted_count = len(result.inserted_ids)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Database insertion failed: {str(e)}")

    return {
        "rows_received": len(data),
        "rows_inserted": inserted_count,
        "rows_failed": len(failed_rows),
        "failed_details": failed_rows
    }

@router.post("/upload/materials")
async def upload_materials(request: Request, file: UploadFile = File(...)):
    try:
        await request.app.state.database.command("ping")
    except Exception:
        raise HTTPException(status_code=503, detail="Database not available")
    
    data = parse_file(file)
    valid_models, failed_rows = validate_and_create_models(data, Materials)

    inserted_count = 0
    if valid_models:
        try:
            collection = request.app.state.database.materials
            documents = [model.dict() for model in valid_models]
            result = await collection.insert_many(documents)
            inserted_count = len(result.inserted_ids)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Database insertion failed: {str(e)}")

    return {
        "rows_received": len(data),
        "rows_inserted": inserted_count,
        "rows_failed": len(failed_rows),
        "failed_details": failed_rows
    }

@router.post("/upload/historical")
async def upload_historical_demand(request: Request, file: UploadFile = File(...), clean_data: bool = True):
    try:
        await request.app.state.database.command("ping")
    except Exception:
        raise HTTPException(status_code=503, detail="Database not available")
    
    data = parse_file(file)
    
    # Apply data cleaning if requested
    if clean_data:
        df = pd.DataFrame(data)
        df = clean_historical_demand_data(df)
        data = df.to_dict('records')
    
    valid_models, failed_rows = validate_and_create_models(data, HistoricalDemand)

    inserted_count = 0
    if valid_models:
        try:
            collection = request.app.state.database.historical_demand
            documents = [model.dict() for model in valid_models]
            result = await collection.insert_many(documents)
            inserted_count = len(result.inserted_ids)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Database insertion failed: {str(e)}")

    return {
        "rows_received": len(data),
        "rows_inserted": inserted_count,
        "rows_failed": len(failed_rows),
        "failed_details": failed_rows,
        "data_cleaned": clean_data
    }

@router.post("/upload/inventory")
async def upload_inventory(request: Request, file: UploadFile = File(...)):
    try:
        await request.app.state.database.command("ping")
    except Exception:
        raise HTTPException(status_code=503, detail="Database not available")
    
    data = parse_file(file)
    valid_models, failed_rows = validate_and_create_models(data, Inventory)

    inserted_count = 0
    if valid_models:
        try:
            collection = request.app.state.database.inventory
            documents = [model.dict() for model in valid_models]
            result = await collection.insert_many(documents)
            inserted_count = len(result.inserted_ids)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Database insertion failed: {str(e)}")

    return {
        "rows_received": len(data),
        "rows_inserted": inserted_count,
        "rows_failed": len(failed_rows),
        "failed_details": failed_rows
    }