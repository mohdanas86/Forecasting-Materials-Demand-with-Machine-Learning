from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional

class Projects(BaseModel):
    project_id: str
    name: str
    budget: float
    start_date: datetime
    end_date: datetime
    location: str
    tower_type: str
    substation_type: str

class Materials(BaseModel):
    material_id: str
    name: str
    unit: str
    lead_time_days: int
    tax_rate: float

class HistoricalDemand(BaseModel):
    project_id: str
    material_id: str
    date: datetime
    quantity: float

class Inventory(BaseModel):
    material_id: str
    date: datetime
    stock_level: float

class PurchaseOrders(BaseModel):
    material_id: str
    order_date: datetime
    qty: float
    expected_delivery: datetime

class Suppliers(BaseModel):
    supplier_id: str
    name: str
    location: str
    reliability_score: float

class Pipeline(BaseModel):
    project_id: str
    material_id: str
    planned_qty: float
    delivery_date: datetime

class ExternalFactors(BaseModel):
    date: datetime
    factor_type: str
    value: float