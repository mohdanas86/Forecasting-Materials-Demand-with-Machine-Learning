#!/usr/bin/env python3
"""
Test script to verify all FastAPI routes are properly implemented and working
"""

import requests
import json
from datetime import datetime

BASE_URL = "http://localhost:8000"

def test_route(endpoint, method="GET", data=None, headers=None):
    """Test a single route"""
    try:
        if method == "GET":
            response = requests.get(f"{BASE_URL}{endpoint}", headers=headers)
        elif method == "POST":
            response = requests.post(f"{BASE_URL}{endpoint}", data=data, headers=headers)
        else:
            print(f"❌ Unsupported method: {method}")
            return False

        if response.status_code == 200:
            print(f"✅ {method} {endpoint} - Status: {response.status_code}")
            return True, response.json()
        else:
            print(f"❌ {method} {endpoint} - Status: {response.status_code} - {response.text}")
            return False, None
    except Exception as e:
        print(f"❌ {method} {endpoint} - Error: {str(e)}")
        return False, None

def main():
    print("🚀 Testing FastAPI Routes for POWERGRID Inventory Forecasting")
    print("=" * 60)

    # Test root endpoint
    print("\n📍 Testing root endpoint...")
    success, _ = test_route("/")
    if not success:
        print("❌ Root endpoint failed, server may not be running")
        return

    # Test forecast endpoints
    print("\n📊 Testing forecast endpoints...")

    # GET /forecast
    success, data = test_route("/forecast")
    if success and data:
        forecasts = data.get("forecasts", [])
        print(f"   📈 Retrieved {len(forecasts)} forecast records")
        if forecasts:
            sample = forecasts[0]
            required_fields = ["material_id", "p10", "p50", "p90", "safety_stock", "reorder_point", "recommendations"]
            missing_fields = [field for field in required_fields if field not in sample]
            if missing_fields:
                print(f"   ⚠️  Missing fields in forecast response: {missing_fields}")
            else:
                print("   ✅ All required forecast fields present")

    # GET /forecast?period=week
    test_route("/forecast?period=week")

    # GET /forecast/summary
    success, data = test_route("/forecast/summary")
    if success and data:
        print(f"   📊 Forecast summary: {data.get('total_materials', 0)} materials, {data.get('total_forecast_points', 0)} points")

    # Test alerts endpoints
    print("\n🚨 Testing alerts endpoints...")

    # GET /alerts
    success, data = test_route("/alerts")
    if success and isinstance(data, list):
        print(f"   🚨 Retrieved {len(data)} alerts")
        if data:
            sample = data[0]
            required_fields = ["alert_id", "alert_type", "severity", "material_id", "title", "description", "recommended_action"]
            missing_fields = [field for field in required_fields if field not in sample]
            if missing_fields:
                print(f"   ⚠️  Missing fields in alert response: {missing_fields}")
            else:
                print("   ✅ All required alert fields present")

    # GET /alerts/summary
    success, data = test_route("/alerts/summary")
    if success and data:
        print(f"   📊 Alert summary: {data.get('total_alerts', 0)} total alerts")

    # GET /alerts/types
    success, data = test_route("/alerts/types")
    if success and data:
        print(f"   📋 Alert types: {list(data.keys())}")

    # Test upload endpoints (without actual file uploads - just check if endpoints exist)
    print("\n📤 Testing upload endpoints...")

    # These would normally require file uploads, but we can test the endpoint availability
    # by checking if they return proper error messages for missing files
    endpoints_to_test = [
        "/upload/projects",
        "/upload/materials",
        "/upload/historical",
        "/upload/inventory"
    ]

    for endpoint in endpoints_to_test:
        try:
            response = requests.post(f"{BASE_URL}{endpoint}")
            if response.status_code in [400, 422]:  # Expected when no file provided
                print(f"✅ POST {endpoint} - Status: {response.status_code} (expected for missing file)")
            else:
                print(f"❌ POST {endpoint} - Status: {response.status_code}")
        except Exception as e:
            print(f"❌ POST {endpoint} - Error: {str(e)}")

    # Test JWT authentication (stub)
    print("\n🔐 Testing JWT authentication stub...")

    # Test with invalid token
    headers = {"Authorization": "Bearer invalid_token"}
    success, _ = test_route("/forecast", headers=headers)
    if not success:
        print("   ✅ JWT validation working (rejected invalid token)")

    # Test with valid token (stub accepts any token >= 10 chars)
    headers = {"Authorization": "Bearer valid_stub_token_12345"}
    success, _ = test_route("/forecast", headers=headers)
    if success:
        print("   ✅ JWT stub accepting valid tokens")

    print("\n" + "=" * 60)
    print("🎉 Route testing completed!")
    print("\n📋 Summary of implemented endpoints:")
    print("   • GET  /forecast - Forecast data with p10/p50/p90, safety stock, reorder point, recommendations")
    print("   • GET  /forecast/summary - Forecast summary statistics")
    print("   • GET  /alerts - Inventory alerts with filtering")
    print("   • GET  /alerts/summary - Alert summary statistics")
    print("   • GET  /alerts/types - Alert type information")
    print("   • POST /upload/projects - Upload project data")
    print("   • POST /upload/materials - Upload material data")
    print("   • POST /upload/historical - Upload historical demand data")
    print("   • POST /upload/inventory - Upload inventory data")
    print("   • JWT Authentication stub implemented")

if __name__ == "__main__":
    main()