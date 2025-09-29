#!/usr/bin/env node

// Simple test script to verify CORS and API connectivity
import axios from 'axios';

const API_BASE = 'http://localhost:8000';

async function testCORS() {
    console.log('🔍 Testing CORS and API connectivity...\n');

    try {
        // Test root endpoint
        console.log('📡 Testing root endpoint...');
        const rootResponse = await axios.get(`${API_BASE}/`);
        console.log('✅ Root endpoint accessible:', rootResponse.data);

        // Test forecast endpoint
        console.log('\n📊 Testing forecast endpoint...');
        const forecastResponse = await axios.get(`${API_BASE}/forecast`);
        console.log('✅ Forecast endpoint accessible');
        console.log(`   📈 Retrieved ${forecastResponse.data.forecasts?.length || 0} forecast records`);

        // Test alerts endpoint
        console.log('\n🚨 Testing alerts endpoint...');
        const alertsResponse = await axios.get(`${API_BASE}/alerts`);
        console.log('✅ Alerts endpoint accessible');
        console.log(`   🚨 Retrieved ${alertsResponse.data?.length || 0} alerts`);

        console.log('\n🎉 All API endpoints are accessible! CORS is working correctly.');

    } catch (error) {
        console.error('❌ API test failed:');
        if (error.code === 'ERR_NETWORK') {
            console.error('   Network error - make sure the backend is running on http://localhost:8000');
        } else if (error.response) {
            console.error(`   HTTP ${error.response.status}: ${error.response.statusText}`);
            console.error('   Response:', error.response.data);
        } else {
            console.error('   Error:', error.message);
        }
        console.log('\n💡 Troubleshooting tips:');
        console.log('   1. Make sure the FastAPI backend is running: uvicorn main:app --reload');
        console.log('   2. Check that CORS middleware is properly configured in main.py');
        console.log('   3. Verify the backend is accessible at http://localhost:8000');
    }
}

testCORS();