import axios from 'axios';

// Create axios instance with base configuration
const api = axios.create({
    baseURL: 'http://localhost:8000', // Direct backend URL (CORS now allows all origins)
    timeout: 30000, // 30 seconds timeout
    headers: {
        'Content-Type': 'application/json',
    },
    withCredentials: false, // Set to true if using cookies/auth
});

// Request interceptor for adding auth headers if needed
api.interceptors.request.use(
    (config) => {
        // Add JWT token if available (for future authentication)
        const token = localStorage.getItem('authToken');
        if (token) {
            config.headers.Authorization = `Bearer ${token}`;
        }
        return config;
    },
    (error) => {
        return Promise.reject(error);
    }
);

// Response interceptor for error handling
api.interceptors.response.use(
    (response) => response,
    (error) => {
        if (error.response?.status === 401) {
            // Handle unauthorized access
            console.error('Unauthorized access - please check authentication');
        } else if (error.response?.status >= 500) {
            console.error('Server error:', error.response.data);
        } else if (!error.response && error.code === 'ERR_NETWORK') {
            // CORS or network error
            console.error('Network/CORS error - please check if backend is running on http://localhost:8000');
            console.error('Make sure the backend has CORS middleware enabled');
        }
        return Promise.reject(error);
    }
);

// API service functions
export const apiService = {
    // Forecast endpoints
    getForecast: (params = {}) => api.get('/forecast', { params }),
    getForecastSummary: () => api.get('/forecast/summary'),

    // Alerts endpoints
    getAlerts: (params = {}) => api.get('/alerts', { params }),
    getAlertsSummary: () => api.get('/alerts/summary'),
    getAlertTypes: () => api.get('/alerts/types'),

    // Upload endpoints
    uploadProjects: (file) => {
        const formData = new FormData();
        formData.append('file', file);
        return api.post('/upload/projects', formData, {
            headers: { 'Content-Type': 'multipart/form-data' }
        });
    },

    uploadMaterials: (file) => {
        const formData = new FormData();
        formData.append('file', file);
        return api.post('/upload/materials', formData, {
            headers: { 'Content-Type': 'multipart/form-data' }
        });
    },

    uploadHistorical: (file, cleanData = true) => {
        const formData = new FormData();
        formData.append('file', file);
        return api.post('/upload/historical', formData, {
            headers: { 'Content-Type': 'multipart/form-data' },
            params: { clean_data: cleanData }
        });
    },

    uploadInventory: (file) => {
        const formData = new FormData();
        formData.append('file', file);
        return api.post('/upload/inventory', formData, {
            headers: { 'Content-Type': 'multipart/form-data' }
        });
    }
};

export default api;