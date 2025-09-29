# Docker Setup for POWERGRID Inventory Forecasting

This directory contains Docker configuration for running the POWERGRID inventory forecasting system.

## Services

### Backend (FastAPI)
- **Port**: 8000
- **Technology**: Python 3.9, FastAPI, Uvicorn
- **Database**: MongoDB Atlas (cloud)
- **Features**: REST API, ML model serving, data processing

### Frontend (React)
- **Port**: 5173
- **Technology**: Node.js 18, React 18, Vite
- **Features**: Modern UI, charts, file uploads, real-time alerts

## Quick Start

1. **Clone and navigate to the project root:**
   ```bash
   cd /path/to/inventory-forecasting
   ```

2. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env with your MongoDB Atlas connection string
   ```

3. **Start all services:**
   ```bash
   cd docker
   docker-compose up --build
   ```

4. **Access the applications:**
   - Frontend: `http://localhost:5173`
   - Backend API: `http://localhost:8000`
   - API Documentation: `http://localhost:8000/docs`

## Development Workflow

### Using Docker for Development

The Docker setup is configured for development with:
- **Hot reload** for both frontend and backend
- **Volume mounting** for live code changes
- **Development servers** instead of production builds

### Making Code Changes

1. **Backend changes**: Edit files in `../backend/` - changes are reflected immediately
2. **Frontend changes**: Edit files in `../frontend/` - changes are reflected immediately
3. **ML changes**: Edit files in `../ml/` - models are reloaded on backend restart

### Environment Variables

Create a `.env` file in the project root:

```env
MONGODB_URL=mongodb+srv://your-username:your-password@your-cluster.mongodb.net/your-database
```

## Docker Commands

### Start Services
```bash
docker-compose up --build
```

### Start in Background
```bash
docker-compose up -d --build
```

### Stop Services
```bash
docker-compose down
```

### View Logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f backend
docker-compose logs -f frontend
```

### Rebuild Specific Service
```bash
docker-compose up --build backend
docker-compose up --build frontend
```

### Clean Up
```bash
# Stop and remove containers
docker-compose down

# Remove volumes (WARNING: deletes data)
docker-compose down -v

# Remove images
docker-compose down --rmi all
```

## Troubleshooting

### Backend Issues
- **Connection refused**: Check MongoDB Atlas connection string
- **Import errors**: Ensure all Python dependencies are installed
- **Port already in use**: Stop other services using port 8000

### Frontend Issues
- **Port already in use**: Stop other services using port 5173
- **Build errors**: Clear node_modules and reinstall
- **CORS errors**: Backend CORS is configured to allow all origins

### General Issues
- **Permission errors**: Ensure Docker has access to project files
- **Memory issues**: Increase Docker memory allocation
- **Network issues**: Check firewall settings

## Production Deployment

For production deployment, consider:
1. Using production Dockerfiles with multi-stage builds
2. Setting up proper environment variables
3. Using Docker secrets for sensitive data
4. Setting up reverse proxy (nginx)
5. Configuring SSL certificates

## File Structure

```
docker/
├── docker-compose.yml    # Multi-service configuration
├── Dockerfile.backend    # Backend container definition
└── Dockerfile.frontend   # Frontend container definition
```

## Ports

- **8000**: FastAPI backend API
- **5173**: Vite development server (frontend)

## Volumes

- Backend code is mounted for live development
- ML models and data directories are mounted
- Frontend code is mounted for live development
- Node modules are cached in a Docker volume