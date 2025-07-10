# Windows Deployment Guide

This guide provides step-by-step instructions for deploying the Enhanced Hotel Ranking Simulation on Windows systems.

## 🎯 Quick Start (Docker - Recommended)

### Prerequisites
- Windows 10/11 with WSL2 enabled
- Docker Desktop for Windows
- Git for Windows

### Installation Steps

1. **Install Docker Desktop**
   ```powershell
   # Download from https://www.docker.com/products/docker-desktop
   # Enable WSL2 integration during installation
   ```

2. **Clone Repository**
   ```powershell
   git clone <repository-url>
   cd enhanced_hotel_ranking
   ```

3. **Deploy with Docker Compose**
   ```powershell
   docker-compose up --build
   ```

4. **Access Application**
   - Frontend: http://localhost:3838/hotel-ranking
   - Backend API: http://localhost:8001

## 🔧 Manual Installation (Advanced)

### Prerequisites
- Python 3.11+ (from python.org)
- R 4.0+ (from r-project.org)
- Git for Windows

### Backend Setup

1. **Install Python Dependencies**
   ```powershell
   cd backend
   pip install -r requirements.txt
   ```

2. **Start Backend Server**
   ```powershell
   python main.py
   ```

### Frontend Setup

1. **Install R Packages**
   ```r
   # Open R or RStudio
   install.packages(c(
     'shiny', 'shinydashboard', 'shinyjs', 'DT', 
     'plotly', 'ggplot2', 'httr', 'jsonlite', 
     'dplyr', 'lubridate'
   ))
   ```

2. **Set Environment Variable**
   ```powershell
   $env:API_URL = "http://localhost:8001"
   ```

3. **Run Shiny Application**
   ```r
   # In R console
   shiny::runApp('frontend/app.R', host='0.0.0.0', port=3838)
   ```

## 🚀 Windows-Specific Scripts

### PowerShell Deployment Script

Create `deploy.ps1`:
```powershell
# Enhanced Hotel Ranking - Windows Deployment Script

Write-Host "Starting Enhanced Hotel Ranking Deployment..." -ForegroundColor Green

# Check if Docker is installed
if (Get-Command docker -ErrorAction SilentlyContinue) {
    Write-Host "Docker found. Using Docker deployment..." -ForegroundColor Yellow
    
    # Stop any existing containers
    docker-compose down
    
    # Build and start services
    docker-compose up --build -d
    
    Write-Host "Deployment complete!" -ForegroundColor Green
    Write-Host "Frontend: http://localhost:3838/hotel-ranking" -ForegroundColor Cyan
    Write-Host "Backend API: http://localhost:8001" -ForegroundColor Cyan
    
} else {
    Write-Host "Docker not found. Please install Docker Desktop for Windows." -ForegroundColor Red
    Write-Host "Download from: https://www.docker.com/products/docker-desktop" -ForegroundColor Yellow
}
```

### Batch File Alternative

Create `deploy.bat`:
```batch
@echo off
echo Starting Enhanced Hotel Ranking Deployment...

where docker >nul 2>nul
if %errorlevel% == 0 (
    echo Docker found. Using Docker deployment...
    docker-compose down
    docker-compose up --build -d
    echo.
    echo Deployment complete!
    echo Frontend: http://localhost:3838/hotel-ranking
    echo Backend API: http://localhost:8001
) else (
    echo Docker not found. Please install Docker Desktop for Windows.
    echo Download from: https://www.docker.com/products/docker-desktop
)
pause
```

## 🔍 Troubleshooting

### Common Windows Issues

1. **Port Already in Use**
   ```powershell
   # Find process using port 8001 or 3838
   netstat -ano | findstr :8001
   netstat -ano | findstr :3838
   
   # Kill process by PID
   taskkill /PID <PID> /F
   ```

2. **Docker Desktop Issues**
   - Ensure WSL2 is enabled
   - Restart Docker Desktop
   - Check Windows Defender firewall settings

3. **Python Path Issues**
   ```powershell
   # Add Python to PATH
   $env:PATH += ";C:\Python311;C:\Python311\Scripts"
   ```

4. **R Package Installation Issues**
   ```r
   # Use different CRAN mirror
   options(repos = c(CRAN = "https://cloud.r-project.org/"))
   
   # Install with dependencies
   install.packages("shiny", dependencies = TRUE)
   ```

### Performance Optimization

1. **Increase Docker Memory**
   - Docker Desktop → Settings → Resources
   - Increase memory allocation to 4GB+

2. **Windows Defender Exclusions**
   - Add project folder to Windows Defender exclusions
   - Exclude Docker Desktop processes

3. **WSL2 Optimization**
   ```powershell
   # Create .wslconfig in user home directory
   [wsl2]
   memory=4GB
   processors=4
   ```

## 🔒 Security Considerations

### Firewall Configuration
```powershell
# Allow applications through Windows Firewall
New-NetFirewallRule -DisplayName "Hotel Ranking Backend" -Direction Inbound -Port 8001 -Protocol TCP -Action Allow
New-NetFirewallRule -DisplayName "Hotel Ranking Frontend" -Direction Inbound -Port 3838 -Protocol TCP -Action Allow
```

### Network Access
- Backend API: http://localhost:8001
- Frontend App: http://localhost:3838/hotel-ranking
- For external access, replace `localhost` with machine IP

## 📊 Monitoring and Logs

### Docker Logs
```powershell
# View backend logs
docker-compose logs backend

# View frontend logs
docker-compose logs frontend

# Follow logs in real-time
docker-compose logs -f
```

### Manual Deployment Logs
- Backend logs: Console output from Python application
- Frontend logs: R console output and browser developer tools

## 🔄 Updates and Maintenance

### Updating the Application
```powershell
# Pull latest changes
git pull origin main

# Rebuild Docker containers
docker-compose down
docker-compose up --build -d
```

### Backup and Restore
```powershell
# Backup configuration
Copy-Item docker-compose.yml backup/
Copy-Item -Recurse data/ backup/

# Restore from backup
Copy-Item backup/docker-compose.yml .
Copy-Item -Recurse backup/data/ .
```

## 🎯 Production Deployment

### IIS Integration (Advanced)
1. Install IIS with Application Request Routing
2. Configure reverse proxy to backend API
3. Host Shiny app through Shiny Server Pro

### Windows Service Setup
```powershell
# Install NSSM (Non-Sucking Service Manager)
# Create Windows service for backend
nssm install HotelRankingBackend python.exe
nssm set HotelRankingBackend AppDirectory C:\path\to\backend
nssm set HotelRankingBackend AppParameters main.py
nssm start HotelRankingBackend
```

## 📞 Support

### Getting Help
1. Check logs for error messages
2. Verify all prerequisites are installed
3. Test individual components separately
4. Contact support with specific error details

### System Requirements
- **Minimum**: 4GB RAM, 2 CPU cores, 10GB disk space
- **Recommended**: 8GB RAM, 4 CPU cores, 20GB disk space
- **OS**: Windows 10 version 1903+ or Windows 11

### Performance Benchmarks
- Scenario generation: < 5 seconds
- Optimization: < 10 seconds for 50 offers
- MAB simulation: < 30 seconds for 100 iterations

