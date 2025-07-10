# Enhanced Hotel Ranking - Windows Deployment Script
# Automatically deploys the application using Docker

param(
    [switch]$Manual,
    [switch]$Help
)

if ($Help) {
    Write-Host @"
Enhanced Hotel Ranking Deployment Script

Usage:
  .\deploy.ps1          # Deploy using Docker (recommended)
  .\deploy.ps1 -Manual  # Manual deployment instructions
  .\deploy.ps1 -Help    # Show this help

Requirements:
  - Docker Desktop for Windows (for Docker deployment)
  - Python 3.11+ and R 4.0+ (for manual deployment)
"@ -ForegroundColor Cyan
    exit 0
}

Write-Host "🏨 Enhanced Hotel Ranking Simulation" -ForegroundColor Magenta
Write-Host "=====================================" -ForegroundColor Magenta
Write-Host ""

if ($Manual) {
    Write-Host "Manual Deployment Instructions:" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "1. Backend Setup:" -ForegroundColor Green
    Write-Host "   cd backend" -ForegroundColor Gray
    Write-Host "   pip install -r requirements.txt" -ForegroundColor Gray
    Write-Host "   python main.py" -ForegroundColor Gray
    Write-Host ""
    Write-Host "2. Frontend Setup (in new terminal):" -ForegroundColor Green
    Write-Host "   `$env:API_URL = 'http://localhost:8001'" -ForegroundColor Gray
    Write-Host "   R -e \"shiny::runApp('frontend/app.R', host='0.0.0.0', port=3838)\"" -ForegroundColor Gray
    Write-Host ""
    Write-Host "3. Access Application:" -ForegroundColor Green
    Write-Host "   Frontend: http://localhost:3838" -ForegroundColor Cyan
    Write-Host "   Backend API: http://localhost:8001" -ForegroundColor Cyan
    exit 0
}

# Check if Docker is installed
Write-Host "Checking Docker installation..." -ForegroundColor Yellow
if (Get-Command docker -ErrorAction SilentlyContinue) {
    Write-Host "✓ Docker found" -ForegroundColor Green
    
    # Check if Docker is running
    try {
        docker version | Out-Null
        Write-Host "✓ Docker is running" -ForegroundColor Green
    } catch {
        Write-Host "✗ Docker is not running. Please start Docker Desktop." -ForegroundColor Red
        exit 1
    }
    
    Write-Host ""
    Write-Host "Starting deployment..." -ForegroundColor Yellow
    
    # Stop any existing containers
    Write-Host "Stopping existing containers..." -ForegroundColor Gray
    docker-compose down 2>$null
    
    # Build and start services
    Write-Host "Building and starting services..." -ForegroundColor Gray
    docker-compose up --build -d
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "🎉 Deployment successful!" -ForegroundColor Green
        Write-Host ""
        Write-Host "Application URLs:" -ForegroundColor Cyan
        Write-Host "  Frontend: http://localhost:3838/hotel-ranking" -ForegroundColor White
        Write-Host "  Backend API: http://localhost:8001" -ForegroundColor White
        Write-Host ""
        Write-Host "To view logs: docker-compose logs -f" -ForegroundColor Gray
        Write-Host "To stop: docker-compose down" -ForegroundColor Gray
        
        # Wait a moment and test connectivity
        Write-Host ""
        Write-Host "Testing connectivity..." -ForegroundColor Yellow
        Start-Sleep -Seconds 10
        
        try {
            $response = Invoke-WebRequest -Uri "http://localhost:8001/" -TimeoutSec 5
            if ($response.StatusCode -eq 200) {
                Write-Host "✓ Backend API is responding" -ForegroundColor Green
            }
        } catch {
            Write-Host "⚠ Backend API not yet ready (this is normal, may take a few more seconds)" -ForegroundColor Yellow
        }
        
    } else {
        Write-Host ""
        Write-Host "✗ Deployment failed. Check the logs:" -ForegroundColor Red
        Write-Host "docker-compose logs" -ForegroundColor Gray
    }
    
} else {
    Write-Host "✗ Docker not found" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please install Docker Desktop for Windows:" -ForegroundColor Yellow
    Write-Host "https://www.docker.com/products/docker-desktop" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Or use manual deployment:" -ForegroundColor Yellow
    Write-Host ".\deploy.ps1 -Manual" -ForegroundColor Gray
}

