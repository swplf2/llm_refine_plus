# Multi-GPU LLM Refine System - Windows Installation Script
# Run in PowerShell as Administrator for best results

param(
    [string]$Environment = "production",
    [switch]$CreateVenv = $true,
    [string]$PythonVersion = "3.11"
)

Write-Host "=== Multi-GPU LLM Refine System Installation ===" -ForegroundColor Green
Write-Host "Environment: $Environment" -ForegroundColor Yellow
Write-Host "Python Version: $PythonVersion" -ForegroundColor Yellow

# Check if Python is installed
try {
    $pythonCmd = "python$PythonVersion"
    if (!(Get-Command $pythonCmd -ErrorAction SilentlyContinue)) {
        $pythonCmd = "python"
    }
    
    $pythonVersion = & $pythonCmd --version
    Write-Host "Found Python: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Python not found. Please install Python $PythonVersion first." -ForegroundColor Red
    Write-Host "Download from: https://www.python.org/downloads/" -ForegroundColor Yellow
    exit 1
}

# Create virtual environment if requested
if ($CreateVenv) {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    & $pythonCmd -m venv llm_refine_env
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Failed to create virtual environment" -ForegroundColor Red
        exit 1
    }
    
    # Activate virtual environment
    Write-Host "Activating virtual environment..." -ForegroundColor Yellow
    & ".\llm_refine_env\Scripts\Activate.ps1"
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Failed to activate virtual environment" -ForegroundColor Red
        exit 1
    }
    
    Write-Host "Virtual environment activated successfully!" -ForegroundColor Green
}

# Upgrade pip
Write-Host "Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# Install requirements based on environment
$requirementsFile = switch ($Environment.ToLower()) {
    "minimal" { "requirements-minimal.txt" }
    "production" { "requirements-production.txt" }
    "dev" { "requirements-dev.txt" }
    "development" { "requirements-dev.txt" }
    default { "requirements.txt" }
}

Write-Host "Installing dependencies from $requirementsFile..." -ForegroundColor Yellow
python -m pip install -r $requirementsFile

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to install requirements" -ForegroundColor Red
    exit 1
}

# Check if NVIDIA drivers are available (optional)
Write-Host "Checking for NVIDIA GPU support..." -ForegroundColor Yellow
try {
    nvidia-smi
    Write-Host "NVIDIA GPU detected!" -ForegroundColor Green
} catch {
    Write-Host "WARNING: NVIDIA GPU or drivers not detected. Multi-GPU features may not work." -ForegroundColor Yellow
    Write-Host "Install NVIDIA drivers and CUDA toolkit if you have NVIDIA GPUs." -ForegroundColor Yellow
}

# Check Ollama installation
Write-Host "Checking Ollama installation..." -ForegroundColor Yellow
try {
    ollama --version
    Write-Host "Ollama is installed!" -ForegroundColor Green
} catch {
    Write-Host "WARNING: Ollama not found." -ForegroundColor Yellow
    Write-Host "Please install Ollama from: https://ollama.ai/" -ForegroundColor Yellow
    Write-Host "After installation, pull required models with:" -ForegroundColor Yellow
    Write-Host "  ollama pull llama2" -ForegroundColor Cyan
    Write-Host "  ollama pull mistral" -ForegroundColor Cyan
}

Write-Host "`n=== Installation Complete ===" -ForegroundColor Green
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Install Ollama if not already installed" -ForegroundColor White
Write-Host "2. Pull required language models" -ForegroundColor White
Write-Host "3. Run the demo: python demo_multi_gpu.py" -ForegroundColor White
Write-Host "4. Check README.md for usage examples" -ForegroundColor White

if ($CreateVenv) {
    Write-Host "`nTo activate the environment in future sessions:" -ForegroundColor Yellow
    Write-Host "  .\llm_refine_env\Scripts\Activate.ps1" -ForegroundColor Cyan
}
