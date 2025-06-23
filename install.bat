@echo off
REM Multi-GPU LLM Refine System - Windows Installation Script
REM Run as Administrator for best results

echo === Multi-GPU LLM Refine System Installation ===

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.11+ first.
    echo Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo Found Python: 
python --version

REM Create virtual environment
echo Creating virtual environment...
python -m venv llm_refine_env
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)

REM Activate virtual environment
echo Activating virtual environment...
call llm_refine_env\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo Installing dependencies...
python -m pip install -r requirements-production.txt
if errorlevel 1 (
    echo ERROR: Failed to install requirements
    pause
    exit /b 1
)

REM Check for NVIDIA GPU
echo Checking for NVIDIA GPU support...
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo WARNING: NVIDIA GPU or drivers not detected.
    echo Multi-GPU features may not work properly.
) else (
    echo NVIDIA GPU detected!
)

REM Check Ollama
echo Checking Ollama installation...
ollama --version >nul 2>&1
if errorlevel 1 (
    echo WARNING: Ollama not found.
    echo Please install Ollama from: https://ollama.ai/
    echo After installation, pull required models with:
    echo   ollama pull llama2
    echo   ollama pull mistral
) else (
    echo Ollama is installed!
)

echo.
echo === Installation Complete ===
echo Next steps:
echo 1. Install Ollama if not already installed
echo 2. Pull required language models
echo 3. Run the demo: python demo_multi_gpu.py
echo 4. Check README.md for usage examples
echo.
echo To activate the environment in future sessions:
echo   llm_refine_env\Scripts\activate.bat
echo.
pause
