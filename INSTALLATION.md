# Installation Guide for Multi-GPU LLM Refine System

## Automated Installation (Recommended)

### Windows PowerShell (Recommended)
```powershell
# Run as Administrator for best results
.\install.ps1

# Or specify environment type
.\install.ps1 -Environment "production"
.\install.ps1 -Environment "dev"
.\install.ps1 -Environment "minimal"
```

### Windows Batch File
```cmd
# Run as Administrator
install.bat
```

### Verify Installation
```bash
# Check if everything is installed correctly
python verify_installation.py
```

## Manual Installation Options

### 1. Minimal Installation (Core Features Only)
```bash
pip install -r requirements-minimal.txt
```

### 2. Production Installation (Recommended)
```bash
pip install -r requirements-production.txt
```

### 3. Full Installation (All Features)
```bash
pip install -r requirements.txt
```

### 4. Development Installation
```bash
pip install -r requirements-dev.txt
```

## System Requirements

### Hardware Requirements
- **Minimum**: 1x NVIDIA GPU with 8GB+ VRAM
- **Recommended**: 4x NVIDIA RTX 3090/4090 or equivalent
- **Memory**: 16GB+ system RAM (32GB+ recommended)
- **Storage**: NVMe SSD for best I/O performance

### Software Requirements
- **Python**: 3.8+ (3.10+ recommended)
- **CUDA**: 11.0+ (11.8+ recommended)
- **Ollama**: Latest version from https://ollama.ai/

## Pre-Installation Steps

### 1. Install Ollama
```bash
# Windows: Download from https://ollama.ai/
# Linux/macOS:
curl -fsSL https://ollama.ai/install.sh | sh
```

### 2. Download Required Models
```bash
# Primary model (recommended)
ollama pull llama3.1:8b-instruct-fp16

# Alternative models
ollama pull llama3.1:8b-instruct-q4_0  # For memory-constrained setups
ollama pull llama2:13b-chat             # For higher quality
```

### 3. Verify GPU Setup
```bash
# Check NVIDIA drivers
nvidia-smi

# Check CUDA installation
python -c "import torch; print(torch.cuda.is_available())"
```

## Installation Verification

### Test Basic Installation
```python
# test_installation.py
try:
    from langchain_ollama import ChatOllama
    from langgraph.graph import StateGraph
    from pydantic import BaseModel
    print("✅ All core dependencies installed successfully!")
    
    # Test Ollama connection
    llm = ChatOllama(model="llama3.1:8b-instruct-fp16")
    response = llm.invoke("Test connection")
    print("✅ Ollama connection successful!")
    
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
except Exception as e:
    print(f"❌ Connection error: {e}")
    print("Make sure Ollama is running and the model is downloaded")
```

### Test Multi-GPU Setup
```python
# test_multi_gpu.py
from optimized_hybrid_refine import MultiGPUOllamaManager

manager = MultiGPUOllamaManager(num_gpus=2)
try:
    active_gpus = manager.setup_multi_gpu_ollama()
    print(f"✅ {active_gpus} GPUs configured successfully!")
finally:
    manager.cleanup()
```

## Troubleshooting Installation Issues

### Common Issues and Solutions

#### 1. LangChain Import Errors
```bash
# Update to latest versions
pip install --upgrade langchain langchain-core langchain-ollama

# If conflicts, try fresh install
pip uninstall langchain langchain-core langchain-ollama
pip install langchain langchain-core langchain-ollama
```

#### 2. Ollama Connection Issues
```bash
# Start Ollama service
ollama serve

# Check if model exists
ollama list

# Pull model if missing
ollama pull llama3.1:8b-instruct-fp16
```

#### 3. CUDA/GPU Issues
```bash
# Reinstall PyTorch with CUDA
pip uninstall torch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### 4. Pydantic Version Conflicts
```bash
# Use specific compatible version
pip install "pydantic>=2.0.0,<3.0.0"
```

#### 5. Memory Issues During Installation
```bash
# Install packages one by one
pip install langchain
pip install langchain-core
pip install langchain-ollama
pip install langgraph
pip install pydantic
pip install tqdm
```

## Virtual Environment Setup (Recommended)

### Using venv
```bash
# Create virtual environment
python -m venv llm_refine_env

# Activate (Windows)
llm_refine_env\Scripts\activate

# Activate (Linux/macOS)  
source llm_refine_env/bin/activate

# Install requirements
pip install -r requirements-minimal.txt
```

### Using conda
```bash
# Create conda environment
conda create -n llm_refine python=3.10

# Activate environment
conda activate llm_refine

# Install requirements
pip install -r requirements-minimal.txt
```

## Docker Setup (Alternative)

### Dockerfile Example
```dockerfile
FROM nvidia/cuda:11.8-devel-ubuntu20.04

# Install Python and pip
RUN apt-get update && apt-get install -y python3 python3-pip

# Copy requirements
COPY requirements-minimal.txt .

# Install Python dependencies
RUN pip3 install -r requirements-minimal.txt

# Install Ollama
RUN curl -fsSL https://ollama.ai/install.sh | sh

# Copy application code
COPY . /app
WORKDIR /app

# Expose ports for multi-GPU setup
EXPOSE 11434-11440

CMD ["python3", "demo_multi_gpu.py"]
```

### Docker Compose
```yaml
version: '3.8'
services:
  llm-refine:
    build: .
    ports:
      - "11434-11440:11434-11440"
    environment:
      - CUDA_VISIBLE_DEVICES=0,1,2,3
    runtime: nvidia
    volumes:
      - ./data:/app/data
      - ./output:/app/output
```

## Performance Optimization Tips

### 1. GPU Memory Optimization
```bash
# Use memory-efficient model variants
ollama pull llama3.1:8b-instruct-q4_0
```

### 2. System Optimization
```bash
# Increase file descriptor limits (Linux)
ulimit -n 65536

# Set GPU persistence mode
sudo nvidia-smi -pm 1
```

### 3. Python Optimization
```bash
# Use faster JSON library (optional)
pip install orjson

# Use faster regex library (optional)  
pip install regex
```

## Getting Help

If you encounter issues:

1. **Check the examples.md** for detailed usage examples
2. **Run the demo script**: `python demo_multi_gpu.py --requirements`
3. **Check Ollama logs**: `ollama logs`
4. **Verify GPU status**: `nvidia-smi`
5. **Test individual components** using the verification scripts above

## Version Compatibility

| Component | Minimum Version | Recommended Version |
|-----------|----------------|-------------------|
| Python | 3.8 | 3.10+ |
| LangChain | 0.1.0 | Latest |
| LangGraph | 0.0.40 | Latest |
| Pydantic | 2.0.0 | Latest |
| CUDA | 11.0 | 11.8+ |
| Ollama | Any | Latest |

## Requirements Files Overview

This project includes several requirements files for different use cases:

| File | Description | Use Case |
|------|-------------|----------|
| `requirements-minimal.txt` | Core dependencies only | Testing, CI/CD, minimal deployments |
| `requirements-production.txt` | Production-ready with monitoring | Production deployments, recommended |
| `requirements.txt` | Full feature set | Development and full functionality |
| `requirements-dev.txt` | Development tools included | Contributors and development |

### Choosing the Right Requirements File

- **New users**: Start with `requirements-production.txt`
- **Production deployment**: Use `requirements-production.txt`
- **Development/Contributing**: Use `requirements-dev.txt`
- **Minimal testing**: Use `requirements-minimal.txt`
- **All features**: Use `requirements.txt`
