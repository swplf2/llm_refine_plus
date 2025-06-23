#!/usr/bin/env python3
"""
Multi-GPU LLM Refine System - Installation Verification Script
Run this script to verify that all dependencies are properly installed.
"""

import sys
import subprocess
import importlib
from typing import List, Tuple

def check_python_version() -> bool:
    """Check if Python version is 3.8 or higher."""
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✓ Python {version.major}.{version.minor}.{version.micro} (OK)")
        return True
    else:
        print(f"✗ Python {version.major}.{version.minor}.{version.micro} (Requires 3.8+)")
        return False

def check_package(package_name: str, import_name: str = None) -> bool:
    """Check if a Python package is installed and importable."""
    if import_name is None:
        import_name = package_name
    
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"✓ {package_name} {version}")
        return True
    except ImportError:
        print(f"✗ {package_name} (not installed)")
        return False

def check_command(command: str) -> bool:
    """Check if a command-line tool is available."""
    try:
        result = subprocess.run([command, '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            # Extract first line of version output
            version_line = result.stdout.split('\n')[0]
            print(f"✓ {command} ({version_line})")
            return True
        else:
            print(f"✗ {command} (command failed)")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print(f"✗ {command} (not found)")
        return False

def check_nvidia_gpu() -> bool:
    """Check for NVIDIA GPU availability."""
    try:
        result = subprocess.run(['nvidia-smi'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✓ NVIDIA GPU detected")
            return True
        else:
            print("✗ NVIDIA GPU not detected or drivers not working")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("✗ nvidia-smi not found (NVIDIA drivers not installed)")
        return False

def main():
    """Run all verification checks."""
    print("=== Multi-GPU LLM Refine System - Installation Verification ===\n")
    
    checks_passed = 0
    total_checks = 0
    
    # Core requirements
    print("Core Dependencies:")
    total_checks += 1
    if check_python_version():
        checks_passed += 1
    
    required_packages = [
        ('langchain', 'langchain'),
        ('langchain-core', 'langchain_core'),
        ('langchain-ollama', 'langchain_ollama'),
        ('langgraph', 'langgraph'),
        ('pydantic', 'pydantic'),
        ('tqdm', 'tqdm'),
        ('typing-extensions', 'typing_extensions'),
    ]
    
    for package_name, import_name in required_packages:
        total_checks += 1
        if check_package(package_name, import_name):
            checks_passed += 1
    
    print("\nOptional Dependencies:")
    optional_packages = [
        ('nvidia-ml-py3', 'pynvml'),
        ('GPUtil', 'GPUtil'),
        ('psutil', 'psutil'),
        ('loguru', 'loguru'),
        ('requests', 'requests'),
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
    ]
    
    optional_passed = 0
    for package_name, import_name in optional_packages:
        if check_package(package_name, import_name):
            optional_passed += 1
    
    print(f"\nOptional packages: {optional_passed}/{len(optional_packages)} installed")
    
    # External tools
    print("\nExternal Tools:")
    external_tools = ['ollama']
    external_passed = 0
    
    for tool in external_tools:
        if check_command(tool):
            external_passed += 1
    
    print(f"External tools: {external_passed}/{len(external_tools)} available")
    
    # GPU check
    print("\nGPU Support:")
    gpu_available = check_nvidia_gpu()
    
    # Summary
    print(f"\n=== Summary ===")
    print(f"Core dependencies: {checks_passed}/{total_checks} passed")
    print(f"Optional dependencies: {optional_passed}/{len(optional_packages)} installed")
    print(f"External tools: {external_passed}/{len(external_tools)} available")
    print(f"GPU support: {'Available' if gpu_available else 'Not available'}")
    
    if checks_passed == total_checks:
        print("\n✓ All core dependencies are installed! System is ready to use.")
        if external_passed < len(external_tools):
            print("⚠ Install missing external tools for full functionality.")
        if not gpu_available:
            print("⚠ GPU support not available. Multi-GPU features won't work.")
        return True
    else:
        print(f"\n✗ {total_checks - checks_passed} core dependencies are missing.")
        print("Run the installation script or install missing packages manually.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
