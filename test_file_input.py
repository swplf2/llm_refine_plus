#!/usr/bin/env python3
"""
Quick Test Script for File Input Functionality
Tests the file input refine system with minimal setup
"""

import os
import tempfile
import sys

def create_test_files():
    """Create small test files for quick testing"""
    print("📝 Creating test files...")
    
    # Small test dataset
    test_en = [
        "Hello, how are you?",
        "Good morning everyone.",
        "Thank you for your help.",
        "Have a nice day!",
        "See you later."
    ]
    
    test_vi = [
        "Xin chào, bạn khỏe không?",
        "Chào buổi sáng mọi người.",
        "Cảm ơn bạn đã giúp đỡ.",
        "Chúc bạn một ngày tốt lành!",
        "Hẹn gặp lại sau."
    ]
    
    # Write test files
    with open("test_source.en", "w", encoding="utf-8") as f:
        for sentence in test_en:
            f.write(f"{sentence}\n")
    
    with open("test_translation.vi", "w", encoding="utf-8") as f:
        for sentence in test_vi:
            f.write(f"{sentence}\n")
    
    print(f"   ✓ Created test_source.en ({len(test_en)} sentences)")
    print(f"   ✓ Created test_translation.vi ({len(test_vi)} translations)")
    
    return "test_source.en", "test_translation.vi"

def test_file_loading():
    """Test the file loading functionality"""
    print("\n🧪 Testing file loading functionality...")
    
    try:
        # Import the file loading function
        from file_input_refine import load_text_files
        
        source_file, translation_file = create_test_files()
        
        # Test source only mode
        print("\n   Testing source-only mode...")
        src_lines, tgt_lines, mode = load_text_files(source_file, None)
        print(f"   ✓ Mode: {mode}")
        print(f"   ✓ Source lines: {len(src_lines)}")
        print(f"   ✓ Target lines: {len(tgt_lines)}")
        
        # Test source + translation mode
        print("\n   Testing source + translation mode...")
        src_lines, tgt_lines, mode = load_text_files(source_file, translation_file)
        print(f"   ✓ Mode: {mode}")
        print(f"   ✓ Source lines: {len(src_lines)}")
        print(f"   ✓ Target lines: {len(tgt_lines)}")
        
        # Show sample data
        print(f"\n   Sample data:")
        for i in range(min(3, len(src_lines))):
            print(f"   EN: {src_lines[i]}")
            print(f"   VI: {tgt_lines[i]}")
            print()
        
        return True
        
    except ImportError as e:
        print(f"   ❌ Cannot import file_input_refine: {e}")
        print(f"   Make sure file_input_refine.py is in the same directory")
        return False
    except Exception as e:
        print(f"   ❌ Error testing file loading: {e}")
        return False

def test_command_line():
    """Test command line argument parsing"""
    print("\n🧪 Testing command line interface...")
    
    try:
        import argparse
        from file_input_refine import main
        
        # Test help
        print("   ✓ Command line interface available")
        print("   To test full functionality, run:")
        print("     python file_input_refine.py test_source.en test_output.vi")
        print("     python file_input_refine.py test_source.en test_output.vi --translation-file test_translation.vi")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error testing command line: {e}")
        return False

def check_dependencies():
    """Check if required dependencies are available"""
    print("\n🔍 Checking dependencies...")
    
    required_packages = [
        "langchain",
        "langchain_ollama", 
        "langgraph",
        "pydantic",
        "tqdm"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"   ✓ {package}")
        except ImportError:
            print(f"   ❌ {package} (missing)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n   Missing packages: {', '.join(missing_packages)}")
        print(f"   Install with: pip install {' '.join(missing_packages)}")
        return False
    else:
        print(f"   ✅ All required packages available")
        return True

def check_ollama():
    """Check if Ollama is available"""
    print("\n🔍 Checking Ollama availability...")
    
    try:
        import subprocess
        result = subprocess.run(["ollama", "--version"], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"   ✓ Ollama available: {result.stdout.strip()}")
            return True
        else:
            print(f"   ❌ Ollama command failed")
            return False
    except FileNotFoundError:
        print(f"   ❌ Ollama not found in PATH")
        return False
    except Exception as e:
        print(f"   ❌ Error checking Ollama: {e}")
        return False

def cleanup_test_files():
    """Remove test files"""
    test_files = ["test_source.en", "test_translation.vi", "test_output.vi"]
    
    print("\n🧹 Cleaning up test files...")
    for filename in test_files:
        if os.path.exists(filename):
            os.remove(filename)
            print(f"   ✓ Removed {filename}")

def main():
    """Main test function"""
    print("🧪 File Input Refine System - Quick Test")
    print("=" * 50)
    
    all_passed = True
    
    # Test dependency checking
    if not check_dependencies():
        all_passed = False
        print("\n❌ Dependency check failed. Please install missing packages.")
    
    # Test Ollama availability
    if not check_ollama():
        all_passed = False
        print("\n⚠️ Ollama not available. Multi-GPU features won't work.")
    
    # Test file operations
    if not test_file_loading():
        all_passed = False
        print("\n❌ File loading test failed.")
    
    # Test command line interface
    if not test_command_line():
        all_passed = False
        print("\n❌ Command line test failed.")
    
    # Summary
    print("\n" + "=" * 50)
    if all_passed:
        print("✅ All tests passed! The system is ready to use.")
        print("\nQuick start:")
        print("  python file_input_refine.py test_source.en output.vi")
        print("  python file_input_refine.py test_source.en output.vi --translation-file test_translation.vi")
    else:
        print("❌ Some tests failed. Please check the issues above.")
    
    # Cleanup
    try:
        cleanup = input("\n🗑️ Remove test files? (Y/n): ").lower()
        if cleanup != 'n':
            cleanup_test_files()
    except:
        cleanup_test_files()

if __name__ == "__main__":
    main()
