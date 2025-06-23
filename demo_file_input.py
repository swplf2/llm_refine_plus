"""
Demo script for Multi-GPU LLM Refine System with File Input Support
Shows how to use the system with source files and existing translation files
"""

import os
import sys
from file_input_refine import run_multi_gpu_llm_refine_with_files

def create_sample_files():
    """Create sample input files for demonstration"""
    print("📝 Creating sample input files...")
    
    # Sample English sentences
    sample_en = [
        "Hello, how are you today?",
        "The weather is beautiful this morning.",
        "I would like to learn Vietnamese language.",
        "Machine translation has improved significantly in recent years.",
        "Artificial intelligence is transforming many industries.",
        "Thank you for your help and support.",
        "Programming requires patience and practice.",
        "The sunset over the mountains was breathtaking.",
        "Education is the key to personal development.",
        "Technology connects people around the world."
    ]
    
    # Sample Vietnamese translations (some good, some could be improved)
    sample_vi = [
        "Xin chào, hôm nay bạn khỏe không?",
        "Thời tiết đẹp vào buổi sáng này.",
        "Tôi muốn học ngôn ngữ tiếng Việt.",
        "Dịch máy đã cải thiện đáng kể trong những năm gần đây.",
        "Trí tuệ nhân tạo đang thay đổi nhiều ngành công nghiệp.",
        "Cảm ơn bạn đã giúp đỡ và hỗ trợ.",
        "Lập trình đòi hỏi sự kiên nhẫn và thực hành.",
        "Hoàng hôn trên núi thật ngoạn mục.",
        "Giáo dục là chìa khóa để phát triển cá nhân.",
        "Công nghệ kết nối mọi người trên khắp thế giới."
    ]
    
    # Write source file
    with open("sample_source.en", "w", encoding="utf-8") as f:
        for sentence in sample_en:
            f.write(f"{sentence}\n")
    
    # Write translation file
    with open("sample_translation.vi", "w", encoding="utf-8") as f:
        for sentence in sample_vi:
            f.write(f"{sentence}\n")
    
    print(f"   ✓ Created sample_source.en ({len(sample_en)} sentences)")
    print(f"   ✓ Created sample_translation.vi ({len(sample_vi)} translations)")
    
    return "sample_source.en", "sample_translation.vi"

def demo_source_only():
    """Demo 1: Source file only (fresh translation)"""
    print("\n" + "="*60)
    print("📋 DEMO 1: Source File Only (Fresh Translation)")
    print("="*60)
    
    source_file, _ = create_sample_files()
    
    print("\n🚀 Running fresh translation with source file only...")
    print(f"   Input: {source_file}")
    print(f"   Output: demo1_fresh_translation.vi")
    
    try:
        result = run_multi_gpu_llm_refine_with_files(
            source_file=source_file,
            output_file="demo1_fresh_translation.vi",
            translation_file=None,  # No existing translation
            num_gpus=2,  # Use fewer GPUs for demo
            model="llama3.1:8b-instruct-fp16",
            max_iterations=3  # Fewer iterations for demo
        )
        
        print("\n✅ Demo 1 completed successfully!")
        print("   Check demo1_fresh_translation.vi for results")
        
    except Exception as e:
        print(f"❌ Demo 1 failed: {e}")

def demo_source_and_translation():
    """Demo 2: Source + existing translation (refinement)"""
    print("\n" + "="*60)
    print("📋 DEMO 2: Source + Existing Translation (Refinement)")
    print("="*60)
    
    source_file, translation_file = create_sample_files()
    
    print("\n🚀 Running translation refinement with existing translations...")
    print(f"   Source: {source_file}")
    print(f"   Existing translations: {translation_file}")
    print(f"   Output: demo2_refined_translation.vi")
    
    try:
        result = run_multi_gpu_llm_refine_with_files(
            source_file=source_file,
            output_file="demo2_refined_translation.vi",
            translation_file=translation_file,  # Provide existing translation
            num_gpus=2,  # Use fewer GPUs for demo
            model="llama3.1:8b-instruct-fp16",
            max_iterations=3  # Fewer iterations for demo
        )
        
        print("\n✅ Demo 2 completed successfully!")
        print("   Check demo2_refined_translation.vi for results")
        print("   Original translations saved as demo2_refined_translation.vi.original")
        
    except Exception as e:
        print(f"❌ Demo 2 failed: {e}")

def demo_command_line():
    """Demo 3: Command line usage examples"""
    print("\n" + "="*60)
    print("📋 DEMO 3: Command Line Usage Examples")
    print("="*60)
    
    print("\n💡 Command line usage examples:")
    print("\n1. Fresh translation (source only):")
    print("   python file_input_refine.py source.en output.vi")
    
    print("\n2. Refinement (source + existing translation):")
    print("   python file_input_refine.py source.en output.vi --translation-file existing.vi")
    
    print("\n3. Custom settings:")
    print("   python file_input_refine.py source.en output.vi \\")
    print("     --translation-file existing.vi \\")
    print("     --num-gpus 4 \\")
    print("     --model llama3.1:8b-instruct-fp16 \\")
    print("     --max-iterations 6")
    
    print("\n4. Help:")
    print("   python file_input_refine.py --help")

def show_file_formats():
    """Show expected file formats"""
    print("\n" + "="*60)
    print("📋 File Format Requirements")
    print("="*60)
    
    print("\n📄 Source File Format (e.g., data.en):")
    print("   - One sentence per line")
    print("   - UTF-8 encoding")
    print("   - Plain text")
    print("\n   Example:")
    print("   Hello, how are you?")
    print("   The weather is nice today.")
    print("   Thank you for your help.")
    
    print("\n📄 Translation File Format (e.g., data.vi):")
    print("   - One translation per line")
    print("   - Must match source file line count")
    print("   - UTF-8 encoding")
    print("   - Can have empty lines (will be treated as missing translations)")
    print("\n   Example:")
    print("   Xin chào, bạn khỏe không?")
    print("   Hôm nay thời tiết đẹp.")
    print("   Cảm ơn bạn đã giúp đỡ.")
    
    print("\n📄 Output File Format:")
    print("   - One refined translation per line")
    print("   - UTF-8 encoding")
    print("   - Same line count as input")

def compare_results():
    """Compare results between different modes"""
    print("\n" + "="*60)
    print("📋 Comparing Results")
    print("="*60)
    
    files_to_check = [
        ("demo1_fresh_translation.vi", "Fresh translations"),
        ("demo2_refined_translation.vi", "Refined translations"),
        ("demo2_refined_translation.vi.original", "Original translations")
    ]
    
    print("\n📊 Results comparison:")
    for filename, description in files_to_check:
        if os.path.exists(filename):
            with open(filename, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            print(f"\n{description} ({filename}):")
            for i, line in enumerate(lines[:3], 1):  # Show first 3 lines
                print(f"   {i:2d}. {line.strip()}")
            if len(lines) > 3:
                print(f"   ... and {len(lines) - 3} more lines")
        else:
            print(f"\n{description}: File not found ({filename})")

def cleanup_demo_files():
    """Clean up demo files"""
    demo_files = [
        "sample_source.en",
        "sample_translation.vi",
        "demo1_fresh_translation.vi",
        "demo2_refined_translation.vi",
        "demo2_refined_translation.vi.original"
    ]
    
    print("\n🧹 Cleaning up demo files...")
    for filename in demo_files:
        if os.path.exists(filename):
            os.remove(filename)
            print(f"   ✓ Removed {filename}")

def main():
    """Main demo function"""
    print("🎯 Multi-GPU LLM Refine System - File Input Demo")
    print("This demo shows how to use the system with file inputs")
    
    # Check if we're in demo mode or running actual refinement
    if len(sys.argv) > 1 and sys.argv[1] != "--demo":
        print("\n💡 Use --demo flag to run demonstration, or see help:")
        print("   python demo_file_input.py --demo")
        print("   python file_input_refine.py --help")
        return
    
    try:
        # Show file format requirements
        show_file_formats()
        
        # Create sample files
        create_sample_files()
        
        # Demo 1: Source only
        demo_source_only()
        
        # Demo 2: Source + existing translation
        demo_source_and_translation()
        
        # Demo 3: Command line examples
        demo_command_line()
        
        # Compare results
        compare_results()
        
        print("\n🎉 All demos completed!")
        print("\nYou can now use the system with your own files:")
        print("  python file_input_refine.py your_source.en your_output.vi")
        print("  python file_input_refine.py your_source.en your_output.vi --translation-file existing.vi")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Ask if user wants to keep demo files
        try:
            keep_files = input("\n🗑️ Keep demo files? (y/N): ").lower().startswith('y')
            if not keep_files:
                cleanup_demo_files()
        except:
            cleanup_demo_files()

if __name__ == "__main__":
    main()
