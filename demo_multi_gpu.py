"""
Demo script for Multi-GPU LLM Refine System
Shows how to use the optimized multi-GPU approach with robust error handling
"""

from optimized_hybrid_refine import run_multi_gpu_llm_refine

def create_sample_data():
    """Create sample English sentences for testing"""
    sample_sentences = [
        "Hello, how are you today?",
        "The weather is beautiful this morning.",
        "I would like to order a cup of coffee, please.",
        "Technology is advancing rapidly in recent years.",
        "Learning a new language requires dedication and practice.",
        "The conference will be held next week in the city center.",
        "She enjoys reading books in her free time.",
        "The project deadline is approaching quickly.",
        "We need to discuss the budget for next quarter.",
        "The team worked together to achieve their goals."
    ]
    
    with open("sample_data.en", "w", encoding="utf-8") as f:
        for sentence in sample_sentences:
            f.write(f"{sentence}\n")
    
    print("✅ Created sample_data.en with 10 test sentences")


def demo_multi_gpu_processing():
    """Demonstrate multi-GPU processing with different configurations"""
    
    print("🚀 Multi-GPU LLM Refine Demo")
    print("=" * 50)
    
    # Create sample data
    create_sample_data()
    
    # Configuration options
    configs = [
        {
            "name": "2 GPUs - Fast Processing",
            "num_gpus": 2,
            "max_iterations": 3,
            "description": "Quick processing with moderate refinement"
        },
        {
            "name": "4 GPUs - Balanced Processing", 
            "num_gpus": 4,
            "max_iterations": 6,
            "description": "Balanced speed and quality with full refinement"
        },
        {
            "name": "Single GPU - Fallback Mode",
            "num_gpus": 1, 
            "max_iterations": 4,
            "description": "Single GPU fallback with robust error handling"
        }
    ]
    
    print("\nAvailable configurations:")
    for i, config in enumerate(configs, 1):
        print(f"{i}. {config['name']}")
        print(f"   {config['description']}")
        print(f"   GPUs: {config['num_gpus']}, Iterations: {config['max_iterations']}")
    
    # Let user choose or default to balanced
    choice = input("\nChoose configuration (1-3) or press Enter for default [2]: ").strip()
    
    if choice == "1":
        selected_config = configs[0]
    elif choice == "3":
        selected_config = configs[2]
    else:
        selected_config = configs[1]  # Default
    
    print(f"\n🎯 Running: {selected_config['name']}")
    print(f"📝 Processing sample_data.en with {selected_config['num_gpus']} GPUs")
    
    try:
        # Run the multi-GPU processing
        final_state = run_multi_gpu_llm_refine(
            source_file="sample_data.en",
            output_file="sample_output.vi",
            num_gpus=selected_config['num_gpus'],
            model="llama3.1:8b-instruct-fp16",
            max_iterations=selected_config['max_iterations'],
            temperature=41.67,
            cooling_rate=0.4
        )
        
        print("\n✅ Processing completed successfully!")
        print("📄 Results saved to sample_output.vi")
        
        # Show sample results
        print("\n📋 Sample Results:")
        print("-" * 30)
        
        with open("sample_data.en", "r", encoding="utf-8") as src_file:
            src_lines = src_file.readlines()
        
        with open("sample_output.vi", "r", encoding="utf-8") as out_file:
            out_lines = out_file.readlines()
        
        # Show first 3 translations
        for i in range(min(3, len(src_lines))):
            print(f"EN: {src_lines[i].strip()}")
            print(f"VI: {out_lines[i].strip()}")
            print()
        
        if len(src_lines) > 3:
            print(f"... and {len(src_lines) - 3} more translations")
    
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        print("\n💡 Troubleshooting tips:")
        print("1. Ensure Ollama is installed and running")
        print("2. Check if the specified model is available")
        print("3. Verify GPU drivers and CUDA installation")
        print("4. Try reducing num_gpus if having GPU issues")


def show_system_requirements():
    """Show system requirements and setup instructions"""
    print("\n📋 System Requirements:")
    print("=" * 50)
    print("• Python 3.8+")
    print("• NVIDIA GPUs with CUDA support")
    print("• Ollama installed and configured")
    print("• Required Python packages:")
    print("  - langchain")
    print("  - langchain-ollama")
    print("  - langgraph")
    print("  - pydantic")
    print("  - tqdm")
    
    print("\n🔧 Setup Instructions:")
    print("1. Install Ollama: https://ollama.ai/")
    print("2. Pull model: ollama pull llama3.1:8b-instruct-fp16")
    print("3. Install Python packages: pip install langchain langchain-ollama langgraph pydantic tqdm")
    print("4. Ensure CUDA is properly configured")
    
    print("\n💡 Usage Tips:")
    print("• Start with fewer GPUs (1-2) to test setup")
    print("• Monitor GPU memory usage during processing")
    print("• Use smaller batch sizes if encountering memory issues")
    print("• Check Ollama logs if models fail to load")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--requirements":
        show_system_requirements()
    else:
        demo_multi_gpu_processing()
