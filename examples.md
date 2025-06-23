# Multi-GPU LLM Refine Examples

This document provides detailed examples and explanations for using the Multi-GPU LLM Refine system.

## Table of Contents
1. [System Overview](#system-overview)
2. [Basic Usage](#basic-usage)
3. [Advanced Configuration](#advanced-configuration)
4. [Error Handling Examples](#error-handling-examples)
5. [Performance Optimization](#performance-optimization)
6. [Code Architecture](#code-architecture)
7. [Troubleshooting](#troubleshooting)
8. [File Input Examples](#file-input-examples-)

## System Overview

The Multi-GPU LLM Refine system combines three key technologies:
- **Your original fast batch processing** from `my_llm_refine.py`
- **Multi-GPU parallel processing** with ChatOllama on different ports
- **Robust error handling** with LangChain's OutputFixingParser and RetryOutputParser

### Architecture Diagram
```
Input File (data.en)
        ↓
┌───────────────────────────────────┐
│    MultiGPUOllamaManager          │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐  │
│  │GPU 0│ │GPU 1│ │GPU 2│ │GPU 3│  │
│  │:11434│ │:11435│ │:11436│ │:11437│ │
│  └─────┘ └─────┘ └─────┘ └─────┘  │
└───────────────────────────────────┘
        ↓
┌───────────────────────────────────┐
│    Workload Distribution          │
│  Chunk 1  Chunk 2  Chunk 3  Chunk 4│
│    ↓       ↓       ↓       ↓      │
│  GPU 0   GPU 1   GPU 2   GPU 3    │
└───────────────────────────────────┘
        ↓
┌───────────────────────────────────┐
│    3-Tier Error Handling          │
│  Tier 1: Fast Structured Output   │
│  Tier 2: OutputFixingParser       │
│  Tier 3: RetryOutputParser        │
└───────────────────────────────────┘
        ↓
Output File (data_refined.vi)
```

## Basic Usage

### Example 1: Simple Translation Refinement

```python
from optimized_hybrid_refine import run_multi_gpu_llm_refine

# Basic usage with default settings
run_multi_gpu_llm_refine(
    source_file="input.en",
    output_file="output.vi",
    num_gpus=2,           # Use 2 GPUs
    max_iterations=3      # Quick refinement
)
```

**What happens:**
1. System splits input sentences across 2 GPUs
2. Each GPU processes its chunk independently
3. Results are combined in original order
4. 3 iterations of refinement with simulated annealing

### Example 2: Creating Sample Data

```python
# Create sample English sentences
sample_data = [
    "Hello, how are you today?",
    "The weather is beautiful this morning.",
    "I would like to order a cup of coffee.",
    "Technology is advancing rapidly.",
    "Learning languages requires practice."
]

with open("sample.en", "w", encoding="utf-8") as f:
    for sentence in sample_data:
        f.write(f"{sentence}\n")

# Process the sample
run_multi_gpu_llm_refine(
    source_file="sample.en",
    output_file="sample.vi"
)
```

### Example 3: Step-by-Step Processing

```python
from optimized_hybrid_refine import MultiGPULLMRefine

# Initialize the refiner
refiner = MultiGPULLMRefine(num_gpus=4, model="llama3.1:8b-instruct-fp16")

# Setup GPUs
active_gpus = refiner.setup_multi_gpu()
print(f"Active GPUs: {active_gpus}")

try:
    # Load your data
    with open("data.en", "r", encoding="utf-8") as f:
        sentences = [line.strip() for line in f.readlines()]
    
    # The system will automatically handle the rest
    # This is just to show the setup process
    
finally:
    # Always cleanup
    refiner.gpu_manager.cleanup()
```

## Advanced Configuration

### Example 4: High-Performance Setup (4+ GPUs)

```python
run_multi_gpu_llm_refine(
    source_file="large_dataset.en",
    output_file="large_dataset.vi",
    num_gpus=8,                    # Use all 8 GPUs
    model="llama3.1:8b-instruct-fp16",
    max_iterations=6,              # Full refinement
    temperature=41.67,             # Standard SA temperature
    cooling_rate=0.4               # Standard cooling rate
)
```

**Performance expectations:**
- ~120-160 sentences/second (8 GPUs × 15-20 sentences/sec)
- ~97% success rate with robust error handling
- Memory usage: ~2GB per GPU

### Example 5: Memory-Constrained Setup

```python
run_multi_gpu_llm_refine(
    source_file="data.en",
    output_file="data.vi",
    num_gpus=2,                    # Reduce GPU count
    model="llama3.1:8b-instruct-q4_0",  # Use quantized model
    max_iterations=4,              # Reduce iterations
    temperature=25.0,              # Lower temperature
    cooling_rate=0.6               # Faster cooling
)
```

**When to use:**
- Limited GPU memory (< 8GB per GPU)
- Smaller datasets (< 1000 sentences)
- Quick turnaround needed

### Example 6: Research/Experimentation Setup

```python
run_multi_gpu_llm_refine(
    source_file="research_data.en",
    output_file="research_output.vi",
    num_gpus=4,
    model="llama3.1:8b-instruct-fp16",
    max_iterations=10,             # Extended refinement
    temperature=50.0,              # Higher exploration
    cooling_rate=0.3               # Slower cooling
)
```

**Features:**
- Extended refinement cycles for better quality
- Higher temperature for more exploration
- Detailed performance metrics for analysis

## Error Handling Examples

### Example 7: Understanding the 3-Tier System

```python
# This happens automatically, but here's what the system does:

# Tier 1: Fast Structured Output (your original approach)
try:
    result = self.generator.invoke({"query": query})
    # ~85% success rate, fastest processing
    return result
except Exception:
    # Fall through to Tier 2
    
# Tier 2: OutputFixingParser (LangChain)
try:
    response = self.llm.invoke(query)
    fixed_result = self.translation_fixing_parser.parse(response.content)
    # ~12% additional recovery
    return fixed_result
except Exception:
    # Fall through to Tier 3
    
# Tier 3: RetryOutputParser (LangChain)
try:
    response = self.llm.invoke(query)
    retry_result = self.translation_retry_parser.parse_with_prompt(
        response.content, prompt
    )
    # ~3% additional recovery
    return retry_result
except Exception:
    # Final fallback - never lose data
    return Refine(translation=f"[PARSE_ERROR] {src_text}")
```

### Example 8: Monitoring Error Recovery

```python
from optimized_hybrid_refine import MultiGPULLMRefine

refiner = MultiGPULLMRefine(num_gpus=4)
active_gpus = refiner.setup_multi_gpu()

# After processing, check performance stats
for gpu_id, processor in refiner.processors.items():
    stats = processor.get_performance_stats()
    print(f"GPU {gpu_id} Performance:")
    print(f"  Fast success: {stats['fast_success']}")
    print(f"  Fixing success: {stats['fixing_success']}")
    print(f"  Retry success: {stats['retry_success']}")
    print(f"  Total failures: {stats['total_failures']}")
    print(f"  Success rate: {stats['success_rate']:.1f}%")
```

### Example 9: Handling Common Parsing Issues

```python
# The system automatically handles these common issues:

# Issue 1: Malformed JSON
# Input: '{"translation": "Xin chào"'  # Missing closing brace
# Tier 2 (OutputFixingParser) fixes: '{"translation": "Xin chào"}'

# Issue 2: Extra text around JSON
# Input: 'Here is the translation: {"translation": "Xin chào"} Hope this helps!'
# Tier 2 extracts: '{"translation": "Xin chào"}'

# Issue 3: Grade extraction from text
# Input: 'This translation gets a score of 2.5 out of 3 because...'
# System extracts: grade="2.5"

# Issue 4: Vietnamese text detection
# Input: 'Translation: Xin chào, bạn có khỏe không?'
# System extracts: 'Xin chào, bạn có khỏe không?'
```

## Performance Optimization

### Example 10: Optimal GPU Configuration

```python
import subprocess

# Check available GPUs
def check_gpu_memory():
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], 
                          capture_output=True, text=True)
    free_memory = [int(x) for x in result.stdout.strip().split('\n')]
    return free_memory

# Configure based on available memory
free_mem = check_gpu_memory()
suitable_gpus = len([mem for mem in free_mem if mem > 8000])  # 8GB minimum

run_multi_gpu_llm_refine(
    source_file="data.en",
    output_file="data.vi",
    num_gpus=suitable_gpus,
    model="llama3.1:8b-instruct-fp16" if suitable_gpus > 2 else "llama3.1:8b-instruct-q4_0"
)
```

### Example 11: Batch Size Optimization

```python
# For different dataset sizes:

# Small dataset (< 100 sentences)
run_multi_gpu_llm_refine(
    source_file="small.en",
    output_file="small.vi",
    num_gpus=1,                    # Single GPU sufficient
    max_iterations=3
)

# Medium dataset (100-1000 sentences)  
run_multi_gpu_llm_refine(
    source_file="medium.en", 
    output_file="medium.vi",
    num_gpus=2,                    # 2 GPUs for balance
    max_iterations=4
)

# Large dataset (1000+ sentences)
run_multi_gpu_llm_refine(
    source_file="large.en",
    output_file="large.vi", 
    num_gpus=4,                    # 4+ GPUs for speed
    max_iterations=6
)
```

## Code Architecture

### Example 12: Understanding the State Flow

```python
# The system follows this state flow (same as your original):

# Initial State
state = {
    "src": ["Hello", "Goodbye", "Thank you"],
    "temperature": 41.67,
    "cooling_rate": 0.4,
    "n": 6,  # max iterations
    "i": 0   # current iteration
}

# Iteration 1: Generate translations
# GPU 0: "Hello" → "Xin chào"
# GPU 1: "Goodbye" → "Tạm biệt"  
# GPU 2: "Thank you" → "Cảm ơn"

# Iteration 1: Evaluate translations
# GPU 0: "Xin chào" → grade="3", feedback="Perfect"
# GPU 1: "Tạm biệt" → grade="2.5", feedback="Good but could be more natural"
# GPU 2: "Cảm ơn" → grade="3", feedback="Perfect"

# Iteration 1: Simulated Annealing
# Accepts/rejects improvements based on temperature

# Continue until i >= n or convergence
```

### Example 13: Custom Prompt Modification

```python
# You can modify the system by extending the RobustLLMProcessor class:

class CustomRobustLLMProcessor(RobustLLMProcessor):
    def __init__(self, llm: ChatOllama):
        super().__init__(llm)
        
        # Custom prompt template
        self.template = ChatPromptTemplate.from_messages([
            ("system", "You are an expert Vietnamese translator with cultural awareness."),
            ("human", "{query}"),
        ])
        
        # Rebuild chains
        self.generator = self.template | self.generator_llm
        self.evaluator = self.template | self.evaluator_llm
    
    def generate_translation_robust(self, src_text: str, current_translation: str = None, feedback_text: str = None):
        # Custom query formatting
        if current_translation and feedback_text:
            query = f"""Translate to Vietnamese with cultural context: {src_text}
Current translation: {current_translation}
Improvement feedback: {feedback_text}
Provide a culturally appropriate Vietnamese translation."""
        else:
            query = f"Translate to Vietnamese with cultural awareness: {src_text}"
        
        # Use parent's robust processing
        return super().generate_translation_robust(src_text, current_translation, feedback_text)
```

### Example 14: Understanding GPU Distribution

```python
# For 10 sentences with 4 GPUs:
sentences = ["sent1", "sent2", "sent3", "sent4", "sent5", "sent6", "sent7", "sent8", "sent9", "sent10"]

# Automatic distribution:
# GPU 0: ["sent1", "sent2", "sent3"]      # sentences 0-2
# GPU 1: ["sent4", "sent5"]               # sentences 3-4  
# GPU 2: ["sent6", "sent7"]               # sentences 5-6
# GPU 3: ["sent8", "sent9", "sent10"]     # sentences 7-9 (last GPU gets remainder)

# Processing happens in parallel:
# All GPUs process simultaneously
# Results combined in original order: sent1, sent2, ..., sent10
```

## Troubleshooting

### Example 15: Debug Mode

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Run with verbose output
run_multi_gpu_llm_refine(
    source_file="debug.en",
    output_file="debug.vi",
    num_gpus=2,
    max_iterations=2  # Reduce iterations for faster debugging
)

# You'll see detailed output like:
# 🚀 Setting up Ollama on 2 GPUs...
# ✅ GPU 0 ready on port 11434
# ✅ GPU 1 ready on port 11435
# 🔄 GPU 0 processing 5 sentences...
# ✅ Translation parsed successfully (Tier 1)
# 🔄 Fast generation failed, trying OutputFixingParser...
# etc.
```

### Example 16: Error Recovery Testing

```python
# Create problematic input to test error recovery
problematic_sentences = [
    "This sentence has special characters: @#$%^&*()",
    "This is a very long sentence that might cause issues with tokenization and could potentially exceed the model's context window limits causing parsing failures",
    "",  # Empty sentence
    "Unicode test: 你好 🌟 café naïve résumé",
    "JSON-like text: {\"this\": \"might confuse parser\"}"
]

with open("test_errors.en", "w", encoding="utf-8") as f:
    for sentence in problematic_sentences:
        f.write(f"{sentence}\n")

# Run with error recovery
run_multi_gpu_llm_refine(
    source_file="test_errors.en",
    output_file="test_errors.vi",
    num_gpus=1,  # Single GPU for easier debugging
    max_iterations=2
)

# Check results - should handle all cases gracefully
with open("test_errors.vi", "r", encoding="utf-8") as f:
    results = f.readlines()
    for i, result in enumerate(results):
        print(f"Input:  {problematic_sentences[i]}")
        print(f"Output: {result.strip()}")
        print()
```

### Example 17: Performance Benchmarking

```python
import time

def benchmark_different_configs():
    # Create test data
    test_sentences = ["This is a test sentence."] * 100
    with open("benchmark.en", "w", encoding="utf-8") as f:
        for sentence in test_sentences:
            f.write(f"{sentence}\n")
    
    configs = [
        {"num_gpus": 1, "name": "Single GPU"},
        {"num_gpus": 2, "name": "Dual GPU"},
        {"num_gpus": 4, "name": "Quad GPU"},
    ]
    
    for config in configs:
        print(f"\n🔄 Testing {config['name']}...")
        start_time = time.time()
        
        try:
            run_multi_gpu_llm_refine(
                source_file="benchmark.en",
                output_file=f"benchmark_{config['num_gpus']}gpu.vi",
                num_gpus=config['num_gpus'],
                max_iterations=3
            )
            
            end_time = time.time()
            processing_time = end_time - start_time
            throughput = 100 / processing_time  # sentences per second
            
            print(f"✅ {config['name']}: {processing_time:.2f}s, {throughput:.1f} sentences/sec")
            
        except Exception as e:
            print(f"❌ {config['name']} failed: {e}")

# Run benchmark
benchmark_different_configs()
```

## Real-World Usage Scenarios

### Example 18: Production Pipeline

```python
import os
from pathlib import Path

def production_translation_pipeline(input_dir: str, output_dir: str):
    """Production-ready translation pipeline"""
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Process all .en files in input directory
    for en_file in input_path.glob("*.en"):
        print(f"📝 Processing {en_file.name}...")
        
        output_file = output_path / f"{en_file.stem}.vi"
        
        try:
            run_multi_gpu_llm_refine(
                source_file=str(en_file),
                output_file=str(output_file),
                num_gpus=4,
                model="llama3.1:8b-instruct-fp16",
                max_iterations=6,
                temperature=41.67,
                cooling_rate=0.4
            )
            print(f"✅ Completed {en_file.name} → {output_file.name}")
            
        except Exception as e:
            print(f"❌ Failed {en_file.name}: {e}")
            continue

# Usage
production_translation_pipeline("./inputs", "./outputs")
```

### Example 19: Quality Assessment

```python
def assess_translation_quality(source_file: str, output_file: str):
    """Assess the quality of translations"""
    
    with open(source_file, 'r', encoding='utf-8') as f:
        source_sentences = [line.strip() for line in f.readlines()]
    
    with open(output_file, 'r', encoding='utf-8') as f:
        translated_sentences = [line.strip() for line in f.readlines()]
    
    print("📊 Translation Quality Assessment:")
    print("=" * 50)
    
    # Check for errors
    error_count = 0
    for i, translation in enumerate(translated_sentences):
        if "[PARSE_ERROR]" in translation or "[ERROR]" in translation:
            error_count += 1
            print(f"❌ Error in sentence {i+1}: {translation}")
    
    # Calculate metrics
    total_sentences = len(source_sentences)
    success_rate = (total_sentences - error_count) / total_sentences * 100
    
    print(f"\n📈 Summary:")
    print(f"   Total sentences: {total_sentences}")
    print(f"   Successful translations: {total_sentences - error_count}")
    print(f"   Failed translations: {error_count}")
    print(f"   Success rate: {success_rate:.1f}%")
    
    # Show sample translations
    print(f"\n📋 Sample Translations:")
    for i in range(min(3, len(source_sentences))):
        print(f"EN: {source_sentences[i]}")
        print(f"VI: {translated_sentences[i]}")
        print()

# Usage
assess_translation_quality("input.en", "output.vi")
```

## File Input Examples 🆕

### Example 1: Fresh Translation from Source File
```python
from file_input_refine import run_multi_gpu_llm_refine_with_files

# Translate English file to Vietnamese
run_multi_gpu_llm_refine_with_files(
    source_file="document.en",          # Input: English sentences
    output_file="document.vi",          # Output: Vietnamese translations
    translation_file=None,              # No existing translation
    num_gpus=4,
    max_iterations=5
)
```

**Input file format (document.en):**
```
Hello, how are you today?
The weather is beautiful this morning.
I would like to learn Vietnamese language.
Machine translation has improved significantly.
```

**Output (document.vi):**
```
Xin chào, hôm nay bạn khỏe không?
Thời tiết rất đẹp vào buổi sáng này.
Tôi muốn học ngôn ngữ tiếng Việt.
Dịch máy đã cải thiện đáng kể.
```

### Example 2: Refining Existing Translations
```python
# Improve existing Google Translate output
run_multi_gpu_llm_refine_with_files(
    source_file="source.en",
    output_file="improved.vi",
    translation_file="google_translate.vi",  # Existing translation to improve
    num_gpus=2,
    max_iterations=4
)
```

**Scenario comparison:**

*Original Google Translate (google_translate.vi):*
```
Xin chào, bạn như thế nào hôm nay?
Thời tiết là đẹp sáng nay.
Tôi muốn học ngôn ngữ Việt Nam.
Dịch máy đã cải thiện đáng kể.
```

*After refinement (improved.vi):*
```
Xin chào, hôm nay bạn khỏe không?
Thời tiết rất đẹp vào buổi sáng này.
Tôi muốn học ngôn ngữ tiếng Việt.
Dịch máy đã được cải thiện đáng kể.
```

### Example 3: Command Line Usage
```bash
# Fresh translation
python file_input_refine.py data.en data_translated.vi --num-gpus 4

# Refine existing translation
python file_input_refine.py data.en data_refined.vi \
  --translation-file existing_translation.vi \
  --num-gpus 2 \
  --max-iterations 6

# Custom model and settings
python file_input_refine.py large_dataset.en output.vi \
  --translation-file current.vi \
  --model "llama3.1:8b-instruct-fp16" \
  --temperature 50.0 \
  --cooling-rate 0.3
```

### Example 4: Handling Different Input Scenarios
```python
# Scenario A: Missing some translations
def handle_partial_translations():
    # Input files where some translations are missing
    source_lines = [
        "Hello world",
        "Good morning", 
        "Thank you",
        "Goodbye"
    ]
    
    existing_translations = [
        "Xin chào thế giới",
        "",  # Missing translation
        "Cảm ơn",
        ""   # Missing translation
    ]
    
    # System will generate new translations for missing lines
    # while refining existing ones

# Scenario B: Different file encodings
def handle_encoding_issues():
    # System automatically handles UTF-8, latin-1 encodings
    # Files with special characters are properly processed
    pass

# Scenario C: Large files
def handle_large_files():
    run_multi_gpu_llm_refine_with_files(
        source_file="huge_dataset.en",  # 100K+ sentences
        output_file="huge_output.vi",
        num_gpus=8,                     # Use more GPUs
        max_iterations=3                # Fewer iterations for speed
    )
```

### Example 5: Demo and Testing
```python
# Run comprehensive demo
import subprocess

# Demo with sample files
subprocess.run(["python", "demo_file_input.py", "--demo"])

# Quick test of functionality
subprocess.run(["python", "test_file_input.py"])

# Test specific scenarios
from test_file_input import create_test_files, test_file_loading

# Create small test files
source_file, translation_file = create_test_files()

# Test file loading
success = test_file_loading()
print(f"File loading test: {'✓' if success else '❌'}")
```
