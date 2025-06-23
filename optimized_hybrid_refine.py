"""
Multi-GPU Optimized LLM Refine System
Combines the speed of my_llm_refine.py with robust error handling and multi-GPU processing

Features:
- Multi-GPU parallel processing with ChatOllama on different ports
- OutputFixingParser and RetryOutputParser for robust error handling
- Fast batch processing based on original my_llm_refine.py architecture
- Intelligent workload distribution across GPUs
- Comprehensive error recovery and performance monitoring
"""

from typing_extensions import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field, ValidationError
from langchain_ollama import ChatOllama
from langchain.output_parsers import OutputFixingParser, RetryOutputParser
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.exceptions import OutputParserException
from langchain.prompts import ChatPromptTemplate
from tqdm import tqdm
import random
import math
import json
import re
import time
import subprocess
import os
import hashlib
from typing import List, Dict, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor


# Enhanced State models for multi-GPU processing
class MultiGPUState(TypedDict):
    src: List[str]
    tgt: List[str]
    feedback: List[str]
    score: List[str]
    temperature: float
    cooling_rate: float
    n: int
    i: int
    
    # Multi-GPU tracking
    gpu_assignments: List[int]
    chunk_results: List[Dict]
    processing_times: List[float]

class HiddenState(TypedDict):
    h_tgt: List[str]
    h_feedback: List[str]
    h_score: List[str]
    
    # Performance metrics
    api_calls: int
    parse_successes: int
    parse_failures: int
    retry_attempts: int

class OverallState(MultiGPUState, HiddenState):
    pass


# Enhanced Pydantic models with robust validation
class Feedback(BaseModel):
    """Enhanced feedback model with robust validation and fallbacks"""
    grade: Literal["1", "1.5", "2", "2.5", "3"] = Field(
        default="2", 
        description="Translation quality score: 3=Perfect, 2.5=Minor errors, 2=Some errors, 1.5=Significant errors, 1=Poor"
    )
    feedback: str = Field(
        default="Could not evaluate", 
        description="Detailed feedback explaining the grade",
        min_length=5,
        max_length=500
    )
    
    def __init__(self, **data):
        # Smart grade extraction and validation
        if 'grade' in data:
            grade_str = str(data['grade'])
            # Extract valid grade from various formats
            grade_match = re.search(r'([123](?:\.5)?)', grade_str)
            if grade_match:
                data['grade'] = grade_match.group(1)
            elif any(keyword in grade_str.lower() for keyword in ['perfect', 'excellent', 'hoàn hảo']):
                data['grade'] = "3"
            elif any(keyword in grade_str.lower() for keyword in ['good', 'tốt']):
                data['grade'] = "2.5"
            elif any(keyword in grade_str.lower() for keyword in ['poor', 'bad', 'kém']):
                data['grade'] = "1"
            else:
                data['grade'] = "2"  # Safe default
        
        # Clean and validate feedback text
        if 'feedback' in data and data['feedback']:
            feedback = str(data['feedback']).strip()
            # Remove common parsing artifacts
            feedback = re.sub(r'^(Feedback|Grade|Score|Điểm|Phản hồi):\s*', '', feedback, flags=re.IGNORECASE)
            feedback = re.sub(r'["\'\`]{1,3}', '', feedback)
            data['feedback'] = feedback[:500] if len(feedback) > 500 else feedback
        
        super().__init__(**data)


class Refine(BaseModel):
    """Enhanced translation model with cleanup and validation"""
    translation: str = Field(
        default="", 
        description="High-quality Vietnamese translation",
        min_length=1,
        max_length=1000
    )
    
    def __init__(self, **data):
        # Clean translation text
        if 'translation' in data and data['translation']:
            translation = str(data['translation']).strip()
            # Remove common parsing artifacts
            translation = re.sub(r'^["\'\`]{1,3}|["\'\`]{1,3}$', '', translation)
            translation = re.sub(r'^(Translation|Dịch|Bản dịch):\s*', '', translation, flags=re.IGNORECASE)
            translation = re.sub(r'\n+', ' ', translation)  # Replace newlines
            translation = re.sub(r'\s+', ' ', translation)  # Normalize spaces
            data['translation'] = translation
        
        super().__init__(**data)


class MultiGPUOllamaManager:
    """Enhanced multi-GPU manager with automatic port management and cleanup"""
    
    def __init__(self, num_gpus: int = 4, base_port: int = 11434, model: str = "llama3.1:8b-instruct-fp16"):
        self.num_gpus = num_gpus
        self.base_port = base_port
        self.model = model
        self.ollama_processes = []
        self.llm_instances = []
        self.active_ports = []
        
    def setup_multi_gpu_ollama(self):
        """Setup Ollama instances on multiple GPUs with better error handling"""
        print(f"🚀 Setting up Ollama on {self.num_gpus} GPUs...")
        
        for gpu_id in range(self.num_gpus):
            port = self.base_port + gpu_id
            
            try:
                # Kill existing processes on this port
                subprocess.run(f"taskkill /F /FI \"COMMANDLINE like *{port}*\" 2>nul", shell=True, check=False)
                time.sleep(2)
                
                # Setup environment for this GPU
                env = {
                    **os.environ,
                    'CUDA_VISIBLE_DEVICES': str(gpu_id),
                    'OLLAMA_HOST': f'0.0.0.0:{port}',
                    'OLLAMA_MODELS': os.environ.get('OLLAMA_MODELS', './models')
                }
                
                # Start Ollama server for this GPU
                print(f"   Starting Ollama on GPU {gpu_id}, port {port}...")
                process = subprocess.Popen(
                    ['ollama', 'serve'],
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
                )
                
                self.ollama_processes.append(process)
                time.sleep(8)  # Wait for startup
                
                # Test connection and create LLM instance
                llm = ChatOllama(
                    model=self.model,
                    base_url=f"http://localhost:{port}",
                    temperature=0.6,
                    top_p=0.9,
                    num_predict=128,
                    timeout=30
                )
                
                # Test with a simple call
                test_response = llm.invoke("Test connection")
                
                self.llm_instances.append(llm)
                self.active_ports.append(port)
                print(f"✅ GPU {gpu_id} ready on port {port}")
                
            except Exception as e:
                print(f"❌ Failed to setup GPU {gpu_id}: {e}")
                # Continue with other GPUs
        
        print(f"✅ {len(self.llm_instances)} GPUs successfully configured")
        return len(self.llm_instances)
    
    def get_llm(self, gpu_id: int) -> ChatOllama:
        """Get LLM instance for specific GPU with fallback"""
        if gpu_id < len(self.llm_instances):
            return self.llm_instances[gpu_id]
        # Fallback to round-robin if GPU ID out of range
        return self.llm_instances[gpu_id % len(self.llm_instances)] if self.llm_instances else None
    
    def get_available_gpus(self) -> int:
        """Get number of available GPUs"""
        return len(self.llm_instances)
    
    def cleanup(self):
        """Clean up all Ollama processes"""
        print("🧹 Cleaning up Ollama processes...")
        for process in self.ollama_processes:
            try:
                process.terminate()
                process.wait(timeout=5)
            except:
                process.kill()
        
        # Kill any remaining processes on our ports
        for port in self.active_ports:
            subprocess.run(f"taskkill /F /FI \"COMMANDLINE like *{port}*\" 2>nul", shell=True, check=False)


class RobustLLMProcessor:
    """
    Robust LLM processor with OutputFixingParser and RetryOutputParser integration
    """
    
    def __init__(self, llm: ChatOllama):
        self.llm = llm
        
        # Setup base parsers
        self.translation_parser = PydanticOutputParser(pydantic_object=Refine)
        self.feedback_parser = PydanticOutputParser(pydantic_object=Feedback)
        
        # Create robust parsers with LangChain's error handling
        self.translation_fixing_parser = OutputFixingParser.from_llm(
            parser=self.translation_parser, 
            llm=self.llm
        )
        self.feedback_fixing_parser = OutputFixingParser.from_llm(
            parser=self.feedback_parser, 
            llm=self.llm
        )
        
        # Create retry parsers for maximum robustness
        self.translation_retry_parser = RetryOutputParser.from_llm(
            parser=self.translation_parser, 
            llm=self.llm
        )
        self.feedback_retry_parser = RetryOutputParser.from_llm(
            parser=self.feedback_parser, 
            llm=self.llm
        )
        
        # Setup optimized prompts
        self.template = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant that translates English to Vietnamese."),
            ("human", "{query}"),
        ])
        
        # Create chains with structured output (fast path)
        self.generator_llm = self.llm.with_structured_output(Refine)
        self.evaluator_llm = self.llm.with_structured_output(Feedback)
        
        self.generator = self.template | self.generator_llm
        self.evaluator = self.template | self.evaluator_llm
        
        # Performance tracking
        self.stats = {
            'fast_success': 0,
            'fixing_success': 0,
            'retry_success': 0,
            'total_failures': 0
        }
    
    def generate_translation_robust(self, src_text: str, current_translation: str = None, feedback_text: str = None) -> Refine:
        """
        Generate translation with 3-tier robustness:
        1. Fast structured output (my_llm_refine.py approach)
        2. OutputFixingParser for malformed outputs
        3. RetryOutputParser for failed attempts
        """
        # Prepare query
        if current_translation and feedback_text:
            query = f"""Translate this from English to Vietnamese: {src_text}. Your current translation is: {current_translation}
Please improve your translation taking into account this feedback: {feedback_text}"""
        else:
            query = f"Translate this from English to Vietnamese: {src_text}"
        
        try:
            # Tier 1: Fast structured output (original approach)
            result = self.generator.invoke({"query": query})
            self.stats['fast_success'] += 1
            return result
            
        except Exception as e:
            print(f"🔄 Fast generation failed, trying OutputFixingParser...")
            
            try:
                # Tier 2: OutputFixingParser
                response = self.llm.invoke(self.template.format_messages(query=query))
                fixed_result = self.translation_fixing_parser.parse(response.content)
                self.stats['fixing_success'] += 1
                return fixed_result
                
            except Exception as e2:
                print(f"🔄 OutputFixingParser failed, trying RetryOutputParser...")
                
                try:
                    # Tier 3: RetryOutputParser (full retry with context)
                    response = self.llm.invoke(self.template.format_messages(query=query))
                    retry_result = self.translation_retry_parser.parse_with_prompt(
                        response.content, 
                        self.template.format_messages(query=query)
                    )
                    self.stats['retry_success'] += 1
                    return retry_result
                    
                except Exception as e3:
                    # Final fallback
                    print(f"❌ All parsing tiers failed: {e3}")
                    self.stats['total_failures'] += 1
                    return Refine(translation=f"[PARSE_ERROR] {src_text}")
    
    def evaluate_translation_robust(self, src_text: str, translation: str) -> Feedback:
        """
        Evaluate translation with 3-tier robustness
        """
        query = f"""You are a language expert. Rate this Vietnamese translation of the English text.

Rate from 1-3:
3: Perfect translation
2.5: Minor errors but understandable  
2: Some errors but mostly correct
1.5: Significant errors, partially understandable
1: Poor translation, hard to understand

English: {src_text}
Vietnamese: {translation}

Provide grade and detailed feedback."""
        
        try:
            # Tier 1: Fast structured output
            result = self.evaluator.invoke({"query": query})
            self.stats['fast_success'] += 1
            return result
            
        except Exception as e:
            print(f"🔄 Fast evaluation failed, trying OutputFixingParser...")
            
            try:
                # Tier 2: OutputFixingParser
                response = self.llm.invoke(self.template.format_messages(query=query))
                fixed_result = self.feedback_fixing_parser.parse(response.content)
                self.stats['fixing_success'] += 1
                return fixed_result
                
            except Exception as e2:
                print(f"🔄 OutputFixingParser failed, trying RetryOutputParser...")
                
                try:
                    # Tier 3: RetryOutputParser
                    response = self.llm.invoke(self.template.format_messages(query=query))
                    retry_result = self.feedback_retry_parser.parse_with_prompt(
                        response.content,
                        self.template.format_messages(query=query)
                    )
                    self.stats['retry_success'] += 1
                    return retry_result
                    
                except Exception as e3:
                    print(f"❌ All evaluation parsing failed: {e3}")
                    self.stats['total_failures'] += 1
                    return Feedback(grade="2", feedback=f"Evaluation failed: {str(e3)[:100]}")
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        total = sum(self.stats.values())
        return {
            **self.stats,
            'total_operations': total,
            'success_rate': (total - self.stats['total_failures']) / max(total, 1) * 100
        }


class MultiGPULLMRefine:
    """
    Main class combining multi-GPU processing with robust error handling
    """
    
    def __init__(self, num_gpus: int = 4, model: str = "llama3.1:8b-instruct-fp16"):
        self.gpu_manager = MultiGPUOllamaManager(num_gpus=num_gpus, model=model)
        self.processors = {}  # GPU-specific processors
        self.performance_stats = {}
        
    def setup_multi_gpu(self):
        """Setup multi-GPU environment"""
        active_gpus = self.gpu_manager.setup_multi_gpu_ollama()
        
        # Create processors for each GPU
        for gpu_id in range(active_gpus):
            llm = self.gpu_manager.get_llm(gpu_id)
            self.processors[gpu_id] = RobustLLMProcessor(llm)
        
        return active_gpus
    
    def process_chunk_on_gpu(self, chunk_data: tuple) -> Dict:
        """Process a chunk of sentences on a specific GPU"""
        chunk, gpu_id, is_refinement, current_translations, feedbacks = chunk_data
        
        processor = self.processors[gpu_id]
        chunk_results = []
        chunk_stats = []
        
        print(f"🔄 GPU {gpu_id} processing {len(chunk)} sentences...")
        
        for i, src_text in enumerate(tqdm(chunk, desc=f"GPU {gpu_id}")):
            try:
                if is_refinement:
                    current_trans = current_translations[i] if i < len(current_translations) else ""
                    feedback = feedbacks[i] if i < len(feedbacks) else ""
                    result = processor.generate_translation_robust(src_text, current_trans, feedback)
                else:
                    result = processor.generate_translation_robust(src_text)
                
                chunk_results.append(result.translation)
                
            except Exception as e:
                print(f"❌ GPU {gpu_id} error on sentence {i}: {e}")
                chunk_results.append(f"[ERROR] {src_text}")
        
        chunk_stats = processor.get_performance_stats()
        
        return {
            'gpu_id': gpu_id,
            'results': chunk_results,
            'stats': chunk_stats
        }
    
    def process_evaluation_chunk_on_gpu(self, chunk_data: tuple) -> Dict:
        """Process evaluation chunk on specific GPU"""
        src_chunk, trans_chunk, gpu_id = chunk_data
        
        processor = self.processors[gpu_id]
        grades = []
        feedbacks = []
        
        print(f"🔍 GPU {gpu_id} evaluating {len(src_chunk)} translations...")
        
        for src_text, translation in tqdm(zip(src_chunk, trans_chunk), desc=f"GPU {gpu_id} Eval"):
            try:
                if not translation.strip() or "[ERROR]" in translation or "[PARSE_ERROR]" in translation:
                    grades.append("1")
                    feedbacks.append("Translation failed or empty")
                    continue
                
                result = processor.evaluate_translation_robust(src_text, translation)
                grades.append(result.grade)
                feedbacks.append(result.feedback)
                
            except Exception as e:
                print(f"❌ GPU {gpu_id} evaluation error: {e}")
                grades.append("1")
                feedbacks.append(f"Evaluation error: {str(e)[:100]}")
        
        chunk_stats = processor.get_performance_stats()
        
        return {
            'gpu_id': gpu_id,
            'grades': grades,
            'feedbacks': feedbacks,
            'stats': chunk_stats
        }


def llm_call_generator_multi_gpu(state: MultiGPUState, refiner: MultiGPULLMRefine) -> HiddenState:
    """Multi-GPU generator based on my_llm_refine.py architecture"""
    print(f"🚀 Multi-GPU Generator iteration {state['i']}")
    start_time = time.time()
    
    active_gpus = refiner.gpu_manager.get_available_gpus()
    if active_gpus == 0:
        raise RuntimeError("No GPUs available")
    
    # Split sentences across GPUs
    src_sentences = state['src']
    chunk_size = len(src_sentences) // active_gpus
    chunks = []
    
    for gpu_id in range(active_gpus):
        start_idx = gpu_id * chunk_size
        if gpu_id == active_gpus - 1:  # Last GPU gets remaining sentences
            end_idx = len(src_sentences)
        else:
            end_idx = (gpu_id + 1) * chunk_size
        
        chunk = src_sentences[start_idx:end_idx]
        
        # Prepare refinement data if available
        is_refinement = state.get("feedback") and any(f for f in state.get("feedback", []))
        current_translations = state.get('tgt', [])
        feedbacks = state.get('feedback', [])
        
        chunks.append((chunk, gpu_id, is_refinement, current_translations[start_idx:end_idx], feedbacks[start_idx:end_idx]))
    
    # Process chunks in parallel
    all_results = []
    total_stats = {'fast_success': 0, 'fixing_success': 0, 'retry_success': 0, 'total_failures': 0}
    
    with ThreadPoolExecutor(max_workers=active_gpus) as executor:
        futures = [executor.submit(refiner.process_chunk_on_gpu, chunk_data) for chunk_data in chunks]
        
        for future in as_completed(futures):
            result = future.result()
            all_results.append(result)
            
            # Aggregate stats
            for key in total_stats:
                total_stats[key] += result['stats'].get(key, 0)
    
    # Combine results in order
    all_results.sort(key=lambda x: x['gpu_id'])
    final_translations = []
    for result in all_results:
        final_translations.extend(result['results'])
    
    processing_time = time.time() - start_time
    print(f"✅ Multi-GPU generation completed in {processing_time:.2f}s")
    print(f"   Success rate: {(sum(total_stats.values()) - total_stats['total_failures']) / max(sum(total_stats.values()), 1) * 100:.1f}%")
    
    return {
        "h_tgt": final_translations,
        "api_calls": sum(total_stats.values()),
        "parse_failures": total_stats['total_failures'],
        "parse_successes": sum(total_stats.values()) - total_stats['total_failures'],
        "retry_attempts": total_stats['fixing_success'] + total_stats['retry_success']
    }


def llm_call_evaluator_multi_gpu(state: OverallState, refiner: MultiGPULLMRefine) -> HiddenState:
    """Multi-GPU evaluator with robust error handling"""
    print(f"🔍 Multi-GPU Evaluator iteration {state['i']}")
    start_time = time.time()
    
    active_gpus = refiner.gpu_manager.get_available_gpus()
    
    # Split data across GPUs
    src_sentences = state['src']
    translations = state['h_tgt']
    chunk_size = len(src_sentences) // active_gpus
    
    chunks = []
    for gpu_id in range(active_gpus):
        start_idx = gpu_id * chunk_size
        if gpu_id == active_gpus - 1:
            end_idx = len(src_sentences)
        else:
            end_idx = (gpu_id + 1) * chunk_size
        
        src_chunk = src_sentences[start_idx:end_idx]
        trans_chunk = translations[start_idx:end_idx]
        chunks.append((src_chunk, trans_chunk, gpu_id))
    
    # Process evaluation in parallel
    all_results = []
    total_stats = {'fast_success': 0, 'fixing_success': 0, 'retry_success': 0, 'total_failures': 0}
    
    with ThreadPoolExecutor(max_workers=active_gpus) as executor:
        futures = [executor.submit(refiner.process_evaluation_chunk_on_gpu, chunk_data) for chunk_data in chunks]
        
        for future in as_completed(futures):
            result = future.result()
            all_results.append(result)
            
            # Aggregate stats
            for key in total_stats:
                total_stats[key] += result['stats'].get(key, 0)
    
    # Combine results in order
    all_results.sort(key=lambda x: x['gpu_id'])
    final_grades = []
    final_feedbacks = []
    
    for result in all_results:
        final_grades.extend(result['grades'])
        final_feedbacks.extend(result['feedbacks'])
    
    processing_time = time.time() - start_time
    print(f"✅ Multi-GPU evaluation completed in {processing_time:.2f}s")
    
    return {
        "h_score": final_grades,
        "h_feedback": final_feedbacks,
        "api_calls": sum(total_stats.values()),
        "parse_failures": total_stats['total_failures'],
        "parse_successes": sum(total_stats.values()) - total_stats['total_failures'],
        "retry_attempts": total_stats['fixing_success'] + total_stats['retry_success']
    }


def loop_condition(state: MultiGPUState):
    """Loop condition for workflow"""
    if state['i'] < state['n']:
        return "continue"
    return "stop"


def simulated_annealing_multi_gpu(state: OverallState) -> MultiGPUState:
    """Enhanced simulated annealing with multi-GPU state management"""
    if state.get('tgt'):
        T = state['temperature']
        cooling_rate = state['cooling_rate']
        
        tgt = state['tgt'].copy()
        feedback = state['feedback'].copy()
        score = state['score'].copy()
        
        accepted_count = 0
        rejected_count = 0
        
        for i in tqdm(range(len(state['src'])), desc="SA Decision"):
            try:
                current_score = float(state['score'][i])
                new_score = float(state['h_score'][i])
                
                delta = new_score - current_score
                acc_point = random.random()
                
                # Simulated annealing acceptance
                if delta > 0:
                    accept = True
                else:
                    p_acc = min(1, math.exp(100 * delta / (state['n'] * T)))
                    accept = p_acc > acc_point
                
                if accept:
                    tgt[i] = state['h_tgt'][i]
                    feedback[i] = state['h_feedback'][i]
                    score[i] = state['h_score'][i]
                    accepted_count += 1
                else:
                    rejected_count += 1
                    
            except (ValueError, TypeError) as e:
                print(f"⚠️ Score parsing error at index {i}: {e}")
                continue
        
        print(f"📊 SA Results: {accepted_count} accepted, {rejected_count} rejected")
        
        return {
            'src': state['src'],
            'tgt': tgt,
            'feedback': feedback,
            'score': score,
            'temperature': T * (1 - cooling_rate),
            'cooling_rate': cooling_rate,
            'n': state['n'],
            'i': state['i'] + 1,
            'gpu_assignments': state.get('gpu_assignments', []),
            'chunk_results': state.get('chunk_results', []),
            'processing_times': state.get('processing_times', [])
        }
    else:
        # First iteration
        return {
            'src': state['src'],
            'tgt': state['h_tgt'],
            'feedback': state['h_feedback'],
            'score': state['h_score'],
            'temperature': state['temperature'],
            'cooling_rate': state['cooling_rate'],
            'n': state['n'],
            'i': state['i'] + 1,
            'gpu_assignments': [],
            'chunk_results': [],
            'processing_times': []
        }


def run_multi_gpu_llm_refine(
    source_file: str,
    output_file: str,
    num_gpus: int = 4,
    model: str = "llama3.1:8b-instruct-fp16",
    max_iterations: int = 6,
    temperature: float = 41.67,
    cooling_rate: float = 0.4
):
    """
    Main function for multi-GPU LLM refinement with robust error handling
    """
    print("🚀 Starting Multi-GPU LLM Refine with Robust Error Handling...")
    
    # Initialize multi-GPU refiner
    refiner = MultiGPULLMRefine(num_gpus=num_gpus, model=model)
    active_gpus = refiner.setup_multi_gpu()
    
    if active_gpus == 0:
        print("❌ No GPUs available, exiting...")
        return
    
    print(f"✅ {active_gpus} GPUs ready for processing")
    
    # Load source data
    with open(source_file, 'r', encoding='utf-8') as f:
        src_sentences = [line.strip() for line in f.readlines() if line.strip()]
    
    print(f"📝 Loaded {len(src_sentences)} sentences for processing")
    
    try:
        # Build workflow (based on my_llm_refine.py structure)
        workflow = StateGraph(OverallState, input=MultiGPUState, output=MultiGPUState)
        
        # Create node functions that use the refiner instance
        def generator_node(state):
            hidden_state = llm_call_generator_multi_gpu(state, refiner)
            return {**state, **hidden_state}
        
        def evaluator_node(state):
            hidden_state = llm_call_evaluator_multi_gpu(state, refiner)
            return {**state, **hidden_state}
        
        # Add nodes (same structure as original)
        workflow.add_node("llm_call_generator", generator_node)
        workflow.add_node("llm_call_evaluator", evaluator_node)
        workflow.add_node("simulated_annealing", simulated_annealing_multi_gpu)
        
        # Add edges (same structure as original)
        workflow.add_edge(START, "llm_call_generator")
        workflow.add_edge("llm_call_generator", "llm_call_evaluator")
        workflow.add_edge("llm_call_evaluator", "simulated_annealing")
        
        workflow.add_conditional_edges(
            "simulated_annealing",
            loop_condition,
            {
                "continue": "llm_call_generator",
                "stop": END,
            },
        )
        
        # Compile and run
        compiled_workflow = workflow.compile()
        
        # Initial state (same structure as original)
        initial_state = {
            "src": src_sentences,
            "temperature": temperature,
            "cooling_rate": cooling_rate,
            "n": max_iterations,
            "i": 0,
            "gpu_assignments": [],
            "chunk_results": [],
            "processing_times": []
        }
        
        # Execute workflow
        print("🔄 Executing multi-GPU optimization workflow...")
        start_time = time.time()
        
        final_state = compiled_workflow.invoke(initial_state, {"recursion_limit": max_iterations * 3})
        
        total_time = time.time() - start_time
        
        # Save results (same format as original)
        with open(output_file, 'w', encoding='utf-8') as f:
            for translation in final_state["tgt"]:
                f.write(f"{translation.replace(chr(10), ' ')}\n")
        
        # Performance summary
        total_sentences = len(final_state['tgt'])
        throughput = total_sentences / total_time
        
        print(f"\n🎯 Multi-GPU LLM Refine Completed!")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   Sentences processed: {total_sentences}")
        print(f"   Active GPUs: {active_gpus}")
        print(f"   Throughput: {throughput:.1f} sentences/second")
        print(f"   Average per GPU: {throughput/active_gpus:.1f} sentences/second")
        print(f"   Output saved to: {output_file}")
        
        return final_state
        
    finally:
        # Always cleanup
        refiner.gpu_manager.cleanup()


if __name__ == "__main__":
    # Example usage
    run_multi_gpu_llm_refine(
        source_file="data.en",
        output_file="data_multi_gpu.vi",
        num_gpus=4,
        model="llama3.1:8b-instruct-fp16",
        max_iterations=6
    )
