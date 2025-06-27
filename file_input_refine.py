"""
Multi-GPU LLM Refine System with File-based Input Support
Enhanced version that can accept both source and existing translation files

Features:
- Support for source + existing translation file input
- Multi-GPU parallel processing with ChatOllama on different ports
- OutputFixingParser and RetryOutputParser for robust error handling
- Fast batch processing and intelligent workload distribution
- Translation refinement mode for improving existing translations
"""

from typing_extensions import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field, ValidationError
from langchain_ollama import ChatOllama
from langchain.output_parsers import OutputFixingParser, RetryOutputParser
from langchain.output_parsers.retry import RetryWithErrorOutputParser
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
import argparse
from typing import List, Dict, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
import traceback


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
    
    # File input tracking
    has_existing_translation: bool
    input_mode: str  # "source_only", "source_and_translation", "refinement_only"

class HiddenState(TypedDict):
    h_tgt: List[str]
    h_feedback: List[str]
    h_score: List[str]
    
    # Performance metrics
    api_calls: int
    parse_successes: int
    parse_failures: int
    # retry_attempts: int

class OverallState(MultiGPUState, HiddenState):
    pass


# Enhanced Pydantic models with robust validation
class Feedback(BaseModel):
    """Enhanced feedback model with robust validation and fallbacks"""
    grade: Literal["1", "1.5", "2", "2.5", "3"] = Field(
        # default="2", 
        description="Translation quality score: 3=Perfect, 2.5=Minor errors, 2=Some errors, 1.5=Significant errors, 1=Poor"
    )
    feedback: str = Field(
        # default="Could not evaluate", 
        description="Detailed feedback explaining the grade",
        min_length=5,
        max_length=1000
    )
    
    def __init__(self, **data):
        # Smart grade extraction and validation
        # print(f"FBDATA: {data}")
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
    """Enhanced translation model with cleanup and validation for multiple language pairs"""
    translation: str = Field(
        # default="", 
        description="High-quality translation in target language",
        min_length=1,
        max_length=2000
    )
    
    def __init__(self, **data):
        # Clean translation text
        # print(data)
        if 'translation' in data and data['translation']:
            translation = str(data['translation']).strip()
            # Remove common parsing artifacts
            translation = re.sub(r'^["\'\`]{1,3}|["\'\`]{1,3}$', '', translation)
            # Remove common prefixes in multiple languages
            translation = re.sub(r'^(Translation|Dịch|Bản dịch|Traducción|Traduction|翻译|翻譯|번역):\s*', '', translation, flags=re.IGNORECASE)
            translation = re.sub(r'\n+', ' ', translation)  # Replace newlines
            translation = re.sub(r'\s+', ' ', translation)  # Normalize spaces
            data['translation'] = translation
        
        super().__init__(**data)


def load_text_files(source_file: str, translation_file: str = None, src_lang: str = "English", tgt_lang: str = "Vietnamese") -> tuple:
    """
    Load source and translation files with proper encoding handling
    
    Args:
        source_file: Path to source language file
        translation_file: Optional path to existing translation file
        src_lang: Source language name (e.g., "English", "French", "Chinese")
        tgt_lang: Target language name (e.g., "Vietnamese", "Spanish", "German")
    
    Returns:
        tuple: (source_lines, translation_lines, input_mode, src_lang, tgt_lang)
    """
    print(f"📂 Loading input files...")
    
    # Load source file
    if not os.path.exists(source_file):
        raise FileNotFoundError(f"Source file not found: {source_file}")
    
    try:
        with open(source_file, 'r', encoding='utf-8') as f:
            source_lines = [line.strip() for line in f.readlines() if line.strip()]
    except UnicodeDecodeError:
        # Try other encodings
        with open(source_file, 'r', encoding='latin-1') as f:
            source_lines = [line.strip() for line in f.readlines() if line.strip()]
    
    print(f"   ✓ Loaded {len(source_lines)} source sentences from {source_file}")
    
    # Load translation file if provided
    translation_lines = []
    input_mode = "source_only"
    
    if translation_file and os.path.exists(translation_file):
        try:
            with open(translation_file, 'r', encoding='utf-8') as f:
                translation_lines = [line.strip() for line in f.readlines() if line.strip()]
        except UnicodeDecodeError:
            with open(translation_file, 'r', encoding='latin-1') as f:
                translation_lines = [line.strip() for line in f.readlines() if line.strip()]
        
        print(f"   ✓ Loaded {len(translation_lines)} existing translations from {translation_file}")
        
        # Validate line counts
        if len(translation_lines) != len(source_lines):
            print(f"   ⚠️  Warning: Source ({len(source_lines)}) and translation ({len(translation_lines)}) line counts don't match")
            # Pad or truncate to match
            if len(translation_lines) < len(source_lines):
                translation_lines.extend([""] * (len(source_lines) - len(translation_lines)))
                print(f"   ↳ Padded translation file with empty lines")
            else:
                translation_lines = translation_lines[:len(source_lines)]
                print(f"   ↳ Truncated translation file to match source")
        
        input_mode = "source_and_translation"
    else:
        if translation_file:
            print(f"   ⚠️  Translation file not found: {translation_file}")
            print(f"   ↳ Proceeding with source-only mode")
            # Initialize empty translations for source-only mode
        translation_lines = [""] * len(source_lines)
        input_mode = "source_only"
    
    print(f"   📋 Input mode: {input_mode}")
    print(f"   🌐 Language pair: {src_lang} → {tgt_lang}")
    return source_lines, translation_lines, input_mode, src_lang, tgt_lang


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
                time.sleep(1)
                
                # Start Ollama server for this GPU
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                env["OLLAMA_HOST"] = f"127.0.0.1:{port}"
                
                process = subprocess.Popen(
                    ["ollama", "serve"],
                    env=env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                
                self.ollama_processes.append(process)
                
                # Wait for server to start
                time.sleep(3)
                
                # Test connection and create LLM instance
                llm = ChatOllama(
                    model=self.model,
                    base_url=f"http://127.0.0.1:{port}",
                    temperature=0.1,
                    num_predict=512,
                    timeout=60
                )
                
                # Test with a simple query
                test_response = llm.invoke("Hello")
                
                self.llm_instances.append(llm)
                self.active_ports.append(port)
                print(f"   ✅ GPU {gpu_id} ready on port {port}")
                
            except Exception as e:
                print(f"   ❌ GPU {gpu_id} failed: {e}")
                continue
        
        active_count = len(self.llm_instances)
        print(f"✅ {active_count}/{self.num_gpus} GPUs successfully initialized")
        
        if active_count == 0:
            raise RuntimeError("No GPUs could be initialized")
        
        return active_count
    
    def get_llm(self, gpu_id: int) -> ChatOllama:
        """Get LLM instance for specific GPU"""
        if gpu_id < len(self.llm_instances):
            return self.llm_instances[gpu_id]
        raise IndexError(f"GPU {gpu_id} not available")
    
    def get_available_gpus(self) -> int:
        """Get number of available GPUs"""
        return len(self.llm_instances)
    
    def cleanup(self):
        """Cleanup all Ollama processes"""
        print("🧹 Cleaning up Ollama processes...")
        for process in self.ollama_processes:
            try:
                process.terminate()
                process.wait(timeout=5)
            except:
                process.kill()
        
        # Kill any remaining processes
        for port in self.active_ports:
            subprocess.run(f"taskkill /F /FI \"COMMANDLINE like *{port}*\" 2>nul", shell=True, check=False)


class RobustLLMProcessor:
    """
    Enhanced LLM processor with 3-tier robustness for translation and evaluation
    Supports multiple language pairs
    """
    
    def __init__(self, llm: ChatOllama, src_lang: str = "English", tgt_lang: str = "Vietnamese"):
        self.llm = llm
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        
        # Setup parsers with multiple fallback tiers
        self.translation_parser = PydanticOutputParser(pydantic_object=Refine)
        self.feedback_parser = PydanticOutputParser(pydantic_object=Feedback)
        
        # Tier 2: OutputFixingParser
        # self.translation_fixing_parser = OutputFixingParser.from_llm(
        #     parser=self.translation_parser, 
        #     llm=self.llm
        # )
        # self.feedback_fixing_parser = OutputFixingParser.from_llm(
        #     parser=self.feedback_parser, 
        #     llm=self.llm
        # )
        
        # Tier 3: RetryOutputParser
        # self.translation_retry_parser = RetryOutputParser.from_llm(
        #     parser=self.translation_parser, 
        #     llm=self.llm
        # )

        # self.feedback_retry_parser = RetryOutputParser.from_llm(
        #     parser=self.feedback_parser, 
        #     llm=self.llm
        # )
        self.translation_retry_parser = RetryWithErrorOutputParser.from_llm(
            parser=self.translation_parser, 
            llm=self.llm,
            max_retries=2
        )

        self.feedback_retry_parser = RetryWithErrorOutputParser.from_llm(
            parser=self.feedback_parser, 
            llm=self.llm,
            max_retries=2
        )



          # Setup optimized prompts with language-specific system message
        self.template = ChatPromptTemplate.from_messages([
            ("system", f"You are a helpful assistant that translates {self.src_lang} to {self.tgt_lang}. Provide accurate, fluent translations."),
            ("human", "{query}"),
        ])
        
        # Create chains with structured output (fast path)
        self.generator_llm = self.llm.with_structured_output(Refine, include_raw=True)
        self.evaluator_llm = self.llm.with_structured_output(Feedback, include_raw=True)
        
        self.generator = self.template | self.generator_llm
        self.evaluator = self.template | self.evaluator_llm
        
        # Performance tracking
        # self.stats = {
        #     'fast_success': 0,
        #     'fixing_success': 0,
        #     'retry_success': 0,
        #     'total_failures': 0
        # }
        self.stats = {
            'total_success': 0,
            'total_failures': 0
        }
    
    def _validate_and_fallback_translation(self, result: Refine, current_translation: str, src_text: str) -> Refine:
        """
        Validate translation result and provide fallback if empty
        
        Args:
            result: Translation result from LLM
            current_translation: Current existing translation (if any)
            src_text: Source text
            
        Returns:
            Valid Refine object with non-empty translation
        """
        # Check if the result translation is empty or just whitespace
        # print(f"result.translation: {result.translation}")
        if not result.translation or not result.translation.strip():
            print(f"⚠️ Empty translation detected, using fallback...")
            # Fallback hierarchy: current_translation > src_text
            if current_translation and current_translation.strip():
                fallback_text = current_translation.strip()
                print(f"   ↳ Using current translation as fallback")
            else:
                fallback_text = src_text.strip()
                print(f"   ↳ Using source text as fallback")
            
            return Refine(translation=fallback_text)
        
        return result
    
    def generate_translation_robust(self, src_text: str, current_translation: str = None, feedback_text: str = None) -> Refine:
        """
        Generate translation with 3-tier robustness:
        1. Fast structured output (my_llm_refine.py approach)
        2. OutputFixingParser for malformed outputs
        3. RetryOutputParser for failed attempts
        """
        # Prepare query based on input mode
        if current_translation and current_translation.strip() and feedback_text:
            # Refinement mode: improve existing translation
            query = f"""Improve this {self.tgt_lang} translation of the {self.src_lang} text based on the feedback provided.

{self.src_lang}: {src_text}
Current {self.tgt_lang} translation: {current_translation}
Feedback for improvement: {feedback_text}

Please provide an improved {self.tgt_lang} translation that addresses the feedback."""
        elif current_translation and current_translation.strip():
            # Review mode: review and potentially improve existing translation
            query = f"""Review and potentially improve this {self.tgt_lang} translation of the {self.src_lang} text.

{self.src_lang}: {src_text}
Current {self.tgt_lang} translation: {current_translation}

If the translation is good, keep it as is. If it can be improved, provide a better version."""
        else:
            # Fresh translation mode
            query = f"Translate this {self.src_lang} text to {self.tgt_lang}: {src_text}"
        response = self.generator.invoke({"query": query})
        raw = response["raw"]
        err = response["parsing_error"]
        if not err:
            # Tier 1: Fast structured output (original approach)
            # print(f"Query: {query}")
            result = response["parsed"]   
            self.stats['total_success'] += 1
            # Validate result and fallback if empty
            # print(f'Response: {result}')
            return self._validate_and_fallback_translation(result, current_translation, src_text)
        else:
            try:
                print(f"🔄 Translation parsing failed, retry...| {query} | {err}")
                prompt = self.template.format_messages(query=query)
                result = self.translation_retry_parser.parse_with_prompt(raw.content, prompt)
                self.stats['total_success'] += 1
                return self._validate_and_fallback_translation(result, current_translation, src_text)
            except Exception as e:
                print(f"❌ Translation parsing failed, use current translation instead! | {query} | {e}")
                self.stats['total_failures'] += 1
                # Use validation function for consistent fallback logic
                fallback_text = current_translation if current_translation and current_translation.strip() else src_text
                fallback_result = Refine(translation=fallback_text)
                # return self._validate_and_fallback_translation(fallback_result, current_translation, src_text)
                return fallback_result
            
        # except Exception as e:
        #     print(f"🔄 Fast generation failed, trying OutputFixingParser...")
            
        #     try:
        #         # Tier 2: OutputFixingParser
        #         response = self.llm.invoke(self.template.format_messages(query=query))
        #         fixed_result = self.translation_fixing_parser.parse(response.content)
        #         self.stats['fixing_success'] += 1
        #         # Validate result and fallback if empty
        #         return self._validate_and_fallback_translation(fixed_result, current_translation, src_text)
                
        #     except Exception as e2:
        #         print(f"🔄 OutputFixingParser failed, trying RetryOutputParser...")
                
        #         try:
        #             # Tier 3: RetryOutputParser (full retry with context)
        #             response = self.llm.invoke(self.template.format_messages(query=query))
        #             retry_result = self.translation_retry_parser.parse_with_prompt(
        #                 response.content, 
        #                 self.template.format_messages(query=query)
        #             )
        #             self.stats['retry_success'] += 1                    # Validate result and fallback if empty
        #             return self._validate_and_fallback_translation(retry_result, current_translation, src_text)
                    
        #         except Exception as e3:
        #             # Final fallback
        #             print(f"❌ All parsing tiers failed: {e3}")
        #             self.stats['total_failures'] += 1
        #             # Use validation function for consistent fallback logic
        #             fallback_text = current_translation if current_translation and current_translation.strip() else src_text
        #             fallback_result = Refine(translation=fallback_text)
        #             # return self._validate_and_fallback_translation(fallback_result, current_translation, src_text)
        #             return fallback_result

        # else:
        #     # Final fallback
        #     print(f"❌ Translation parsing failed, use current translation instead!")
        #     self.stats['total_failures'] += 1
        #     # Use validation function for consistent fallback logic
        #     fallback_text = current_translation if current_translation and current_translation.strip() else src_text
        #     fallback_result = Refine(translation=fallback_text)
        #     # return self._validate_and_fallback_translation(fallback_result, current_translation, src_text)
        #     return fallback_result
    def evaluate_translation_robust(self, src_text: str, translation: str) -> Feedback:
        """
        Evaluate translation with 3-tier robustness
        """
        query = f"""You are a language expert. Rate this {self.tgt_lang} translation of the {self.src_lang} text.

Rate from 1-3:
3: Perfect translation
2.5: Minor errors but understandable  
2: Some errors but mostly correct
1.5: Significant errors, partially understandable
1: Poor translation, hard to understand

{self.src_lang}: {src_text}
{self.tgt_lang}: {translation}

Provide grade and detailed feedback."""
        response = self.evaluator.invoke({"query": query})
        raw = response["raw"]
        err = response["parsing_error"]
        if not err:
            # Tier 1: Fast structured output (original approach)
            # print(f"Query: {query}")
            result = response["parsed"]   
            self.stats['total_success'] += 1
            # Validate result and fallback if empty
            # print(f'Response: {result}')
            return result
        else:
            try:
                print(f"🔄 Evaluation parsing failed, retry...")
                prompt = self.template.format_messages(query=query)
                result = self.feedback_retry_parser.parse_with_prompt(raw.content, prompt)
                self.stats['total_success'] += 1
                return result
            except Exception as e:
                print(f"❌ Evaluation parsing failed, use default evaluation instead!")
                # print(traceback.print_exc())
                self.stats['total_failures'] += 1
                # return Feedback(grade="2", feedback=f"Evaluation failed: {str(e3)[:100]}")
                return Feedback(grade="2", feedback=f"If the translation is good, keep it as is. If it can be improved, provide a better version.")
        # try:
        #     # Tier 1: Fast structured output
        #     # print(f"FBQR: {query}")
        #     result = self.evaluator.invoke({"query": query})
        #     self.stats['total_success'] += 1
        #     # print(f"Feedback: {result}")
        #     return result
            
        # except Exception as e:
        #     print(f"🔄 Fast evaluation failed, trying OutputFixingParser...")
        #     print(traceback.print_exc())
            
        #     try:
        #         # Tier 2: OutputFixingParser
        #         response = self.llm.invoke(self.template.format_messages(query=query))
        #         fixed_result = self.feedback_fixing_parser.parse(response.content)
        #         self.stats['fixing_success'] += 1
        #         return fixed_result
                
        #     except Exception as e2:
        #         print(f"🔄 OutputFixingParser failed, trying RetryOutputParser...")
        #         print(traceback.print_exc())
        #         try:
        #             # Tier 3: RetryOutputParser
        #             response = self.llm.invoke(self.template.format_messages(query=query))
        #             retry_result = self.feedback_retry_parser.parse_with_prompt(
        #                 response.content,
        #                 self.template.format_messages(query=query)
        #             )
        #             self.stats['retry_success'] += 1
        #             return retry_result
                    
        #         except Exception as e3:
        #             print(f"❌ All evaluation parsing failed: {e3}")
        #             print(traceback.print_exc())
        #             self.stats['total_failures'] += 1
        #             # return Feedback(grade="2", feedback=f"Evaluation failed: {str(e3)[:100]}")
        #             return Feedback(grade="2", feedback=f"If the translation is good, keep it as is. If it can be improved, provide a better version.")
        # except Exception as e:
        #     print(f"❌ Evaluation parsing failed, use default evaluation instead!")
        #     # print(traceback.print_exc())
        #     self.stats['total_failures'] += 1
        #     # return Feedback(grade="2", feedback=f"Evaluation failed: {str(e3)[:100]}")
        #     return Feedback(grade="2", feedback=f"If the translation is good, keep it as is. If it can be improved, provide a better version.")
    
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
    Main class combining multi-GPU processing with robust error handling and file input support
    Supports multiple language pairs
    """
    
    def __init__(self, num_gpus: int = 4, model: str = "llama3.1:8b-instruct-fp16", src_lang: str = "English", tgt_lang: str = "Vietnamese"):
        self.gpu_manager = MultiGPUOllamaManager(num_gpus=num_gpus, model=model)
        self.processors = {}  # GPU-specific processors
        self.performance_stats = {}
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        
    def setup_multi_gpu(self):
        """Setup multi-GPU environment"""
        active_gpus = self.gpu_manager.setup_multi_gpu_ollama()
          # Create processors for each GPU
        for gpu_id in range(active_gpus):
            llm = self.gpu_manager.get_llm(gpu_id)
            self.processors[gpu_id] = RobustLLMProcessor(llm, self.src_lang, self.tgt_lang)
        
        return active_gpus
    
    def process_chunk_on_gpu(self, chunk_data: tuple) -> Dict:
        """Process a chunk of sentences on a specific GPU"""
        has_feedback = False
        if len(chunk_data) == 4:
            chunk_src, chunk_tgt, gpu_id, input_mode = chunk_data
        else:
            chunk_src, chunk_tgt, chunk_fb, gpu_id, input_mode = chunk_data
            has_feedback = True
        
        processor = self.processors[gpu_id]
        chunk_results = []
        
        print(f"🔄 GPU {gpu_id} processing {len(chunk_src)} sentences in {input_mode} mode, feedback: {has_feedback}...")
        
        for i, (src_text, existing_translation) in enumerate(tqdm(zip(chunk_src, chunk_tgt), desc=f"GPU {gpu_id}")):
            try:
                if input_mode == "source_and_translation":
                    # Use existing translation as base for refinement
                    if has_feedback:
                        result = processor.generate_translation_robust(src_text, existing_translation, chunk_fb[i])
                    else:
                        result = processor.generate_translation_robust(src_text, existing_translation)
                else:
                    # Fresh translation
                    result = processor.generate_translation_robust(src_text)
                
                chunk_results.append(result.translation)
                
            except Exception as e:
                print(f"❌ GPU {gpu_id} error on sentence {i}: {e}")
                # Fallback to existing translation or source
                fallback = existing_translation if existing_translation.strip() else src_text
                chunk_results.append(fallback)
        
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
                # feedbacks.append(f"Evaluation error: {str(e)[:100]}")
                feedbacks.append(f"If the translation is good, keep it as is. If it can be improved, provide a better version.")
        
        chunk_stats = processor.get_performance_stats()
        
        return {
            'gpu_id': gpu_id,
            'grades': grades,
            'feedbacks': feedbacks,
            'stats': chunk_stats
        }


def llm_call_generator_multi_gpu(state: OverallState, refiner: MultiGPULLMRefine) -> HiddenState:
    """Multi-GPU generator with file input support"""
    print(f"🚀 Multi-GPU Generator iteration {state['i']} (Mode: {state.get('input_mode', 'unknown')})")
    start_time = time.time()
    
    active_gpus = refiner.gpu_manager.get_available_gpus()
    if active_gpus == 0:
        raise RuntimeError("No GPUs available")
    
    # Split sentences across GPUs
    src_sentences = state['src']
    existing_translations = state.get('tgt', [""] * len(src_sentences))
    feedbacks = state.get("h_feedback")
    input_mode = state.get('input_mode', 'source_only')
    
    chunk_size = len(src_sentences) // active_gpus
    chunks = []
    
    for gpu_id in range(active_gpus):
        start_idx = gpu_id * chunk_size
        if gpu_id == active_gpus - 1:  # Last GPU gets remaining sentences
            end_idx = len(src_sentences)
        else:
            end_idx = (gpu_id + 1) * chunk_size
        
        chunk_src = src_sentences[start_idx:end_idx]
        chunk_tgt = existing_translations[start_idx:end_idx]
        if feedbacks:
            chunk_fb = feedbacks[start_idx:end_idx]
            chunks.append((chunk_src, chunk_tgt, chunk_fb, gpu_id, input_mode))
        else:
            chunks.append((chunk_src, chunk_tgt, gpu_id, input_mode))
    
    # Process chunks in parallel
    all_results = []
    # total_stats = {'total_success': 0, 'fixing_success': 0, 'retry_success': 0, 'total_failures': 0}
    total_stats = {'total_success': 0, 'total_failures': 0}
    
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
        # "retry_attempts": total_stats['fixing_success'] + total_stats['retry_success']
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
    # total_stats = {'total_success': 0, 'fixing_success': 0, 'retry_success': 0, 'total_failures': 0}
    total_stats = {'total_success': 0, 'total_failures': 0}
    
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
        # "retry_attempts": total_stats['fixing_success'] + total_stats['retry_success']
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
            'processing_times': state.get('processing_times', []),
            'has_existing_translation': state.get('has_existing_translation', False),
            'input_mode': state.get('input_mode', 'source_only')
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
            'processing_times': [],
            'has_existing_translation': state.get('has_existing_translation', False),
            'input_mode': state.get('input_mode', 'source_only')
        }


def run_multi_gpu_llm_refine_with_files(
    source_file: str,
    output_file: str,
    translation_file: str = None,
    src_lang: str = "English",
    tgt_lang: str = "Vietnamese",
    num_gpus: int = 4,
    model: str = "llama3.1:8b-instruct-fp16",
    max_iterations: int = 6,
    temperature: float = 41.67,
    cooling_rate: float = 0.4
):
    """
    Main function for multi-GPU LLM refinement with file input support and multiple language pairs
    
    Args:
        source_file: Path to source language file (required)
        output_file: Path for output translations
        translation_file: Path to existing translation file (optional)
        src_lang: Source language name (e.g., "English", "French", "Chinese")
        tgt_lang: Target language name (e.g., "Vietnamese", "Spanish", "German")
        num_gpus: Number of GPUs to use
        model: LLM model to use
        max_iterations: Maximum refinement iterations
        temperature: Initial temperature for simulated annealing
        cooling_rate: Cooling rate for simulated annealing
    """
    print("🚀 Starting Multi-GPU LLM Refine with File Input Support...")
    print(f"   Source file: {source_file}")
    if translation_file:
        print(f"   Translation file: {translation_file}")
    print(f"   Output file: {output_file}")
    print(f"   Language pair: {src_lang} → {tgt_lang}")
    print(f"   GPUs: {num_gpus}, Model: {model}")
    
    # Load input files
    src_sentences, tgt_sentences, input_mode, actual_src_lang, actual_tgt_lang = load_text_files(source_file, translation_file, src_lang, tgt_lang)
    
    # Initialize multi-GPU refiner
    refiner = MultiGPULLMRefine(num_gpus=num_gpus, model=model, src_lang=actual_src_lang, tgt_lang=actual_tgt_lang)
    
    try:
        active_gpus = refiner.setup_multi_gpu()
        
        # Build the workflow graph
        workflow = StateGraph(OverallState)
        
        # Add nodes
        workflow.add_node("generator", lambda state: llm_call_generator_multi_gpu(state, refiner))
        workflow.add_node("evaluator", lambda state: llm_call_evaluator_multi_gpu(state, refiner))
        workflow.add_node("annealing", simulated_annealing_multi_gpu)
        
        # Set entry point
        workflow.set_entry_point("generator")
        
        # Add edges
        workflow.add_edge("generator", "evaluator")
        workflow.add_edge("evaluator", "annealing")
        workflow.add_conditional_edges(
            "annealing",
            loop_condition,
            {
                "continue": "generator",
                "stop": END
            }
        )
        
        # Compile workflow
        compiled_workflow = workflow.compile()
        
        # Prepare initial state
        initial_state = {
            "src": src_sentences,
            "tgt": tgt_sentences,
            "feedback": [""] * len(src_sentences),
            "score": ["2"] * len(src_sentences),
            "temperature": temperature,
            "cooling_rate": cooling_rate,
            "n": max_iterations,
            "i": 0,
            "gpu_assignments": [],
            "chunk_results": [],
            "processing_times": [],
            "has_existing_translation": input_mode == "source_and_translation",
            "input_mode": input_mode
        }
        
        # Execute workflow
        print("🔄 Executing multi-GPU optimization workflow...")
        start_time = time.time()
        
        final_state = compiled_workflow.invoke(initial_state, {"recursion_limit": max_iterations * 10})
        
        total_time = time.time() - start_time
        
        # Save results
        with open(output_file, 'w', encoding='utf-8') as f:
            for translation in final_state["tgt"]:
                f.write(f"{translation.replace(chr(10), ' ')}\n")
        
        # Performance summary
        total_sentences = len(final_state['tgt'])
        throughput = total_sentences / total_time
        
        print(f"\n🎯 Multi-GPU LLM Refine Completed!")
        print(f"   Input mode: {input_mode}")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   Sentences processed: {total_sentences}")
        print(f"   Active GPUs: {active_gpus}")
        print(f"   Throughput: {throughput:.1f} sentences/second")
        print(f"   Average per GPU: {throughput/active_gpus:.1f} sentences/second")
        print(f"   Output saved to: {output_file}")
        
        # Show improvement statistics if we had existing translations
        if input_mode == "source_and_translation":
            original_file = f"{output_file}.original"
            with open(original_file, 'w', encoding='utf-8') as f:
                for translation in tgt_sentences:
                    f.write(f"{translation}\n")
            print(f"   Original translations saved to: {original_file}")
        
        return final_state
        
    finally:
        # Always cleanup
        refiner.gpu_manager.cleanup()


def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(description='Multi-GPU LLM Translation Refinement System with Multi-Language Support')
    
    parser.add_argument('source_file', help='Path to source language file')
    parser.add_argument('output_file', help='Path for output translations')
    parser.add_argument('--translation-file', '-t', help='Path to existing translation file (optional)')
    parser.add_argument('--src-lang', '-s', default='English', help='Source language name (default: English)')
    parser.add_argument('--tgt-lang', '-d', default='Vietnamese', help='Target language name (default: Vietnamese)')
    parser.add_argument('--num-gpus', '-g', type=int, default=4, help='Number of GPUs to use (default: 4)')
    parser.add_argument('--model', '-m', default='llama3.1:8b-instruct-fp16', help='LLM model to use')
    parser.add_argument('--max-iterations', '-i', type=int, default=6, help='Maximum refinement iterations (default: 6)')
    parser.add_argument('--temperature', type=float, default=41.67, help='Initial temperature for simulated annealing')
    parser.add_argument('--cooling-rate', type=float, default=0.4, help='Cooling rate for simulated annealing')
    
    args = parser.parse_args()
    
    # Run the refinement
    run_multi_gpu_llm_refine_with_files(
        source_file=args.source_file,
        output_file=args.output_file,
        translation_file=args.translation_file,
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
        num_gpus=args.num_gpus,
        model=args.model,
        max_iterations=args.max_iterations,
        temperature=args.temperature,
        cooling_rate=args.cooling_rate
    )


if __name__ == "__main__":
    main()
