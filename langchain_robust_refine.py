"""
LangChain-based Robust LLM Refine System
Implements LangChain's recommended retry strategies from:
https://python.langchain.com/docs/how_to/output_parser_retry/

Features:
- OutputFixingParser for format correction
- RetryOutputParser for incomplete outputs  
- RunnableParallel chains for robust execution
- Multi-GPU batch processing
- Comprehensive error handling
"""

from typing_extensions import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field, ValidationError
from langchain_ollama import ChatOllama
from langchain.output_parsers import OutputFixingParser, RetryOutputParser
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.exceptions import OutputParserException
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel
from tqdm import tqdm
import random
import math
import asyncio
import json
import re
import hashlib
import subprocess
import time
import os
from typing import List, Dict, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp

class PydanticOutputParser:
    def __init__(self, pydantic_object):
        self.pydantic_object = pydantic_object
        
    def parse(self, text):
        # Simplified parsing logic
        import json
        import re
        
        # Try JSON extraction
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            return self.pydantic_object(**data)
        
        # Fallback parsing for specific types
        if self.pydantic_object == Translation:
            return Translation(translation=text.strip())
        elif self.pydantic_object == Feedback:
            # Extract grade and feedback from text
            grade_match = re.search(r'(\d\.?\d?)', text)
            grade = grade_match.group(1) if grade_match else "2"
            return Feedback(grade=grade, feedback=text[:200])
            
        raise Exception("Could not parse output")
    
    def get_format_instructions(self):
        if self.pydantic_object == Translation:
            return 'Return JSON with "translation" field containing the Vietnamese translation.'
        elif self.pydantic_object == Feedback:
            return 'Return JSON with "grade" (1-3) and "feedback" fields for translation evaluation.'

class PromptTemplate:
    def __init__(self, template, input_variables, partial_variables=None):
        self.template = template
        self.input_variables = input_variables
        self.partial_variables = partial_variables or {}
        
    def format_prompt(self, **kwargs):
        return PromptValue(self.template.format(**kwargs, **self.partial_variables))

class PromptValue:
    def __init__(self, text):
        self.text = text
        
    def to_string(self):
        return self.text

class OutputParserException(Exception):
    pass


# Enhanced State models
class RobustState(TypedDict):
    src: List[str]
    tgt: List[str] 
    feedback: List[str]
    score: List[str]
    temperature: float
    cooling_rate: float
    n: int
    i: int
    
    # Error tracking
    parse_failures: int
    retry_count: List[int]
    failed_sentences: List[int]


class HiddenState(TypedDict):
    h_tgt: List[str]
    h_feedback: List[str] 
    h_score: List[str]
    
    # Performance metrics
    api_calls: int
    parse_successes: int
    parse_failures: int


class OverallState(RobustState, HiddenState):
    pass


# Pydantic models với robust validation
class Feedback(BaseModel):
    """Enhanced feedback model với validation"""
    grade: str = Field(default="2", description="Score: 1, 1.5, 2, 2.5, or 3")
    feedback: str = Field(default="Could not evaluate", description="Detailed feedback")
    
    def model_post_init(self, __context):
        # Validate and fix grade
        valid_grades = ["1", "1.5", "2", "2.5", "3"]
        if self.grade not in valid_grades:
            # Try to extract valid grade from string
            import re
            grade_match = re.search(r'([123]\.?5?)', str(self.grade))
            if grade_match:
                self.grade = grade_match.group(1)
            else:
                self.grade = "2"  # Default fallback


class Translation(BaseModel):
    """Enhanced translation model với validation"""
    translation: str = Field(default="", description="Vietnamese translation")
    
    def model_post_init(self, __context):
        # Clean up translation text
        if self.translation:
            self.translation = self.translation.strip()
            # Remove common parsing artifacts
            self.translation = re.sub(r'^["\']|["\']$', '', self.translation)
            self.translation = re.sub(r'^Translation:\s*', '', self.translation, flags=re.IGNORECASE)


class MultiGPUOllamaManager:
    """Enhanced GPU manager với automatic cleanup"""
    
    def __init__(self, num_gpus: int = 8, base_port: int = 11434):
        self.num_gpus = num_gpus
        self.base_port = base_port
        self.ollama_processes = []
        self.llm_instances = []
        
    def setup_multi_gpu_ollama(self):
        """Setup Ollama instances with better error handling"""
        print(f"🚀 Setting up Ollama on {self.num_gpus} GPUs...")
        
        for gpu_id in range(self.num_gpus):
            port = self.base_port + gpu_id
            
            try:
                # Kill existing processes
                subprocess.run(f"pkill -f 'ollama.*{port}'", shell=True, check=False)
                time.sleep(2)
                
                # Setup environment
                env = {
                    **os.environ,
                    'CUDA_VISIBLE_DEVICES': str(gpu_id),
                    'OLLAMA_HOST': f'0.0.0.0:{port}'
                }
                
                # Start Ollama
                process = subprocess.Popen(
                    ['ollama', 'serve'],
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                
                self.ollama_processes.append(process)
                time.sleep(5)  # Wait for startup
                
                # Create LLM instance
                llm = ChatOllama(
                    model="llama3.1:8b-instruct-fp16",
                    base_url=f"http://localhost:{port}",
                    temperature=0.6,
                    top_p=0.9,
                    num_predict=128,
                )
                
                self.llm_instances.append(llm)
                print(f"✅ GPU {gpu_id} ready on port {port}")
                
            except Exception as e:
                print(f"❌ Failed to setup GPU {gpu_id}: {e}")
                
        return len(self.llm_instances)
    
    def get_llm(self, gpu_id: int) -> ChatOllama:
        if gpu_id < len(self.llm_instances):
            return self.llm_instances[gpu_id]
        return self.llm_instances[0]  # Fallback to first GPU
    
    def cleanup(self):
        print("🧹 Cleaning up Ollama processes...")
        for process in self.ollama_processes:
            try:
                process.terminate()
                process.wait(timeout=5)
            except:
                process.kill()


class LangChainRobustParsers:
    """
    LangChain-based robust parsing với comprehensive error handling
    Theo best practices từ: https://python.langchain.com/docs/how_to/output_parser_retry/
    """
    
    def __init__(self, llm: ChatOllama):
        self.llm = llm
        
        # Setup base parsers
        self.translation_parser = PydanticOutputParser(pydantic_object=Translation)
        self.feedback_parser = PydanticOutputParser(pydantic_object=Feedback)
        
        # Create fixing parsers - sẽ fix malformed output
        self.translation_fixing_parser = OutputFixingParser.from_llm(
            parser=self.translation_parser, 
            llm=self.llm
        )
        self.feedback_fixing_parser = OutputFixingParser.from_llm(
            parser=self.feedback_parser, 
            llm=self.llm
        )
        
        # Create retry parsers - sẽ retry với full prompt context
        self.translation_retry_parser = RetryOutputParser.from_llm(
            parser=self.translation_parser, 
            llm=self.llm
        )
        self.feedback_retry_parser = RetryOutputParser.from_llm(
            parser=self.feedback_parser, 
            llm=self.llm
        )
        
        # Setup structured prompts
        self.translation_prompt = PromptTemplate(
            template="""Dịch câu tiếng Anh sau sang tiếng Việt chính xác và tự nhiên.

{format_instructions}

Câu tiếng Anh: {source_text}

Bản dịch tiếng Việt:""",
            input_variables=["source_text"],
            partial_variables={"format_instructions": self.translation_parser.get_format_instructions()}
        )
        
        self.refinement_prompt = PromptTemplate(
            template="""Cải thiện bản dịch tiếng Việt dựa trên phản hồi.

{format_instructions}

Câu gốc: {source_text}
Bản dịch hiện tại: {current_translation}
Phản hồi cải thiện: {feedback}

Bản dịch được cải thiện:""",
            input_variables=["source_text", "current_translation", "feedback"],
            partial_variables={"format_instructions": self.translation_parser.get_format_instructions()}
        )
        
        self.evaluation_prompt = PromptTemplate(
            template="""Đánh giá chất lượng bản dịch tiếng Việt.

{format_instructions}

Câu gốc (tiếng Anh): {source_text}
Bản dịch (tiếng Việt): {translation}

Thang điểm:
- 3: Hoàn hảo (chính xác và tự nhiên)
- 2.5: Tốt (lỗi nhỏ)
- 2: Khá (hiểu được, có lỗi)
- 1.5: Trung bình (hiểu một phần)
- 1: Kém (khó hiểu)

Đánh giá chi tiết:""",
            input_variables=["source_text", "translation"],
            partial_variables={"format_instructions": self.feedback_parser.get_format_instructions()}
        )
    
    def parse_translation_with_retry(self, source_text: str, current_translation: str = None, feedback: str = None) -> Translation:
        """
        Parse translation với 3-tier fallback system:
        1. Regular PydanticOutputParser
        2. OutputFixingParser (fix malformed output)
        3. RetryOutputParser (retry với full context)
        """
        # Choose appropriate prompt
        if current_translation and feedback:
            prompt_value = self.refinement_prompt.format_prompt(
                source_text=source_text,
                current_translation=current_translation,
                feedback=feedback
            )
        else:
            prompt_value = self.translation_prompt.format_prompt(source_text=source_text)
        
        try:
            # Tier 1: Try regular parsing
            response = self.llm.invoke(prompt_value.to_string())
            result = self.translation_parser.parse(response.content)
            print("✅ Translation parsed successfully (Tier 1)")
            return result
            
        except OutputParserException as e:
            print("🔄 Tier 1 failed, trying OutputFixingParser (Tier 2)...")
            
            try:
                # Tier 2: Try OutputFixingParser
                result = self.translation_fixing_parser.parse(response.content)
                print("✅ Translation fixed and parsed (Tier 2)")
                return result
                
            except Exception as e:
                print("🔄 Tier 2 failed, trying RetryOutputParser (Tier 3)...")
                
                try:
                    # Tier 3: Try RetryOutputParser with full context
                    result = self.translation_retry_parser.parse_with_prompt(
                        response.content, 
                        prompt_value
                    )
                    print("✅ Translation retried and parsed (Tier 3)")
                    return result
                    
                except Exception as e:
                    print(f"❌ All parsing tiers failed: {e}")
                    # Ultimate fallback
                    return Translation(translation=f"[PARSE_FAILED] {source_text}")
    
    def parse_feedback_with_retry(self, source_text: str, translation: str) -> Feedback:
        """
        Parse feedback với 3-tier fallback system
        """
        prompt_value = self.evaluation_prompt.format_prompt(
            source_text=source_text,
            translation=translation
        )
        
        try:
            # Tier 1: Try regular parsing
            response = self.llm.invoke(prompt_value.to_string())
            result = self.feedback_parser.parse(response.content)
            print("✅ Feedback parsed successfully (Tier 1)")
            return result
            
        except OutputParserException as e:
            print("🔄 Tier 1 failed, trying OutputFixingParser (Tier 2)...")
            
            try:
                # Tier 2: Try OutputFixingParser
                result = self.feedback_fixing_parser.parse(response.content)
                print("✅ Feedback fixed and parsed (Tier 2)")
                return result
                
            except Exception as e:
                print("🔄 Tier 2 failed, trying RetryOutputParser (Tier 3)...")
                
                try:
                    # Tier 3: Try RetryOutputParser
                    result = self.feedback_retry_parser.parse_with_prompt(
                        response.content,
                        prompt_value
                    )
                    print("✅ Feedback retried and parsed (Tier 3)")
                    return result
                    
                except Exception as e:
                    print(f"❌ All feedback parsing tiers failed: {e}")
                    # Ultimate fallback
                    return Feedback(
                        grade="2", 
                        feedback=f"Parsing failed: {str(e)[:100]}"
                    )


class LangChainRobustLLMRefine:
    """
    Main class combining LangChain robust parsing với batch processing
    """
    
    def __init__(self, gpu_manager: Optional[MultiGPUOllamaManager] = None):
        self.gpu_manager = gpu_manager or MultiGPUOllamaManager(num_gpus=1)
        
        # Caches for performance
        self.translation_cache = {}
        self.evaluation_cache = {}
        
        # Parser instances for each GPU
        self.parsers = {}
        
        # Statistics
        self.stats = {
            'total_api_calls': 0,
            'tier1_successes': 0,
            'tier2_successes': 0, 
            'tier3_successes': 0,
            'total_failures': 0
        }
    
    def get_or_create_parser(self, gpu_id: int) -> LangChainRobustParsers:
        """Get or create parser for specific GPU"""
        if gpu_id not in self.parsers:
            llm = self.gpu_manager.get_llm(gpu_id)
            self.parsers[gpu_id] = LangChainRobustParsers(llm)
        return self.parsers[gpu_id]
    
    def llm_call_generator_robust(self, state: RobustState, gpu_id: int = 0) -> HiddenState:
        """
        Enhanced generator với LangChain parsing
        """
        print(f"🔄 Generator iteration {state['i']} on GPU {gpu_id}")
        
        parser = self.get_or_create_parser(gpu_id)
        response = []
        api_calls = 0
        parse_failures = 0
        
        is_refinement = state.get("feedback") and any(f for f in state.get("feedback", []))
        
        for i in tqdm(range(len(state['src'])), desc=f"GPU {gpu_id} Generating"):
            src_text = state['src'][i]
            
            # Check cache
            cache_key = f"{src_text}_{state.get('feedback', [''])[i] if is_refinement else ''}"
            if cache_key in self.translation_cache:
                response.append(self.translation_cache[cache_key])
                continue
            
            try:
                if is_refinement and state['tgt'][i]:
                    result = parser.parse_translation_with_retry(
                        source_text=src_text,
                        current_translation=state['tgt'][i],
                        feedback=state.get('feedback', [''])[i]
                    )
                else:
                    result = parser.parse_translation_with_retry(source_text=src_text)
                
                api_calls += 1
                translation = result.translation.strip()
                
                if translation and not translation.startswith("[PARSE_FAILED]"):
                    self.translation_cache[cache_key] = translation
                    response.append(translation)
                else:
                    parse_failures += 1
                    response.append(f"[FAILED] {src_text}")
                    
            except Exception as e:
                print(f"⚠️  Generator failed for sentence {i}: {e}")
                parse_failures += 1
                response.append(f"[ERROR] {src_text}")
        
        return {
            "h_tgt": response,
            "api_calls": api_calls,
            "parse_failures": parse_failures
        }
    
    def llm_call_evaluator_robust(self, state: OverallState, gpu_id: int = 0) -> HiddenState:
        """
        Enhanced evaluator với LangChain parsing
        """
        print(f"🔄 Evaluator iteration {state['i']} on GPU {gpu_id}")
        
        parser = self.get_or_create_parser(gpu_id)
        grades = []
        feedbacks = []
        api_calls = 0
        parse_failures = 0
        
        for i in tqdm(range(len(state['src'])), desc=f"GPU {gpu_id} Evaluating"):
            src_text = state['src'][i]
            translation = state['h_tgt'][i]
            
            # Skip failed translations
            if not translation.strip() or any(marker in translation for marker in ["[FAILED]", "[ERROR]", "[PARSE_FAILED]"]):
                grades.append("1")
                feedbacks.append("Translation failed or empty")
                continue
            
            # Check cache
            cache_key = hashlib.md5(f"{src_text}|||{translation}".encode()).hexdigest()
            if cache_key in self.evaluation_cache:
                grade, feedback_text = self.evaluation_cache[cache_key]
                grades.append(grade)
                feedbacks.append(feedback_text)
                continue
            
            try:
                result = parser.parse_feedback_with_retry(
                    source_text=src_text,
                    translation=translation
                )
                
                api_calls += 1
                
                # Validate and store
                grade = result.grade
                feedback_text = result.feedback
                
                # Double-check grade validity
                try:
                    float_grade = float(grade)
                    if 1.0 <= float_grade <= 3.0:
                        self.evaluation_cache[cache_key] = (grade, feedback_text)
                        grades.append(grade)
                        feedbacks.append(feedback_text)
                    else:
                        raise ValueError(f"Invalid grade: {grade}")
                except ValueError:
                    grades.append("2")
                    feedbacks.append("Grade validation failed, used default")
                    parse_failures += 1
                        
            except Exception as e:
                print(f"⚠️  Evaluator failed for sentence {i}: {e}")
                grades.append("2")
                feedbacks.append("Evaluation completely failed")
                parse_failures += 1
        
        return {
            "h_score": grades,
            "h_feedback": feedbacks,
            "api_calls": api_calls,
            "parse_failures": parse_failures
        }


def simulated_annealing_robust(state: OverallState) -> RobustState:
    """Enhanced SA với comprehensive statistics"""
    if state.get('tgt'):
        T = state['temperature']
        cooling_rate = state['cooling_rate']
        
        tgt = state['tgt'].copy()
        feedback = state['feedback'].copy()
        score = state['score'].copy()
        
        accepted_count = 0
        rejected_count = 0
        
        for i in tqdm(range(len(state['src'])), desc="SA Decision"):
            current_score = float(state['score'][i])
            new_score = float(state['h_score'][i])
            
            delta = new_score - current_score
            
            # SA acceptance logic
            if delta > 0:
                accept = True
            else:
                if T > 0:
                    prob = math.exp(delta / T)
                    accept = random.random() < prob
                else:
                    accept = False
            
            if accept:
                tgt[i] = state['h_tgt'][i]
                feedback[i] = state['h_feedback'][i]
                score[i] = state['h_score'][i]
                accepted_count += 1
            else:
                rejected_count += 1
        
        print(f"📊 SA Results: {accepted_count} accepted, {rejected_count} rejected")
        
        return {
            'tgt': tgt,
            'feedback': feedback,
            'score': score,
            'temperature': T * (1 - cooling_rate),
            'i': state['i'] + 1,
            'parse_failures': state.get('parse_failures', 0),
            'retry_count': state.get('retry_count', []),
            'failed_sentences': state.get('failed_sentences', [])
        }
    else:
        # First iteration
        return {
            'tgt': state['h_tgt'],
            'feedback': state['h_feedback'],
            'score': state['h_score'],
            'temperature': state['temperature'],
            'i': state['i'] + 1,
            'parse_failures': 0,
            'retry_count': [0] * len(state['h_tgt']),
            'failed_sentences': []
        }


def run_langchain_robust_processing(
    source_file: str,
    output_file: str,
    num_gpus: int = 8,
    max_iterations: int = 6,
    temperature: float = 41.67,
    cooling_rate: float = 0.4
):
    """
    Main function sử dụng LangChain robust parsing
    """
    print("🚀 Starting LangChain Robust LLM Refine...")
    
    # Setup GPU manager
    gpu_manager = MultiGPUOllamaManager(num_gpus=num_gpus)
    active_gpus = gpu_manager.setup_multi_gpu_ollama()
    
    if active_gpus == 0:
        print("❌ No GPUs available, exiting...")
        return
    
    print(f"✅ {active_gpus} GPUs ready for processing")
    
    # Load and split data
    with open(source_file, 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f.readlines() if line.strip()]
    
    chunk_size = len(sentences) // active_gpus
    chunks = [sentences[i:i + chunk_size] for i in range(0, len(sentences), chunk_size)]
    
    # Process chunks in parallel
    def process_chunk(chunk_data):
        chunk, gpu_id = chunk_data
        
        try:
            refiner = LangChainRobustLLMRefine(gpu_manager)
            
            # Build workflow
            workflow = StateGraph(OverallState, input=RobustState, output=RobustState)
            
            def generator_node(state):
                return refiner.llm_call_generator_robust(state, gpu_id)
            
            def evaluator_node(state):
                return refiner.llm_call_evaluator_robust(state, gpu_id)
            
            workflow.add_node("generator", generator_node)
            workflow.add_node("evaluator", evaluator_node)
            workflow.add_node("simulated_annealing", simulated_annealing_robust)
            
            workflow.add_edge(START, "generator")
            workflow.add_edge("generator", "evaluator")
            workflow.add_edge("evaluator", "simulated_annealing")
            
            workflow.add_conditional_edges(
                "simulated_annealing",
                lambda state: "continue" if state['i'] < state['n'] else "stop",
                {"continue": "generator", "stop": END}
            )
            
            app = workflow.compile()
            
            # Run processing
            initial_state = {
                "src": chunk,
                "temperature": temperature,
                "cooling_rate": cooling_rate,
                "n": max_iterations,
                "i": 0,
                "parse_failures": 0,
                "retry_count": [0] * len(chunk),
                "failed_sentences": []
            }
            
            result = app.invoke(initial_state)
            
            return {
                "gpu_id": gpu_id,
                "translations": result.get("tgt", []),
                "success": True,
                "parse_failures": result.get("parse_failures", 0),
                "total_sentences": len(chunk)
            }
            
        except Exception as e:
            print(f"❌ GPU {gpu_id} processing failed: {e}")
            return {
                "gpu_id": gpu_id,
                "translations": [f"[ERROR] {s}" for s in chunk],
                "success": False,
                "error": str(e)
            }
    
    # Execute in parallel
    chunk_data = [(chunk, i) for i, chunk in enumerate(chunks)]
    results = []
    
    with ThreadPoolExecutor(max_workers=active_gpus) as executor:
        future_to_gpu = {executor.submit(process_chunk, data): data[1] for data in chunk_data}
        
        for future in tqdm(as_completed(future_to_gpu), total=len(chunk_data), desc="Processing chunks"):
            gpu_id = future_to_gpu[future]
            try:
                result = future.result()
                results.append((gpu_id, result))
                print(f"✅ GPU {gpu_id}: {result['total_sentences']} sentences, {result.get('parse_failures', 0)} failures")
            except Exception as e:
                print(f"❌ GPU {gpu_id} failed: {e}")
    
    # Combine and save results
    results.sort(key=lambda x: x[0])  # Sort by GPU ID
    final_translations = []
    total_failures = 0
    
    for gpu_id, result in results:
        final_translations.extend(result['translations'])
        total_failures += result.get('parse_failures', 0)
    
    # Save output
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines([f"{t.strip()}\n" for t in final_translations])
    
    # Cleanup
    gpu_manager.cleanup()
    
    # Final statistics
    success_rate = (len(final_translations) - total_failures) / len(final_translations) * 100
    print(f"\n🎯 LangChain Robust Processing Completed!")
    print(f"   Total sentences: {len(final_translations)}")
    print(f"   Parse failures: {total_failures}")
    print(f"   Success rate: {success_rate:.1f}%")
    print(f"   Output saved to: {output_file}")


if __name__ == "__main__":
    # Example usage
    run_langchain_robust_processing(
        source_file="/path/to/data.en",
        output_file="/path/to/output.vi",
        num_gpus=8,
        max_iterations=6
    )
