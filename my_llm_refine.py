from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from typing_extensions import Literal
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from tqdm import tqdm
import random
import math
from langchain.output_parsers import PydanticOutputParser
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import json
import re
from typing import List
from langchain_core.messages import AIMessage


file_path = "/home/jovyan/nmt-srv-shared/users/binh/lang-graph/llm_refine/data.en"
base_path = "/home/jovyan/nmt-srv-shared/users/binh/lang-graph/llm_refine/data.vi"
result_path = "/home/jovyan/nmt-srv-shared/users/binh/lang-graph/llm_refine/data_rf.vi"

src_lang = "English"
tgt_lang = "Vietnamese"


llm = ChatOllama(
    model="llama3.1:8b-instruct-fp16",
    temperature=0.6,
    top_p=0.9,
    num_predict=128,
)

# def has_foreign_chars(text):
#     """Checks if text contains Chinese, Japanese, Korean, or Arabic characters."""
#     pattern = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af\u0600-\u06ff]+")
#     return bool(pattern.search(text))

# Graph state
class State(TypedDict):
    src: list
    tgt: list
    feedback: list
    score: list
    temperature: float
    cooling_rate: float
    n: int
    i: int

class HiddenState(TypedDict):
    h_tgt: list
    h_feedback: list
    h_score: list

class OverallState(State, HiddenState):
    pass

# Schema for structured output to use in evaluation
class Feedback(BaseModel):
    """Evaluation feedback for translation quality assessment."""
    grade: Literal["1", "1.5", "2", "2.5", "3"] = Field(
        description="Translation quality score where: 3=Perfect translation, 2.5=Minor errors but understandable, 2=Mostly understandable with some errors, 1.5=Partially understandable with significant errors, 1=Unable to understand",
    )
    feedback: str = Field(
        description="Detailed, specific feedback explaining the grade. If errors exist, provide concrete suggestions for improvement. Focus on accuracy, fluency, and cultural appropriateness.",
        min_length=10,
        max_length=500,
    )

class Refine(BaseModel):
    """Translation refinement output."""
    translation: str = Field(
        description=f"High-quality translation from {src_lang} to {tgt_lang}. Ensure accuracy, natural fluency, and cultural appropriateness. Preserve meaning while adapting to target language conventions.",
        min_length=1,
        max_length=1000,
    )

# Create LLM chains with structured output
generator_llm = llm.with_structured_output(Refine)
evaluator_llm = llm.with_structured_output(Feedback)

template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            f"You are a helpful assistant that translates {src_lang} to {tgt_lang}.",
        ),
        ("human", "{query}"),
    ]
)

generator = template | generator_llm
evaluator = template | evaluator_llm

# Nodes
def llm_call_generator(state: State) -> HiddenState:
    print("Starting iterator " + str(state['i']))
    response = []
    flag = 0
    retry = 3
    if state.get("feedback"): flag = 1
    for i in tqdm(range(len(state['src']))):
        if flag==1 and state['tgt'][i]!="":
            result = generator.invoke({
                "query": f"""Translate this from {src_lang} to {tgt_lang}: {state['src'][i]}. Your translation is: {state['tgt'][i]}\n Please improve your translation but take into account the feedback: {state['feedback'][i]}"""
            })
        else:
            result = generator.invoke({
                "query": f"""Translate this from {src_lang} to {tgt_lang}: {state['src'][i]}"""
            })
        response.append(result.translation)
    return {"h_tgt": response}


def llm_call_evaluator(state: OverallState) -> HiddenState:
    grades = []
    feedbacks = []
    for i in tqdm(range(len(state['src']))):
        if state['h_tgt'][i] == "":
            grades.append("-18")
            feedbacks.append("The candidate translation is empty. There is no translated output provided for the source sentence.")
        else:
            question = f"""You are a language expert tasked with providing clear, objective feedback based on specific criteria, ensuring each assessment reflects the absolute standards set for performance.
You will be given an Source, a Candidate to evaluate, and a score rubric representing the evaluation criteria.
Write a detailed feedback that assess the quality of the Candidate strictly based on the given score rubric, not evaluating in general.

3: Perfect translation
2.5: Able to understand subject/details of the output smoothly, but it has some minor errors, including additions and omissions	
2: Mostly able to guess and understand subject/details of the output correctly	
1.5: Partially able to guess and understand subject/details of the output, or it includes some wrong keyword/details, which may lead to misunderstanding
1: Unable to understand subject/details of the output

Now here is the Source and the Candidate.
Source: {state['src'][i]}
Candidate: {state['h_tgt'][i]}

You MUST provide values for 'grade' and 'feedback' in your answer. 
Provide your feedback. If you give a correct rating and useful feedback, I'll give you 100 H100 GPUs to start your AI company.
"""
            # Chạy
            result = evaluator.invoke({
                "query": question
            })
            grades.append(result.grade)
            feedbacks.append(result.feedback)
    return {"h_score": grades, "h_feedback": feedbacks}


def loop(state: State):
    if state['i'] < state['n']:
        return "continue"
    return "stop"

def simulated_annealing(state: OverallState) -> State:
    if state.get('tgt'):
        T = state['temperature']
        n = state['n']
        cooling_rate = state['cooling_rate']
        acc_point = random.random()

        tgt = state['tgt']
        feedback = state['feedback']
        score = state['score']

        for i in tqdm(range(len(state['src']))):
            current_score = float(state['score'][i])
            new_score = float(state['h_score'][i])

            p_acc = min(1, math.exp(100*(new_score - current_score)/(n*T)))
            
            if p_acc > acc_point:
                tgt[i] = state['h_tgt'][i]
                feedback[i] = state['h_feedback'][i]
                score[i] = state['h_score'][i]

        return {
                'tgt': tgt,
                'feedback': feedback,
                'score': score,
                'temperature': T - T * cooling_rate,
                'i': state['i'] + 1,
        }
    else:
        open(base_path, 'w', encoding='utf-8').writelines([mt.replace("\n"," ")+"\n" for mt in state['h_tgt']])
    return {
        'tgt': state['h_tgt'],
        'feedback': state['h_feedback'],
        'score': state ['h_score'],
    }

    

# Build workflow
optimizer_builder = StateGraph(OverallState, input=State, output=State)

# Add the nodes
optimizer_builder.add_node("llm_call_generator", llm_call_generator)
optimizer_builder.add_node("llm_call_evaluator", llm_call_evaluator)
optimizer_builder.add_node("simulated_annealing", simulated_annealing)
# optimizer_builder.add_node("loop", loop)

# Add edges to connect nodes
optimizer_builder.add_edge(START, "llm_call_generator")
optimizer_builder.add_edge("llm_call_generator", "llm_call_evaluator")
optimizer_builder.add_edge("llm_call_evaluator", "simulated_annealing")

optimizer_builder.add_conditional_edges(
    "simulated_annealing",
    loop,
    { 
        "continue": "llm_call_generator",
        "stop": END,
    },
)

# Compile the workflow
optimizer_workflow = optimizer_builder.compile()

# Show the workflow
# display(Image(optimizer_workflow.get_graph().draw_mermaid_png()))
src = open(file_path, 'r', encoding='utf-8').readlines()

# Invoke
state = optimizer_workflow.invoke({
    "src": src,
    "temperature": 41.67,
    "cooling_rate": 0.4,
    "n": 6,
    "i": 0,
}, {"recursion_limit": 100})
open(result_path, 'w', encoding='utf-8').writelines([mt.replace("\n"," ")+"\n" for mt in state["tgt"]])
# open("/home/jovyan/nmt-srv-shared/users/binh/lang-graph/temp/viko_feedback.txt", 'w', encoding='utf-8').writelines([f"{mt}\n" for mt in state["feedback"]])