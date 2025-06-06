import argparse, json
from openai import OpenAI
import os
from typing import List, Tuple, Dict

reasoner_id = 'Gen-Verse/ReasonFlux-V2-32B/Template-Reasoner'
proposer_id = 'Gen-Verse/ReasonFlux-V2-32B/Template-Proposer'
proposer_url = 'http://localhost:30000/v1'
reasoner_url = 'http://localhost:30001/v1'
PROMPT_EXTRACT = """
Please analyze the problem below and first give **Problem Abstraction and Generalization**,
then based on the abstraction and initial analysis, give **Special Conditions and Applicability Analysis**.
Finally give **High‑Level Solution Strategy**.
""".strip()

PROMPT_SOLVE = """
Please follow the High-Level Steps below to solve the problem step by step,
and give the final answer within \\boxed{{}}.
""".strip()
template_proposer = OpenAI(api_key="EMPTY", base_url=proposer_url)
template_reasoner  = OpenAI(api_key="EMPTY", base_url=reasoner_url)


def extract_boxed_stack(text: str) -> List[str]:
    n = len(text)
    answers = []
    i = 0
    while i < n:
        if text.startswith(r'\boxed', i):
            i += 6  
            while i < n and text[i].isspace():
                i += 1
            if i >= n or text[i] != '{':
                continue
            i += 1  
            depth = 1
            start = i
            while i < n and depth:
                if text[i] == '{':
                    depth += 1
                elif text[i] == '}':
                    depth -= 1
                i += 1
            if depth == 0:
                answers.append(text[start:i-1].strip())
        else:
            i += 1
    if len(answers)>0:
        answer = answers[-1]
    else:
        answer = answers
    return answer
  
# Define the problem
# This is a sample problem, you can replace it with any problem you want to solve.
problem = """Every morning Aya goes for a $9$-kilometer-long walk and stops at a coffee shop afterwards. 
When she walks at a constant speed of $s$ kilometers per hour, the walk takes her 4 hours, 
including $t$ minutes spent in the coffee shop. When she walks $s+2$ kilometers per hour, the walk takes her 2 hours and 24 minutes,
including $t$ minutes spent in the coffee shop. Suppose Aya walks at $s+\\frac{1}{2}$ kilometers per hour. Find the number of minutes the walk takes her, 
including the $t$ minutes spent in the coffee shop.""".strip()


# First we extract the high-level thought template based on the proposer model
template_message = [
    {"role": "system", "content": PROMPT_EXTRACT},
    {"role": "user", "content": problem}
]
template_response = template_proposer.chat.completions.create(
    model=proposer_id,
    messages=template_message,
    temperature=0.7,
    top_p=0.8,  
    max_tokens=16384,
    presence_penalty=1.5,
    extra_body={"top_k": 20, "chat_template_kwargs": {"enable_thinking":False}}
)
template_content = template_response.choices[0].message.content
high_level_steps = template_content.split("**High-Level Solution Strategy**",1)[1]
print("High-Level Steps:\n", high_level_steps)
# Now we can use the reasoner model to solve the problem step by step
reasoner_message = [
    {"role": "system", "content": PROMPT_SOLVE},
    {"role": "user", "content": f"Problem: {problem}\nHigh-Level Steps: {high_level_steps}"}
]
reasoner_response = template_reasoner.chat.completions.create(
    model=reasoner_id,
    messages=reasoner_message,
    temperature=0.7,
    top_p=0.8,
    max_tokens=32768,
    presence_penalty=1.5,
    extra_body={"top_k": 20, "chat_template_kwargs": {"enable_thinking": True}}
)
reasoner_content = reasoner_response.choices[0].message.content

print("Reasoner Content:\n", reasoner_content)
print("Final Answer:\n", extract_boxed_stack(reasoner_content))



