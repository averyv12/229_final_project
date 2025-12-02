# Code comes from a Colab Notebook
!pip install -U torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 transformers accelerate datasets --quiet

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, concatenate_datasets
import torch
import random
import re
import json

model_id = "microsoft/Phi-3-mini-4k-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.generation_config._from_model_config = False

subjects = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]

dataset = [load_dataset("EleutherAI/hendrycks_math", subject, split="test") for subject in subjects]
dataset = concatenate_datasets(dataset)
print(len(dataset))
print(dataset[0])

level5 = [ex for ex in dataset if ex.get("level")=='Level 3']
print("Found level‑3 examples:", len(level5))
# For this snippet: sample 200 random
eval_examples = random.sample(level5, 200)
print(f"Exmaple of level 3: {eval_examples[0]}")

def make_prompt(question: str) -> str:
    # You can change prompt style; simple version:
    return f"<|user|>\n{question}. DO NOT show your work or intermediate steps, just return your final answer, which would be a numeric answer. Make sure your final answer is presented like this: '\\boxed{{ANSWER}}'. \n<|assistant|>\n"

def extract_boxed(text):
    start = text.find("\\boxed{")
    if start == -1:
        return None
    i = start + len("\\boxed{")
    depth = 1
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    return text[start + len("\\boxed{"):i-1].strip() if depth == 0 else None

def normalize(a):
    return a.strip().rstrip(".")

correct = 0
for idx, ex in enumerate(eval_examples):
    q = ex["problem"]
    solution = extract_boxed(ex["solution"])
    prompt = make_prompt(q)
    # if idx == 0:
    #   print(prompt)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=512, temperature=0.0, do_sample=False)

    generated_tokens = gen[0][inputs["input_ids"].shape[1]:]
    out = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    answer = extract_boxed(out)

    #print(idx, "Q:", q)
    print(f"{idx} -------")
    print("Generated:", answer)
    print("Solution :", solution)
    print("---------")

    if answer and solution and normalize(answer) == normalize(solution):
        correct += 1

print(correct)
accuracy = correct / 200
print(f"Accuracy: {accuracy}")
