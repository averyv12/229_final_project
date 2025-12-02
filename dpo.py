# Code is copied over from a Colab Notebook
!pip install -U torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 transformers accelerate datasets --quiet

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, concatenate_datasets
import torch
import random
import re
import json

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

!pip install torch transformers datasets accelerate bitsandbytes

from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import json
from torch.cuda.amp import autocast, GradScaler
from torch.nn.utils.rnn import pad_sequence
import bitsandbytes as bnb

# Eval_examples is a custom made DPO dataset
dataset = load_dataset("json", data_files={"train": "/content/eval_examples.jsonl"})
dataset = dataset['train']

output_model_name = "microsoft/Phi-3-mini-4k-instruct"
output_tokenizer = AutoTokenizer.from_pretrained(output_model_name, load_in_8bit=True)

if output_tokenizer.pad_token is None:
    output_tokenizer.pad_token = output_tokenizer.eos_token
    output_tokenizer.pad_token_id = output_tokenizer.eos_token_id

max_length = 128

def tokenize_dpo(example):
    prompt_ids = output_tokenizer(example["prompt"], truncation=True, max_length=max_length)["input_ids"]
    chosen_ids = output_tokenizer(example["chosen"], truncation=True, max_length=max_length)["input_ids"]
    rejected_ids = output_tokenizer(example["rejected"], truncation=True, max_length=max_length)["input_ids"]
    return {
        "prompt_ids": prompt_ids,
        "chosen_ids": chosen_ids,
        "rejected_ids": rejected_ids
    }

tokenized_dataset = dataset.map(tokenize_dpo, remove_columns=["prompt","chosen","rejected"])

def dpo_collate_fn(batch):
    prompt_ids   = [torch.tensor(b["prompt_ids"], dtype=torch.long) for b in batch]
    chosen_ids   = [torch.tensor(b["chosen_ids"], dtype=torch.long) for b in batch]
    rejected_ids = [torch.tensor(b["rejected_ids"], dtype=torch.long) for b in batch]

    pad_id = output_tokenizer.pad_token_id

    prompt_ids   = pad_sequence(prompt_ids,   batch_first=True, padding_value=pad_id)
    chosen_ids   = pad_sequence(chosen_ids,   batch_first=True, padding_value=pad_id)
    rejected_ids = pad_sequence(rejected_ids, batch_first=True, padding_value=pad_id)

    return {
        "prompt_ids": prompt_ids,
        "chosen_ids": chosen_ids,
        "rejected_ids": rejected_ids,
    }

output_model = AutoModelForCausalLM.from_pretrained(output_model_name)
output_model.resize_token_embeddings(len(output_tokenizer))

device = "cuda" if torch.cuda.is_available() else "cpu"
output_model.to(device)
output_model.train()

def dpo_loss(chosen_logps, rejected_logps, ref_chosen_logps, ref_rejected_logps, beta=0.7):
    """
    chosen_logps:        log pθ(y⁺ | x)
    rejected_logps:      log pθ(y⁻ | x)
    ref_chosen_logps:    log πref(y⁺ | x)
    ref_rejected_logps:  log πref(y⁻ | x)
    """
    logits_diff = (chosen_logps - rejected_logps) - (ref_chosen_logps - ref_rejected_logps)
    loss = -F.logsigmoid(beta * logits_diff).mean()
    return loss

def compute_sequence_logprob(logits, labels, ignore_index=-100):
    """
    logits: (B, T, V)
    labels: (B, T)
    Returns per-example log probability of entire sequence, ignoring padding (-100)
    """
    B, T, V = logits.size()
    mask = labels.ne(ignore_index)
    safe_labels = labels.clone()
    safe_labels[~mask] = 0
    log_probs = torch.log_softmax(logits, dim=-1)
    token_logprobs = torch.gather(log_probs, 2, safe_labels.unsqueeze(-1)).squeeze(-1)
    token_logprobs = token_logprobs * mask.float()
    return token_logprobs.sum(dim=1)

def train_dpo(model_name,tokenized_dataset,epochs=3,batch_size=1,lr=1e-5,beta=0.7,device="cuda"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.gradient_checkpointing_enable()
    model.to(device)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.resize_token_embeddings(len(tokenizer))

    # ---- reference model (frozen copy) ----
    ref_model = AutoModelForCausalLM.from_pretrained(model_name)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False
    ref_model.to("cpu")

    dataloader = DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=True, collate_fn=dpo_collate_fn)
    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=lr)
    scaler = GradScaler()

    for epoch in range(epochs):
        for i, batch in enumerate(dataloader):

            optimizer.zero_grad(set_to_none=True)

            # unpack batch
            chosen_ids   = batch["chosen_ids"].to(device)
            rejected_ids = batch["rejected_ids"].to(device)


            # labels for LM: replace pad with -100
            chosen_labels   = chosen_ids.clone()
            rejected_labels = rejected_ids.clone()

            chosen_labels[chosen_labels == tokenizer.pad_token_id] = -100
            rejected_labels[rejected_labels == tokenizer.pad_token_id] = -100

            # ---- forward pass θ ----
            # model_chosen = model(input_ids=chosen_ids, labels=chosen_labels)
            # model_rejected = model(input_ids=rejected_ids, labels=rejected_labels)

            # chosen_logps   = compute_sequence_logprob(model_chosen.logits, chosen_labels)
            # rejected_logps = compute_sequence_logprob(model_rejected.logits, rejected_labels)

            with autocast():  # mixed precision
                model_chosen   = model(input_ids=chosen_ids, labels=chosen_labels)
                model_rejected = model(input_ids=rejected_ids, labels=rejected_labels)

                chosen_logps   = compute_sequence_logprob(model_chosen.logits, chosen_labels)
                rejected_logps = compute_sequence_logprob(model_rejected.logits, rejected_labels)

            # ---- forward pass reference ----
            with torch.no_grad():
                # move inputs to CPU for the ref model
                ref_chosen = ref_model(input_ids=chosen_ids.cpu(), labels=chosen_labels.cpu())
                ref_rejected = ref_model(input_ids=rejected_ids.cpu(), labels=rejected_labels.cpu())

                # compute logprobs on CPU and move result back to GPU
                ref_chosen_logps_cpu   = compute_sequence_logprob(ref_chosen.logits, chosen_labels.cpu())
                ref_rejected_logps_cpu = compute_sequence_logprob(ref_rejected.logits, rejected_labels.cpu())
            
            ref_chosen_logps = ref_chosen_logps_cpu.detach().to(device)
            ref_rejected_logps = ref_rejected_logps_cpu.detach().to(device)

            # ---- DPO loss ----
            loss = dpo_loss(chosen_logps, rejected_logps, ref_chosen_logps, ref_rejected_logps, beta=beta)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            loss_item = loss.item()


            del model_chosen, model_rejected, chosen_logps, rejected_logps
            del ref_chosen, ref_rejected, ref_chosen_logps, ref_rejected_logps
            del ref_chosen_logps_cpu, ref_rejected_logps_cpu
            del chosen_ids, rejected_ids, chosen_labels, rejected_labels
            del loss
            torch.cuda.empty_cache()

            if i % 10 == 0:
              print(f"Batch {i+1} Loss: {loss_item:.8f}")

        print(f"Epoch {epoch+1} loss: {loss_item:.8f}")

    return model

output_model = train_dpo(output_model_name, tokenized_dataset)
output_model.save_pretrained("phi-3-mini-dpo-math")
output_tokenizer.save_pretrained("phi-3-mini-dpo-math")

# Uncomment below if you want to load in model
# output_model_path = "phi-3-mini-dpo-math"
# output_tokenizer = AutoTokenizer.from_pretrained(model_path)
# output_model = AutoModelForCausalLM.from_pretrained(model_path)

output_model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
output_model.to(device)

def better_prompt(prompt):
  new_prompt = f"Rewrite the following prompt to be clearer, more helpful, and more structured, but without solving it or adding any steps toward a solution: {prompt}. Output only the improved prompt, nothing else."
  inputs = output_tokenizer(new_prompt, return_tensors="pt").to(device)

  output_ids = output_model.generate(
      **inputs,
      max_new_tokens=250,
      temperature=0.7,
      top_p=0.9,
      do_sample=True,
      pad_token_id=output_tokenizer.eos_token_id
  )

  response = output_tokenizer.decode(output_ids[0], skip_special_tokens=True)
  return response

reg_model_id = "microsoft/Phi-3-mini-4k-instruct"
reg_tokenizer = AutoTokenizer.from_pretrained(reg_model_id)
reg_model = AutoModelForCausalLM.from_pretrained(
    reg_model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)
reg_model.generation_config._from_model_config = False

dpo_correct = 0
for idx, ex in enumerate(eval_examples):
    q = ex["problem"]
    new_q = better_prompt(q)
    solution = extract_boxed(ex["solution"])
    prompt = make_prompt(new_q)
    inputs = reg_tokenizer(prompt, return_tensors="pt").to(reg_model.device)

    with torch.no_grad():
        gen = reg_model.generate(**inputs, max_new_tokens=256, temperature=0.0, do_sample=False)

    generated_tokens = gen[0][inputs["input_ids"].shape[1]:]
    out = reg_tokenizer.decode(generated_tokens, skip_special_tokens=True)
    answer = extract_boxed(out)

    print(f"{idx} -------")
    if idx % 10 == 0:
      print(f"Old prompt: {q} \n")
      print(f"New prompt: {new_q} \n")
    print("Generated:", answer)
    print("Solution :", solution)
    print("---------")

    if answer and solution and normalize(answer) == normalize(solution):
        dpo_correct += 1

dpo_accuracy = (dpo_correct) / 200
print(f"Accuracy: {dpo_accuracy}")
