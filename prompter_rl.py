#Copied over from a collab notebook
!pip install -q "transformers==4.46.3" "accelerate>=0.33.0" "datasets" "sentencepiece" "wandb"

import json
import random
from collections import defaultdict
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import torch
from torch import nn
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
import wandb

# =========================================================
# USER CONFIG
# =========================================================

MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"

# ---- Your local Q-net checkpoint paths (from the sidebar) ----
OLD_QNET_CKPT = "qnet_best_epochs10_seed1.pt"         # for OLD hints
NEW_QNET_CKPT = "qnet_newhints_epochs10_seed0.pt"     # for NEW hints

HIDDEN_DIM = 512          # must match training
N_EVAL_EXAMPLES = 200     # Level-3 problems to evaluate
TEST_SEED = 0             # random seed for test subset
ACTIONS_OF_INTEREST = [0, 3, 8]  # 3 hints to compare vs RL

SUBJECTS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]

# ---- W&B config ----
WANDB_ENTITY = "prompter_agent"      # <--- change to your entity
WANDB_PROJECT = "hint_eval_old_new"  # <--- any project name you like

wandb.login()

# =========================================================
# Hint templates (OLD and NEW)
# =========================================================

HINT_TEMPLATES = [
    "First, restate the problem in your own words and identify what is being asked.",
    "List all given quantities, define variables for unknowns, and write any obvious equations.",
    "Check if the problem can be broken into smaller subproblems; try to solve the first subproblem explicitly.",
    "Try a simpler or special case (for example, plug in small numbers) to see a pattern, then generalize.",
    "Look for relevant formulas or theorems (such as algebraic identities or combinatorics formulas) that might apply.",
    "Draw a diagram or table if it could help organize the information, and label all key quantities clearly.",
    "Track units or dimensions carefully to check whether each step of the solution makes sense.",
    "Check for symmetry or invariants in the problem that might simplify the reasoning.",
    "After each intermediate step, verify that your result is consistent with the original conditions.",
    "Once you reach an answer, plug it back into the original problem to verify that it is correct.",
]
NUM_HINTS = len(HINT_TEMPLATES)

NEW_HINT_TEMPLATES = [
    # 0 – Number theory / consecutive primes / special integer patterns
    "If the problem involves consecutive integers or primes and a condition on their sum or square, introduce an integer variable n and write the numbers in terms of n (for example, n, n+2, n+4 for consecutive odd primes). Express the given condition as an equation in n, solve it algebraically, and then test the resulting candidates to ensure they are actually prime and satisfy the original statement.",
    # 1 – Rational equations / algebraic expressions with denominators
    "When you see an equation with fractions or rational expressions, first determine which x-values are not allowed (where denominators are zero). Then multiply both sides by the least common denominator to clear fractions and simplify to a polynomial equation. Solve that equation, but finally plug each candidate back into the original expression to remove any extraneous solutions that make a denominator zero or break the original equation.",
    # 2 – Polynomial manipulation / factoring / Vieta-style reasoning
    "If the problem involves polynomials or symmetric expressions in the roots, look for factoring opportunities or substitutions (like letting t = x + 1/x or u = x^2). Try to rewrite the expression into products of simpler factors, or use identities and Vieta’s formulas to connect coefficients with sums and products of roots. Reduce the expression step by step before plugging in values.",
    # 3 – Multi-step algebra / word problems / systems of equations
    "For word problems or algebraic puzzles, start by clearly defining variables for every unknown quantity. Translate each sentence into an algebraic equation, forming a system if needed. Solve the system systematically (substitution, elimination, or matrix methods) and then interpret the solution in the context of the original story, discarding any solutions that are impossible (like negative lengths or counts when they must be positive).",
    # 4 – Geometry / diagrams / lengths and angles
    "For geometry problems, draw a clean, labeled diagram (even mentally) and mark all known side lengths and angles. Identify which theorems might apply—such as similar triangles, Pythagorean theorem, angle-chasing, or area formulas. If appropriate, introduce coordinates or auxiliary lines to simplify relationships, then carefully compute the requested length, angle, or area, checking that it is consistent with the geometric configuration.",
    # 5 – Counting / combinatorics / arrangements
    "For counting problems, first define precisely what objects you are counting. Decide whether a direct count, casework, permutations/combinations, or inclusion–exclusion is most appropriate. Break the problem into disjoint, manageable cases and make sure you are neither overcounting nor undercounting. Whenever possible, check your reasoning on a simpler or smaller version of the problem to see if the pattern matches your formula.",
    # 6 – Probability / random processes
    "For probability questions, express the probability as (number of favorable outcomes) divided by (number of possible outcomes), or as an expectation over simpler events. Clearly define the sample space and any relevant events. Use conditioning, the law of total probability, or complementary probabilities (1 − P(bad event)) when direct computation is messy. Simplify fractions at the end and confirm that your answer is between 0 and 1 and makes intuitive sense.",
    # 7 – Sequences, series, recurrences, or functional patterns
    "If the problem involves sequences, series, or functional patterns, write out the first several terms or evaluate the function at a few key points to detect structure. Look for arithmetic or geometric progressions, telescoping behavior, or a useful recurrence. Then derive a closed-form expression or a simplified sum, and verify it by checking it against the first few terms you computed.",
    # 8 – Inequalities / optimization / extremal values
    "When dealing with inequalities or optimization, first rewrite the expression so all terms are on one side if possible. Consider applying standard tools like AM–GM, Cauchy–Schwarz, or completing the square to bound the expression. Identify where equality could hold and check endpoints or critical points (from derivatives or algebraic reasoning). Make sure your final answer respects any domain restrictions given in the problem.",
    # 9 – Messy expressions / substitutions / simplifying structure
    "If the given expression looks messy, search for a substitution or reparameterization that simplifies it—such as letting t equal a repeated sub-expression, using symmetry (like x ↔ 1/x), or grouping terms cleverly. Simplify systematically at each step, canceling common factors and reducing complex fractions. Only at the very end plug in numerical values or compute the final numeric result, checking for arithmetic slips.",
]
NUM_NEW_HINTS = len(NEW_HINT_TEMPLATES)

# =========================================================
# Helpers: correctness, prompts, model loading
# =========================================================

def is_level_3(ex: Dict[str, Any]) -> bool:
    return ex.get("level") == "Level 3"

def extract_solution(text: str) -> Optional[str]:
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
    if depth != 0:
        return None
    return text[start + len("\\boxed{"): i - 1].strip()

def normalize(a: str) -> str:
    return a.strip().rstrip(".")

def is_correct(model_output: str, gold_solution: str) -> int:
    pred = extract_solution(model_output) or model_output
    gold = extract_solution(gold_solution) or gold_solution
    pred_n = normalize(pred)
    gold_n = normalize(gold)
    if pred_n == gold_n:
        return 1
    try:
        if float(pred_n) == float(gold_n):
            return 1
    except Exception:
        pass
    return 0

def get_subject(ex: Dict[str, Any]) -> str:
    return ex.get("subject") or ex.get("type") or "unknown"

def make_prompt_baseline(question: str) -> str:
    return (
        f"<|user|>\n{question}. "
        "DO NOT show your work or intermediate steps, just return your final answer. "
        "Make sure your final answer is presented like this: '\\boxed{ANSWER}'. \n"
        "<|assistant|>\n"
    )

def make_prompt_with_hint(question: str, hint_text: str) -> str:
    return (
        "<|user|>\n"
        "You are a helpful math tutor. Solve the following problem.\n\n"
        f"Hint: {hint_text}\n\n"
        f"{question}\n\n"
        "You may show your reasoning, but you MUST put the final answer "
        "in this format: '\\boxed{ANSWER}'.\n"
        "<|assistant|>\n"
    )

def load_phi3(model_id: str):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA available:", torch.cuda.get_device_name(0))
        torch_dtype = torch.float16
        device_map = "auto"
    else:
        device = torch.device("cpu")
        print("CUDA NOT available, using CPU.")
        torch_dtype = torch.float32
        device_map = None

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )
    model.generation_config.use_cache = False
    model.eval()
    if device.type == "cpu":
        model.to(device)
    print("Using device:", device)
    return tokenizer, model, device

@torch.no_grad()
def run_solver_with_hint(
    question: str,
    hint_text: str,
    *,
    model,
    tokenizer,
    device,
    max_new_tokens: int = 256,
) -> str:
    prompt = make_prompt_with_hint(question, hint_text)
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    gen = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        do_sample=False,
        use_cache=False,
    )
    generated_tokens = gen[0][enc["input_ids"].shape[1]:]
    out = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return out

@torch.no_grad()
def encode_question(
    question: str,
    model,
    tokenizer,
    device,
    max_length: int = 256,
) -> torch.Tensor:
    enc = tokenizer(
        question,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    ).to(device)
    out = model(**enc, output_hidden_states=True)
    last_hidden = out.hidden_states[-1]   # [1, T, d_model]
    emb = last_hidden.mean(dim=1)         # [1, d_model]
    return emb.squeeze(0).float()         # [d_model]

# =========================================================
# Q-network
# =========================================================

class MLPQNet(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, num_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

@torch.no_grad()
def next_action(qnet: nn.Module, state: torch.Tensor) -> int:
    q_values = qnet(state)
    return int(torch.argmax(q_values, dim=-1).item())

# =========================================================
# Evaluation helpers (parameterized by hint list)
# =========================================================

@torch.no_grad()
def evaluate_baseline(
    examples: List[Dict[str, Any]],
    model,
    tokenizer,
    device,
    max_examples: Optional[int] = None,
) -> Tuple[float, List[Dict[str, Any]]]:
    n = len(examples) if max_examples is None else min(len(examples), max_examples)
    correct = 0
    recs = []
    model.eval()
    for idx in range(n):
        ex = examples[idx]
        q = ex["problem"]
        gold_solution = ex["solution"]
        subject = get_subject(ex)

        prompt = make_prompt_baseline(q)
        enc = tokenizer(prompt, return_tensors="pt").to(device)
        gen = model.generate(
            **enc,
            max_new_tokens=256,
            temperature=0.0,
            do_sample=False,
            use_cache=False,
        )
        gen_tokens = gen[0][enc["input_ids"].shape[1]:]
        out = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        is_corr = int(is_correct(out, gold_solution))
        correct += is_corr

        recs.append(
            {"idx": idx, "subject": subject, "baseline_correct": is_corr}
        )
    acc = correct / float(n)
    print(f"Baseline accuracy on {n} examples: {acc:.3f}")
    return acc, recs

@torch.no_grad()
def evaluate_random_prompter(
    examples: List[Dict[str, Any]],
    model,
    tokenizer,
    device,
    hints: List[str],
    max_examples: Optional[int] = None,
    seed: Optional[int] = None,
) -> Tuple[float, List[Dict[str, Any]]]:
    if seed is not None:
        np.random.seed(seed)
    num_hints = len(hints)

    n = len(examples) if max_examples is None else min(len(examples), max_examples)
    correct = 0
    recs = []
    model.eval()
    for idx in range(n):
        ex = examples[idx]
        q = ex["problem"]
        gold_solution = ex["solution"]
        subject = get_subject(ex)

        hint_id = int(np.random.randint(num_hints))
        hint_text = hints[hint_id]
        out = run_solver_with_hint(
            question=q,
            hint_text=hint_text,
            model=model,
            tokenizer=tokenizer,
            device=device,
        )
        is_corr = int(is_correct(out, gold_solution))
        correct += is_corr
        recs.append(
            {"idx": idx, "subject": subject, "hint_id": hint_id, "random_correct": is_corr}
        )
    acc = correct / float(n)
    print(f"Random prompter accuracy on {n} examples: {acc:.3f}")
    return acc, recs

@torch.no_grad()
def evaluate_with_prompter(
    examples: List[Dict[str, Any]],
    qnet: nn.Module,
    model,
    tokenizer,
    device,
    hints: List[str],
    max_examples: Optional[int] = None,
) -> Tuple[float, List[Dict[str, Any]]]:
    n = len(examples) if max_examples is None else min(len(examples), max_examples)
    correct = 0
    recs = []
    qnet.eval()
    model.eval()
    for idx in range(n):
        ex = examples[idx]
        q = ex["problem"]
        gold_solution = ex["solution"]
        subject = get_subject(ex)

        emb = encode_question(q, model, tokenizer, device)
        state = emb.unsqueeze(0)
        hint_id = next_action(qnet, state)
        hint_text = hints[hint_id]

        out = run_solver_with_hint(
            question=q,
            hint_text=hint_text,
            model=model,
            tokenizer=tokenizer,
            device=device,
        )
        is_corr = int(is_correct(out, gold_solution))
        correct += is_corr
        recs.append(
            {"idx": idx, "subject": subject, "hint_id": hint_id, "rl_correct": is_corr}
        )
    acc = correct / float(n)
    print(f"RL prompter accuracy on {n} examples: {acc:.3f}")
    return acc, recs

@torch.no_grad()
def evaluate_single_action(
    examples: List[Dict[str, Any]],
    action_id: int,
    model,
    tokenizer,
    device,
    hints: List[str],
    max_examples: Optional[int] = None,
) -> Tuple[float, List[Dict[str, Any]]]:
    n = len(examples) if max_examples is None else min(len(examples), max_examples)
    correct = 0
    recs = []
    model.eval()
    hint_text = hints[action_id]
    for idx in range(n):
        ex = examples[idx]
        q = ex["problem"]
        gold_solution = ex["solution"]
        subject = get_subject(ex)
        out = run_solver_with_hint(
            question=q,
            hint_text=hint_text,
            model=model,
            tokenizer=tokenizer,
            device=device,
        )
        is_corr = int(is_correct(out, gold_solution))
        correct += is_corr
        recs.append(
            {"idx": idx, "subject": subject, "action_id": action_id, "single_correct": is_corr}
        )
    acc = correct / float(n)
    print(f"[Single-action {action_id}] accuracy: {acc:.3f}")
    return acc, recs

def compute_rl_conditional_acc_by_action(rl_recs: List[Dict[str, Any]]) -> Dict[int, float]:
    stats = defaultdict(lambda: {"hits": 0, "n": 0})
    for r in rl_recs:
        a = r["hint_id"]
        c = r["rl_correct"]
        stats[a]["hits"] += c
        stats[a]["n"] += 1
    acc_by_action = {}
    for a, s in stats.items():
        acc_by_action[a] = (s["hits"] / s["n"]) if s["n"] > 0 else float("nan")
    return acc_by_action

# =========================================================
# 1) Load Phi-3 and build Level-3 test subset
# =========================================================

tokenizer_eval, phi3_eval, device_eval = load_phi3(MODEL_ID)

print("Loading Hendrycks MATH test splits...")
dataset_list = [
    load_dataset("EleutherAI/hendrycks_math", subject, split="test")
    for subject in SUBJECTS
]
full_dataset = concatenate_datasets(dataset_list)
print("Total examples before Level-3 filter:", len(full_dataset))

level3_all = [ex for ex in full_dataset if is_level_3(ex)]
print("Total Level-3 examples:", len(level3_all))

n_total = len(level3_all)
n_test = min(N_EVAL_EXAMPLES, n_total)
random.seed(TEST_SEED)
random.shuffle(level3_all)
math_test_problems = level3_all[:n_test]
print(f"Using {n_test} random Level-3 examples as test set.")

# d_model from Phi-3
d_model = phi3_eval.config.hidden_size
print("Detected d_model from Phi-3:", d_model)

# =========================================================
# Start W&B run
# =========================================================

run = wandb.init(
    entity=WANDB_ENTITY,
    project=WANDB_PROJECT,
    config={
        "model_id": MODEL_ID,
        "hidden_dim": HIDDEN_DIM,
        "n_eval_examples": n_test,
        "test_seed": TEST_SEED,
        "old_qnet_ckpt": OLD_QNET_CKPT,
        "new_qnet_ckpt": NEW_QNET_CKPT,
        "actions_of_interest": ACTIONS_OF_INTEREST,
    },
    name="old_vs_new_hints_eval",
)

# =========================================================
# 2) First comparison: OLD HINTS
# =========================================================

print("\n================ OLD HINTS COMPARISON ================\n")

baseline_acc, baseline_recs = evaluate_baseline(
    math_test_problems,
    model=phi3_eval,
    tokenizer=tokenizer_eval,
    device=device_eval,
    max_examples=n_test,
)

qnet_old = MLPQNet(d_model, HIDDEN_DIM, NUM_HINTS).to(device_eval)
print(f"Loading OLD hints Q-net from '{OLD_QNET_CKPT}'")
qnet_old.load_state_dict(torch.load(OLD_QNET_CKPT, map_location=device_eval))
qnet_old.eval()

random_old_acc, random_old_recs = evaluate_random_prompter(
    math_test_problems,
    model=phi3_eval,
    tokenizer=tokenizer_eval,
    device=device_eval,
    hints=HINT_TEMPLATES,
    max_examples=n_test,
    seed=TEST_SEED,
)

rl_old_acc, rl_old_recs = evaluate_with_prompter(
    math_test_problems,
    qnet=qnet_old,
    model=phi3_eval,
    tokenizer=tokenizer_eval,
    device=device_eval,
    hints=HINT_TEMPLATES,
    max_examples=n_test,
)

print("\n=== OLD HINTS: overall accuracies ===")
print(f"Baseline (no hint):        {baseline_acc:.3f}")
print(f"Random prompter:           {random_old_acc:.3f}")
print(f"RL prompter (old hints):   {rl_old_acc:.3f}")
print(f"RL - baseline:             {rl_old_acc - baseline_acc:+.3f}")
print(f"RL - random:               {rl_old_acc - random_old_acc:+.3f}")

# Aggregate per-subject (old)
by_idx_old = {}
for r in baseline_recs:
    by_idx_old[r["idx"]] = {
        "subject": r["subject"],
        "baseline_correct": r["baseline_correct"],
    }
for r in random_old_recs:
    d = by_idx_old.setdefault(r["idx"], {})
    d["subject"] = d.get("subject", r["subject"])
    d["random_correct"] = r["random_correct"]
for r in rl_old_recs:
    d = by_idx_old.setdefault(r["idx"], {})
    d["subject"] = d.get("subject", r["subject"])
    d["rl_correct"] = r["rl_correct"]

stats_old = defaultdict(lambda: {"n": 0, "baseline": 0, "random": 0, "rl": 0})
for ex in by_idx_old.values():
    subj = ex.get("subject", "unknown")
    stats_old[subj]["n"] += 1
    stats_old[subj]["baseline"] += ex.get("baseline_correct", 0)
    stats_old[subj]["random"] += ex.get("random_correct", 0)
    stats_old[subj]["rl"] += ex.get("rl_correct", 0)

subjects_list_old = sorted(stats_old.keys())
baseline_subj_old = []
random_subj_old = []
rl_subj_old = []
Ns_old = []

for s in subjects_list_old:
    n = stats_old[s]["n"]
    Ns_old.append(n)
    baseline_subj_old.append(stats_old[s]["baseline"] / n)
    random_subj_old.append(stats_old[s]["random"] / n)
    rl_subj_old.append(stats_old[s]["rl"] / n)

print("\nSubject-wise stats (old hints):")
print("{:<30} {:>6} {:>10} {:>10} {:>10}".format("Subject", "N", "Base", "Random", "RL"))
subject_stats_old = {}
for s, n, ba, ra, qa in zip(subjects_list_old, Ns_old, baseline_subj_old, random_subj_old, rl_subj_old):
    print(f"{s:<30} {n:>6} {ba:>10.3f} {ra:>10.3f} {qa:>10.3f}")
    subject_stats_old[s] = {
        "n": int(n),
        "baseline": float(ba),
        "random": float(ra),
        "rl": float(qa),
    }

# Per-hint stats (old)
hint_freqs_old = np.zeros(NUM_HINTS, dtype=float)
hint_correct_counts_old = np.zeros(NUM_HINTS, dtype=float)
for rec in rl_old_recs:
    a = rec["hint_id"]
    c = rec["rl_correct"]
    hint_freqs_old[a] += 1
    hint_correct_counts_old[a] += c
hint_freqs_old /= n_test
hint_accs_old = np.zeros(NUM_HINTS, dtype=float)
hint_stats_old = {}
for a in range(NUM_HINTS):
    if hint_freqs_old[a] > 0:
        hint_accs_old[a] = hint_correct_counts_old[a] / (hint_freqs_old[a] * n_test)
    else:
        hint_accs_old[a] = np.nan
    hint_stats_old[a] = {
        "freq": float(hint_freqs_old[a]),
        "acc": None if np.isnan(hint_accs_old[a]) else float(hint_accs_old[a]),
    }

print("\nPer-hint stats (old hints):")
print("{:<5} {:>8} {:>12}  {}".format("ID", "Freq", "Acc", "Short hint text"))
for a in range(NUM_HINTS):
    short_text = HINT_TEMPLATES[a].split(".")[0][:60]
    acc_str = f"{hint_accs_old[a]:>12.3f}" if not np.isnan(hint_accs_old[a]) else "    n/a"
    print(f"{a:<5} {hint_freqs_old[a]:>8.3f} {acc_str}  {short_text}")

# 3 actions vs RL conditional (old)
single_action_results_old = {}
for a in ACTIONS_OF_INTEREST:
    acc_single, recs_single = evaluate_single_action(
        math_test_problems,
        action_id=a,
        model=phi3_eval,
        tokenizer=tokenizer_eval,
        device=device_eval,
        hints=HINT_TEMPLATES,
        max_examples=n_test,
    )
    single_action_results_old[a] = {
        "acc_single_full": acc_single,
    }

rl_cond_acc_old = compute_rl_conditional_acc_by_action(rl_old_recs)

print("\n=== OLD HINTS: 3 single actions vs RL conditional ===")
single_vs_rl_old = {}
for a in ACTIONS_OF_INTEREST:
    acc_single = single_action_results_old[a]["acc_single_full"]
    acc_rl_cond = rl_cond_acc_old.get(a, float("nan"))
    single_vs_rl_old[a] = {
        "single": float(acc_single),
        "rl_cond": None if np.isnan(acc_rl_cond) else float(acc_rl_cond),
    }
    print(f"Action {a} :: '{HINT_TEMPLATES[a]}'")
    print(f"  Single-action accuracy:         {acc_single:.3f}")
    if np.isnan(acc_rl_cond):
        print(f"  RL conditional accuracy:        n/a (never chosen)")
    else:
        print(f"  RL conditional accuracy (when chosen): {acc_rl_cond:.3f}")
    print()

# =========================================================
# 3) NEW HINTS COMPARISON
# =========================================================

print("\n================ NEW HINTS COMPARISON ================\n")
print("Baseline from earlier (no hints) is reused.\n")

qnet_new = MLPQNet(d_model, HIDDEN_DIM, NUM_NEW_HINTS).to(device_eval)
print(f"Loading NEW hints Q-net from '{NEW_QNET_CKPT}'")
qnet_new.load_state_dict(torch.load(NEW_QNET_CKPT, map_location=device_eval))
qnet_new.eval()

random_new_acc, random_new_recs = evaluate_random_prompter(
    math_test_problems,
    model=phi3_eval,
    tokenizer=tokenizer_eval,
    device=device_eval,
    hints=NEW_HINT_TEMPLATES,
    max_examples=n_test,
    seed=TEST_SEED,
)

rl_new_acc, rl_new_recs = evaluate_with_prompter(
    math_test_problems,
    qnet=qnet_new,
    model=phi3_eval,
    tokenizer=tokenizer_eval,
    device=device_eval,
    hints=NEW_HINT_TEMPLATES,
    max_examples=n_test,
)

print("\n=== NEW HINTS: overall accuracies ===")
print(f"Baseline (no hint):        {baseline_acc:.3f}  (same as before)")
print(f"Random prompter (new):     {random_new_acc:.3f}")
print(f"RL prompter (new hints):   {rl_new_acc:.3f}")
print(f"RL(new) - baseline:        {rl_new_acc - baseline_acc:+.3f}")
print(f"RL(new) - random(new):     {rl_new_acc - random_new_acc:+.3f}")

# Per-hint stats (new)
hint_freqs_new = np.zeros(NUM_NEW_HINTS, dtype=float)
hint_correct_counts_new = np.zeros(NUM_NEW_HINTS, dtype=float)
for rec in rl_new_recs:
    a = rec["hint_id"]
    c = rec["rl_correct"]
    hint_freqs_new[a] += 1
    hint_correct_counts_new[a] += c
hint_freqs_new /= n_test
hint_accs_new = np.zeros(NUM_NEW_HINTS, dtype=float)
hint_stats_new = {}
for a in range(NUM_NEW_HINTS):
    if hint_freqs_new[a] > 0:
        hint_accs_new[a] = hint_correct_counts_new[a] / (hint_freqs_new[a] * n_test)
    else:
        hint_accs_new[a] = np.nan
    hint_stats_new[a] = {
        "freq": float(hint_freqs_new[a]),
        "acc": None if np.isnan(hint_accs_new[a]) else float(hint_accs_new[a]),
    }

print("\nPer-hint stats (new hints):")
print("{:<5} {:>8} {:>12}  {}".format("ID", "Freq", "Acc", "Short hint text"))
for a in range(NUM_NEW_HINTS):
    short_text = NEW_HINT_TEMPLATES[a].split(".")[0][:60]
    acc_str = f"{hint_accs_new[a]:>12.3f}" if not np.isnan(hint_accs_new[a]) else "    n/a"
    print(f"{a:<5} {hint_freqs_new[a]:>8.3f} {acc_str}  {short_text}")


# =========================================================
# 4) METRIC SUMMARY (no W&B)
# =========================================================

print("\n=== OVERALL ACCURACIES (HINTS) ===")
print(f"Baseline (no hint):        {baseline_acc:.3f}")
print(f"Random prompter (hints):   {random_old_acc:.3f}")
print(f"RL prompter (hints):       {rl_old_acc:.3f}")
print(f"RL(hints) - baseline:      {rl_old_acc - baseline_acc:+.3f}")
print(f"RL(hints) - random:        {rl_old_acc - random_old_acc:+.3f}")

print("\n=== OVERALL ACCURACIES (NEW HINTS) ===")
print(f"Baseline (no hint):        {baseline_acc:.3f}")
print(f"Random prompter (new):     {random_new_acc:.3f}")
print(f"RL prompter (new hints):   {rl_new_acc:.3f}")
print(f"RL(new) - baseline:        {rl_new_acc - baseline_acc:+.3f}")
print(f"RL(new) - random(new):     {rl_new_acc - random_new_acc:+.3f}")

print("\n=== SUBJECT-WISE STATS (HINTS) ===")
print("{:<30} {:>6} {:>10} {:>10} {:>10}".format("Subject", "N", "Base", "Random", "RL"))
for s, n, ba, ra, qa in zip(subjects_list_old, Ns_old, baseline_subj_old, random_subj_old, rl_subj_old):
    print(f"{s:<30} {n:>6} {ba:>10.3f} {ra:>10.3f} {qa:>10.3f}")

print("\n=== PER-HINT STATS (HINTS) ===")
print("{:<5} {:>8} {:>12}  {}".format("ID", "Freq", "Acc", "Short hint text"))
for a in range(NUM_HINTS):
    short_text = HINT_TEMPLATES[a].split(".")[0][:60]
    acc_str = f"{hint_accs_old[a]:>12.3f}" if not np.isnan(hint_accs_old[a]) else "    n/a"
    print(f"{a:<5} {hint_freqs_old[a]:>8.3f} {acc_str}  {short_text}")

print("\n=== PER-HINT STATS (NEW HINTS) ===")
print("{:<5} {:>8} {:>12}  {}".format("ID", "Freq", "Acc", "Short hint text"))
for a in range(NUM_NEW_HINTS):
    short_text = NEW_HINT_TEMPLATES[a].split(".")[0][:60]
    acc_str = f"{hint_accs_new[a]:>12.3f}" if not np.isnan(hint_accs_new[a]) else "    n/a"
    print(f"{a:<5} {hint_freqs_new[a]:>8.3f} {acc_str}  {short_text}")

print("\n=== 3 SINGLE ACTIONS VS RL CONDITIONAL (HINTS) ===")
for a in ACTIONS_OF_INTEREST:
    acc_single = single_action_results_old[a]["acc_single_full"]
    acc_rl_cond = rl_cond_acc_old.get(a, float("nan"))
    print(f"Action {a} :: '{HINT_TEMPLATES[a]}'")
    print(f"  Single-action accuracy (all problems): {acc_single:.3f}")
    if np.isnan(acc_rl_cond):
        print("  RL conditional accuracy:              n/a (never chosen)")
    else:
        print(f"  RL conditional accuracy (when chosen): {acc_rl_cond:.3f}")
    print()

# =========================================================
# 5) PLOTTING (ALL PLOTS AT THE END) — PNGs saved LOCALLY
# =========================================================

plot_files = {}

# Hints overall (old Q-net)
methods_old = ["Baseline\n(no hint)", "Random\n(hints)", "RL\n(hints)"]
accs_old = [baseline_acc, random_old_acc, rl_old_acc]

plt.figure(figsize=(4, 4))
x = np.arange(len(methods_old))
bars = plt.bar(x, accs_old)
plt.xticks(x, methods_old)
plt.ylim(0, 1.0)
plt.ylabel("Accuracy")
plt.title("Hints: Overall Accuracy")
for bar, val in zip(bars, accs_old):
    h = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        h + 0.01,
        f"{val:.2f}",
        ha="center",
        va="bottom",
        fontsize=9,
    )
plt.tight_layout()
fname = "hints_overall_accuracy.png"
plt.savefig(fname, dpi=300)
plt.show()
plot_files["hints_overall"] = fname

# Hints per-subject
x_pos = np.arange(len(subjects_list_old))
width = 0.25
plt.figure(figsize=(8, 4))
b1 = plt.bar(x_pos - width, baseline_subj_old, width, label="Baseline")
b2 = plt.bar(x_pos,         random_subj_old,   width, label="Random (hints)")
b3 = plt.bar(x_pos + width, rl_subj_old,       width, label="RL (hints)")
plt.xticks(x_pos, subjects_list_old, rotation=45, ha="right")
plt.ylim(0, 1.0)
plt.ylabel("Accuracy")
plt.title("Hints: Accuracy by Subject")
plt.legend()
for bars_group in (b1, b2, b3):
    for bar in bars_group:
        h = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.01,
            f"{h:.2f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )
plt.tight_layout()
fname = "hints_accuracy_by_subject.png"
plt.savefig(fname, dpi=300)
plt.show()
plot_files["hints_by_subject"] = fname

# Hints RL hint usage
plt.figure(figsize=(7, 3))
x = np.arange(NUM_HINTS)
plt.bar(x, hint_freqs_old)
plt.xticks(x, [str(i) for i in range(NUM_HINTS)])
plt.ylabel("Fraction of test problems")
plt.xlabel("Hint ID")
plt.title("Hints: RL Hint Usage Frequency")
for i, f in enumerate(hint_freqs_old):
    plt.text(i, f + 0.005, f"{f:.2f}", ha="center", va="bottom", fontsize=7)
plt.ylim(0, max(hint_freqs_old) * 1.15 + 0.02)
plt.tight_layout()
fname = "hints_rl_hint_frequencies.png"
plt.savefig(fname, dpi=300)
plt.show()
plot_files["hints_hint_freq"] = fname

# Hints RL conditional acc
plt.figure(figsize=(7, 3))
plt.bar(x, hint_accs_old)
plt.xticks(x, [str(i) for i in range(NUM_HINTS)])
plt.ylabel("Accuracy on subset")
plt.xlabel("Hint ID")
plt.title("Hints: RL Conditional Accuracy by Hint")
for i, acc in enumerate(hint_accs_old):
    if not np.isnan(acc):
        plt.text(i, acc + 0.01, f"{acc:.2f}", ha="center", va="bottom", fontsize=7)
plt.ylim(0, 1.05)
plt.tight_layout()
fname = "hints_rl_conditional_accuracy.png"
plt.savefig(fname, dpi=300)
plt.show()
plot_files["hints_hint_acc"] = fname

# New hints overall
methods_new = ["Baseline\n(no hint)", "Random\n(new hints)", "RL\n(new hints)"]
accs_new = [baseline_acc, random_new_acc, rl_new_acc]

plt.figure(figsize=(4, 4))
x = np.arange(len(methods_new))
bars = plt.bar(x, accs_new)
plt.xticks(x, methods_new)
plt.ylim(0, 1.0)
plt.ylabel("Accuracy")
plt.title("New Hints: Overall Accuracy")
for bar, val in zip(bars, accs_new):
    h = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        h + 0.01,
        f"{val:.2f}",
        ha="center",
        va="bottom",
        fontsize=9,
    )
plt.tight_layout()
fname = "new_hints_overall_accuracy.png"
plt.savefig(fname, dpi=300)
plt.show()
plot_files["new_overall"] = fname

# New hints RL hint usage
plt.figure(figsize=(7, 3))
x = np.arange(NUM_NEW_HINTS)
plt.bar(x, hint_freqs_new)
plt.xticks(x, [str(i) for i in range(NUM_NEW_HINTS)])
plt.ylabel("Fraction of test problems")
plt.xlabel("Hint ID")
plt.title("New Hints: RL Hint Usage Frequency")
for i, f in enumerate(hint_freqs_new):
    plt.text(i, f + 0.005, f"{f:.2f}", ha="center", va="bottom", fontsize=7)
plt.ylim(0, max(hint_freqs_new) * 1.15 + 0.02)
plt.tight_layout()
fname = "new_hints_rl_hint_frequencies.png"
plt.savefig(fname, dpi=300)
plt.show()
plot_files["new_hint_freq"] = fname

# New hints RL conditional acc
plt.figure(figsize=(7, 3))
plt.bar(x, hint_accs_new)
plt.xticks(x, [str(i) for i in range(NUM_NEW_HINTS)])
plt.ylabel("Accuracy on subset")
plt.xlabel("Hint ID")
plt.title("New Hints: RL Conditional Accuracy by Hint")
for i, acc in enumerate(hint_accs_new):
    if not np.isnan(acc):
        plt.text(i, acc + 0.01, f"{acc:.2f}", ha="center", va="bottom", fontsize=7)
plt.ylim(0, 1.05)
plt.tight_layout()
fname = "new_hints_rl_conditional_accuracy.png"
plt.savefig(fname, dpi=300)
plt.show()
plot_files["new_hint_acc"] = fname

print("\nDone. PNGs saved in current Colab directory:")
for k, v in plot_files.items():
    print(f" - {k}: {v}")

# Save current test set to disk so it can be reused in another runtime
import json
from datetime import datetime

# Choose a filename you like
TEST_SAVE_PATH = "math_level3_testset_current.jsonl"

meta = {
    "created_at": datetime.utcnow().isoformat() + "Z",
    "num_examples": len(math_test_problems),
}

# Save the test examples (one JSON object per line)
with open(TEST_SAVE_PATH, "w") as f:
    for ex in math_test_problems:
        # ex is a HF row; convert to plain dict
        json.dump(dict(ex), f)
        f.write("\n")

print(f"Saved {len(math_test_problems)} examples to {TEST_SAVE_PATH}")

# (optional) save a tiny metadata file too
with open("math_level3_testset_current_meta.json", "w") as f:
    json.dump(meta, f, indent=2)
print("Saved metadata to math_level3_testset_current_meta.json")

#to reuse test set

import json

TEST_SAVE_PATH = "math_level3_testset_current.jsonl"

math_test_problems = []
with open(TEST_SAVE_PATH, "r") as f:
    for line in f:
        if line.strip():
            math_test_problems.append(json.loads(line))

print("Reloaded", len(math_test_problems), "examples")
