import json
import pickle
import re
import os
import dspy
from dspy.teleprompt import BootstrapFewShot
from dspy.evaluate import Evaluate

# Configure DSPy to use Ollama with phi3-mini-4k-instruct
lm = dspy.LM("ollama_chat/phi3:mini", api_base="http://localhost:11434", api_key="", cache=False)
dspy.configure(lm=lm)

# Define the signature for solving math problems
class MathProblemSolver(dspy.Signature):
    """Solve the given math problem step by step. Provide a clear, detailed solution. You MUST return the answer in the format of \\boxed{...}"""
    problem = dspy.InputField(desc="The math problem to solve")
    reasoning = dspy.OutputField(desc="Step-by-step reasoning to the problem")
    solution = dspy.OutputField(desc="Make sure the final answer is presented as '\\boxed{{ANSWER}}'.")

# Create the module
class MathSolver(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_answer = dspy.ChainOfThought(MathProblemSolver)
    
    def forward(self, problem):
        return self.generate_answer(problem=problem)

# Load training examples from JSONL file (with gold_answer field)
def load_training_examples(filepath):
    examples = []
    with open(filepath, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            # Use hint_id 0 as the base example (or you can use all of them)
            if data.get('hint_id', 0) == 0:
                examples.append(dspy.Example(
                    problem=data['problem'],
                    solution=data['gold_answer']
                ).with_inputs('problem'))
    return examples

# Load test examples from JSONL file (with solution field)
def load_test_examples(filepath):
    examples = []
    with open(filepath, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            examples.append(dspy.Example(
                problem=data['problem'],
                solution=data['solution']
            ).with_inputs('problem'))
    return examples

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
    if a is None:
        return None
    return a.strip().rstrip(".")

# Define a metric to evaluate the LLM response
def math_accuracy_metric(gold, pred, trace=None):
    """Evaluate if the predicted solution matches the gold answer.
    Returns:
    - 1.0 if the answer matches
    - 0.0 otherwise
    """
    gold_answer = gold.solution
    pred_answer = pred.solution
    
    # Extract boxed answers from both (common format: \boxed{...})
    
    gold_boxed = normalize(extract_boxed(gold_answer))
    pred_boxed = normalize(extract_boxed(pred_answer))
    
    print(f"Gold boxed: {gold_boxed}")
    print(f"Pred boxed: {pred_boxed}")
    
    if gold_boxed and pred_boxed:
        # Normalize: remove spaces, convert to lowercase for comparison
        gold_final = gold_boxed.strip().lower().replace(' ', '')
        pred_final = pred_boxed.strip().lower().replace(' ', '')
        if gold_final == pred_final:
            return 1.0  # Correct answer
    
    return 0.0  # No \boxed found in prediction

def main():
    # Load training examples
    print("Loading training examples...")
    trainset = load_training_examples('240examples_10each_lvl3.jsonl')
    print(f"Loaded {len(trainset)} training examples")
    
    # Load test examples
    print("Loading test examples...")
    testset = load_test_examples('math_level3_testset_current.jsonl')
    print(f"Loaded {len(testset)} test examples")
    
    # Initialize the solver
    solver = MathSolver()
    
    # # Test on a single example first (from test set)
    # print("\n=== Testing on a single example ===")
    # test_example = testset[0]
    # print(f"Problem: {test_example.problem[:100]}...")
    # result = solver(problem=test_example.problem)
    # print(f"\nGenerated Solution:\n{result.solution}")
    # print(f"\nGold Solution:\n{test_example.solution}")
    
    # Set up evaluation on test set
    print("\n=== Setting up evaluation on test set ===")
    evaluate = Evaluate(
        devset=testset,  # Use test set for evaluation
        metric=math_accuracy_metric,
        num_threads=1,
        display_progress=True
    )
    
    # # Evaluate baseline
    # print("\n=== Evaluating baseline ===")
    # baseline_score = evaluate(solver)
    # print(f"Baseline accuracy: {baseline_score}")
    
    # Set up prompt optimizer using BootstrapFewShot
    print("\n=== Setting up prompt optimizer (BootstrapFewShot) ===")
    optimizer = BootstrapFewShot(
        metric=math_accuracy_metric,
        max_bootstrapped_demos=4,
        max_labeled_demos=8,
    )
    
    # Check if optimized solver already exists
    optimized_solver_path = 'optimized_solver_BootstrapFewShot_model.json'
    optimized_solver = None
    
    if os.path.exists(optimized_solver_path):
        print(f"\n=== Loading existing optimized solver from {optimized_solver_path} ===")
        optimized_solver = MathSolver()
        optimized_solver.load(optimized_solver_path)
        print("Loaded optimized solver successfully!")
    else:
        # Optimize the solver on a subset of training data
        print("\n=== Optimizing prompts on subset of training set ===")
        train_subset = trainset[:20]  # Use first 20 examples for BootstrapFewShot
        print(f"Using {len(train_subset)} examples for optimization")
        optimized_solver = optimizer.compile(
            student=solver,
            trainset=train_subset,
        )
        
        # Save the optimized solver
        print(f"\n=== Saving optimized solver to {optimized_solver_path} ===")
        optimized_solver.save(optimized_solver_path)
        print("Optimized solver saved successfully!")
    
    # Evaluate optimized solver
    print("\n=== Evaluating optimized solver ===")
    optimized_score = evaluate(optimized_solver)
    print(f"Optimized accuracy: {optimized_score}")

if __name__ == "__main__":
    main()