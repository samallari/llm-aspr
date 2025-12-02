"""
Qwen Classifier for Acceptance Classification - Command Line Script
Updated pipeline based on notebooks/qwen_classifier.ipynb

Usage:
    python qwen_eval.py <model_id> <dataset> <shots> --output <output_file>

Example:
    python qwen_eval.py Qwen/Qwen2.5-1.5B-Instruct smallari/openreview-acceptance-classification-RAW 0 --output results.json
"""

import argparse
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm
import json
import os

def format_data(entry):
    """Formats data into a readable string for the model."""
    submission_text = f"Title: {entry['title']}\nAbstract: {entry['abstract']}\n\nReviews:\n"
    for i, review in enumerate(entry['reviews']):
        submission_text += f"Review {i+1}:\n"
        submission_text += f"Summary: {review.get('summary', 'N/A')}\n"
        submission_text += f"Strengths: {review.get('strengths', 'N/A')}\n"
        submission_text += f"Weaknesses: {review.get('weaknesses', 'N/A')}\n"
        submission_text += f"Questions: {review.get('questions', 'N/A')}\n"
        # submission_text += f"Rating: {review.get('rating', 'N/A')}\n"
        # submission_text += f"Confidence: {review.get('confidence', 'N/A')}\n"
        submission_text += "\n"
    return submission_text.strip()

def create_prompt(entry, few_shot_entries=None):
    """
    Creates a structured chat prompt for the model.
    :param entry: A single record from the dataset.
    :param few_shot_entries: Optional list of example pairs, where each pair is a list of message dicts.
    """
    messages = [
        {"role": "system", "content": "You are an expert Area Chair for a computer science conference. Your goal is to determine if a paper should be accepted or rejected based on the provided peer reviews.\n\nYou must think step-by-step to reach a conclusion. Output your thought process inside <reasoning> tags, following this specific structure:\n\n1. **Review Analysis**: Briefly list the scores (if available) and the general sentiment of each reviewer (e.g., Reviewer 1: Weak Accept, Reviewer 2: Strong Reject).\n2. **Key Strengths**: Identify the strongest points agreed upon by the reviewers.\n3. **Critical Weaknesses & Severity**: Identify the weaknesses. Crucially, determine if these are \"fatal flaws\" (methodological errors, lack of novelty) or \"fixable issues\" (typos, clarity).\n4. **Conflict Resolution**: If reviewers disagree, evaluate which argument is more grounded in the paper's evidence. Discard vague or unsubstantiated reviewer claims.\n5. **Final Verdict Formulation**: Weigh the technical contribution against the severity of the flaws.\n\nAfter your analysis, output your final decision inside <decision> tags. The valid values are ACCEPT or REJECT.\n\nExample Output Structure:\n<reasoning>\n[Your step-by-step analysis here]\n</reasoning>\n<decision>\nACCEPT\n</decision>"}
    ]

    # Add few-shot examples if provided
    if few_shot_entries:
        for example_pair in few_shot_entries:
            # example_pair is a list of message dicts [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
            for message in example_pair:
                messages.append(message)
    
    # Target entry
    user_content = format_data(entry)
    messages.append({"role": "user", "content": user_content})

    return messages

def evaluate_model(model_id, dataset, shots=0, few_shot_prompt_path="three_shot_prompt.json"):
    """
    Evaluate the model on the dataset.
    
    :param model_id: Hugging Face model identifier
    :param dataset: Loaded dataset
    :param shots: Number of few-shot examples
    :param few_shot_prompt_path: Path to the JSON file containing few-shot examples
    :return: Tuple of (shots, accuracy, all_results)
    """
    print(f"Loading model: {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto"
    )

    # Access the 'raw' split
    data = dataset['raw']

    # Filter out entries with empty reviews
    print("Filtering dataset for non-empty reviews...")
    # Using the same filtering logic as the notebook
    filtered_dataset = dataset.filter(lambda x: len(x['reviews']) > 0)
    
    # Remove first three entries as they are used for few-shot examples (as per notebook logic)
    # This prevents data leakage if the few-shot prompt uses the first 3 examples
    filtered_dataset = filtered_dataset.filter(lambda x, idx: idx > 2, with_indices=True)
    
    test_entries = filtered_dataset['raw']
    num_test_samples = len(test_entries)
    
    print(f"Filtered dataset size: {num_test_samples}")

    # Load few-shot examples if needed
    few_shot_entries = None
    if shots > 0:
        if os.path.exists(few_shot_prompt_path):
            print(f"Loading few-shot examples from {few_shot_prompt_path}...")
            with open(few_shot_prompt_path, "r") as f:
                full_few_shot_prompt = json.load(f)
                # Use the first 'shots' examples
                few_shot_entries = full_few_shot_prompt[:shots]
        else:
            print(f"Warning: Few-shot prompt file not found at {few_shot_prompt_path}. Proceeding with zero-shot.")
            shots = 0

    correct_predictions = 0
    all_results = [] # List to store all results for analysis

    print(f"Starting evaluation (Shots: {shots}, Test Samples: {num_test_samples})...")

    for idx, entry in tqdm(enumerate(test_entries), total=num_test_samples, desc="Evaluating samples"):
        # Create prompt with pre-made few-shot examples
        messages = create_prompt(entry, few_shot_entries)

        # Prepare input
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        # Generate response
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=8192, 
            do_sample=False  # Deterministic for reproducibility (same as temperature = 0)
        )

        # Decode output
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        # Determine ground truth (Label 1 = Accept, Label 0 = Reject)
        ground_truth = "Accept" if entry['label'] == 1 else "Reject"

        # Check prediction
        # We check if the ground truth keyword is present in the response (case-insensitive)
        # Note: The prompt asks to output <decision>ACCEPT</decision> or similar, but we check loosely for now
        # or we can parse the <decision> tag if we want to be stricter.
        # The notebook used: ground_truth.lower() in response.lower()
        is_correct = ground_truth.lower() in response.lower()

        # Store results for analysis
        all_results.append({
            "model_response": response,
            "ground_truth": ground_truth,
            "is_correct": is_correct
        })

        if is_correct:
            correct_predictions += 1

    accuracy = correct_predictions / num_test_samples if num_test_samples > 0 else 0

    # Clean up to free memory
    del model
    del tokenizer
    torch.cuda.empty_cache()

    return shots, accuracy, all_results

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen classifier on acceptance classification dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("model_id", type=str, help="Hugging Face model ID (e.g., Qwen/Qwen2.5-1.5B-Instruct)")
    parser.add_argument("dataset", type=str, help="Hugging Face dataset ID")
    parser.add_argument("shots", type=int, help="Number of few-shot examples (0 for zero-shot)")
    parser.add_argument("--output", type=str, default=None, help="Optional path to save results as JSON")

    args = parser.parse_args()

    # Check GPU availability
    print(f"GPU Available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("Warning: No GPU detected. Evaluation may be very slow.")

    # Load dataset
    print(f"Loading dataset: {args.dataset}...")
    dataset = load_dataset(args.dataset)
    print(f"Dataset loaded successfully.")

    # Run evaluation
    # Assuming three_shot_prompt.json is in notebooks/ relative to where script is run (repo root)
    shots, accuracy, results = evaluate_model(args.model_id, dataset, shots=args.shots)

    # Print results
    print("\n" + "="*50)
    print(f"MODEL: {args.model_id}")
    print(f"SHOTS: {shots}")
    print(f"ACCURACY: {accuracy:.2%}")
    print(f"CORRECT: {sum(r['is_correct'] for r in results)}/{len(results)}")
    print("="*50)

    # Save results if output path provided
    if args.output:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        
        output_data = {
            "model_id": args.model_id,
            "dataset": args.dataset,
            "shots": shots,
            "accuracy": accuracy,
            "total_samples": len(results),
            "correct_predictions": sum(r['is_correct'] for r in results),
            "results": results
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output}")

if __name__ == "__main__":
    main()
