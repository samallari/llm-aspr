"""
Qwen Classifier for Acceptance Classification - Command Line Script

Usage:
    python qwen_eval.py <model_id> <dataset> <shots>

Example:
    python qwen_eval.py Qwen/Qwen2.5-1.5B-Instruct smallari/openreview-acceptance-classification-RAW 0 --output results.json
"""

import argparse
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm
import json


def format_data(entry):
    """Formats data into a readable string for the model."""
    submission_text = f"Title: {entry['title']}\nAbstract: {entry['abstract']}\n\nReviews:\n"
    for i, review in enumerate(entry['reviews']):
        submission_text += f"Review {i+1}:\n"
        submission_text += f"Summary: {review.get('summary', 'N/A')}\n"
        submission_text += f"Strengths: {review.get('strengths', 'N/A')}\n"
        submission_text += f"Weaknesses: {review.get('weaknesses', 'N/A')}\n"
        submission_text += f"Questions: {review.get('questions', 'N/A')}\n"
        submission_text += "\n"
    return submission_text.strip()


def create_prompt(entry, few_shot_entries=None):
    """
    Creates a structured chat prompt for the model.
    :param entry: A single entry from the dataset.
    :param few_shot_entries: Optional list of few-shot examples to include in the prompt.
    """
    messages = [
        {"role": "system", "content": "You are an expert reviewer. Predict whether the paper was accepted or rejected based on the following reviews. Output only 'Accept' or 'Reject'."}
    ]

    if few_shot_entries:
        for shot in few_shot_entries:
            user_content = format_data(shot)
            assistant_content = shot['decision']
            messages.append({"role": "user", "content": user_content})
            messages.append({"role": "assistant", "content": assistant_content})

    # Target entry
    user_content = format_data(entry)
    messages.append({"role": "user", "content": user_content})

    return messages


def evaluate_model(model_id, dataset, shots=0):
    """
    Evaluate the model on the entire dataset.

    :param model_id: Hugging Face model identifier
    :param dataset: Loaded dataset
    :param shots: Number of few-shot examples (default: 0 for zero-shot)
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
    filtered_indices = [i for i, entry in enumerate(data) if len(entry['reviews']) > 0]
    print(f"Filtered dataset size: {len(filtered_indices)} (removed {len(data) - len(filtered_indices)} entries)")

    # Select few-shot and test entries ensuring no overlap
    few_shot_entries = [data[filtered_indices[i]] for i in range(shots)]
    test_entries = [data[filtered_indices[i]] for i in range(shots, len(filtered_indices))]
    num_test_samples = len(test_entries)

    correct_predictions = 0
    all_results = []  # List to store all results for analysis

    print(f"Starting evaluation (Shots: {shots}, Test Samples: {num_test_samples})...")

    for idx, entry in tqdm(enumerate(test_entries), total=num_test_samples, desc="Evaluating samples"):
        # Create prompt
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
            max_new_tokens=10,
            do_sample=False  # Deterministic for reproducibility
        )

        # Decode output
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        # Determine ground truth (Label 1 = Accept, Label 0 = Reject)
        ground_truth = "Accept" if entry['label'] == 1 else "Reject"

        # Check if prediction is correct
        is_correct = ground_truth.lower() in response.lower()

        # Store results for analysis
        all_results.append({
            "model_response": response,
            "ground_truth": ground_truth,
            "is_correct": is_correct
        })

        if is_correct:
            correct_predictions += 1

    accuracy = correct_predictions / num_test_samples

    # Clean up to free memory
    del model
    del tokenizer
    torch.cuda.empty_cache()

    return shots, accuracy, all_results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen classifier on acceptance classification dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python qwen_eval.py Qwen/Qwen2.5-1.5B-Instruct smallari/openreview-acceptance-classification-RAW 0
  python qwen_eval.py Qwen/Qwen2.5-3B-Instruct smallari/openreview-acceptance-classification-RAW 3
        """
    )

    parser.add_argument("model_id", type=str, help="Hugging Face model ID (e.g., Qwen/Qwen2.5-1.5B-Instruct)")
    parser.add_argument("dataset", type=str, help="Hugging Face dataset ID (e.g., smallari/openreview-acceptance-classification-RAW)")
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
    print(f"Dataset loaded successfully. Raw split size: {len(dataset['raw'])}")

    # Run evaluation
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

