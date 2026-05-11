#!/usr/bin/env python3
"""
eval.py
Runs lm-evaluation-harness on a given model against
the professional_law MMLU category.

Usage:
    python3 eval.py --model_path meta-llama/Llama-3.1-8B \
                    --model_name baseline_raw \
                    --output_dir ../results

Runs three times:
    python3 eval.py --model_path meta-llama/Llama-3.1-8B
    python3 eval.py --model_path meta-llama/Llama-3.1-8B-Instruct
    python3 eval.py --model_path /mnt/data/outputs/finetuned-llama
"""

import argparse
import json
import os
import sys
import subprocess
from datetime import datetime
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a model on MMLU professional_law"
    )

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="HuggingFace model ID or local path to model"
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Short name for output files e.g. baseline_raw, baseline_instruct, finetuned"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="../results",
        help="Directory to save evaluation results"
    )

    parser.add_argument(
        "--num_fewshot",
        type=int,
        default=5,
        help="Number of few-shot examples (default 5 — standard for MMLU)"
    )

    parser.add_argument(
        "--batch_size",
        type=str,
        default="auto",
        help="Batch size for evaluation (auto lets lm-eval decide)"
    )

    return parser.parse_args()


def run_evaluation(model_path, model_name, output_dir, num_fewshot, batch_size):
    """
    Runs lm-evaluation-harness via subprocess.
    We call it as a subprocess rather than importing it directly
    because lm-eval manages its own process and GPU memory cleanly
    this way — avoids memory conflicts with other imports.
    """

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Build output filename from model name and timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if model_name is None:
        # Auto-generate name from model path
        model_name = model_path.replace("/", "_").replace("-", "_")
    output_file = os.path.join(output_dir, f"eval_{model_name}_{timestamp}.json")

    print("\n" + "="*60)
    print(f"EVALUATING MODEL: {model_path}")
    print(f"Task:             mmlu_professional_law")
    print(f"Few-shot:         {num_fewshot}")
    print(f"Output:           {output_file}")
    print("="*60 + "\n")

    # Build the lm-eval command
    # This is exactly what you'd run manually on the command line
    cmd = [
        "lm_eval",
        "--model", "hf",                           # use HuggingFace model
        "--model_args", f"pretrained={model_path}",# which model to load
        "--tasks", "mmlu_professional_law",         # which benchmark to run
        "--num_fewshot", str(num_fewshot),          # 5-shot is MMLU standard
        "--batch_size", batch_size,                 # how many examples per batch
        "--output_path", output_file,               # where to save results
        "--log_samples",                            # save individual question results
    ]

    print(f"Running command:\n{' '.join(cmd)}\n")

    # Run lm-eval and stream output to terminal in real time
    # This lets you watch progress as it evaluates each question
    result = subprocess.run(
        cmd,
        text=True
    )

    if result.returncode != 0:
        print(f"\n[FAIL] Evaluation failed with return code {result.returncode}")
        sys.exit(1)

    print(f"\n[PASS] Evaluation complete")
    print(f"Results saved to: {output_file}")

    return output_file


def parse_results(output_file):
    """
    Reads the lm-eval JSON output and extracts the key metrics.

    lm-eval saves a JSON file with this structure:
    {
      "results": {
        "mmlu_professional_law": {
          "acc,none": 0.634,
          "acc_stderr,none": 0.013
        }
      }
    }

    acc = accuracy (percentage correct as a decimal)
    acc_stderr = standard error (how reliable the measurement is)
    """

    # lm-eval sometimes saves to a subdirectory — find the actual file
    output_path = Path(output_file)

    if output_path.is_dir():
        # Search for JSON files in the directory
        json_files = list(output_path.glob("**/*.json"))
        if not json_files:
            print(f"[WARN] No JSON results found in {output_file}")
            return None
        output_path = json_files[0]

    if not output_path.exists():
        # Try finding it as a directory
        possible_dir = Path(str(output_path).replace(".json", ""))
        if possible_dir.exists():
            json_files = list(possible_dir.glob("**/*.json"))
            if json_files:
                output_path = json_files[0]

    try:
        with open(output_path, "r") as f:
            data = json.load(f)

        results = data.get("results", {})
        task_results = results.get("mmlu_professional_law", {})

        # Extract accuracy — lm-eval uses "acc,none" as the key
        accuracy = task_results.get("acc,none",
                   task_results.get("acc", None))
        std_err  = task_results.get("acc_stderr,none",
                   task_results.get("acc_stderr", None))

        if accuracy is not None:
            acc_pct = accuracy * 100
            err_pct = std_err * 100 if std_err else 0

            print("\n" + "="*60)
            print("EVALUATION RESULTS")
            print("="*60)
            print(f"Task:     mmlu_professional_law")
            print(f"Accuracy: {acc_pct:.2f}%")
            print(f"Std Err:  ±{err_pct:.2f}%")
            print(f"Meaning:  Model answered {acc_pct:.1f}% of professional")
            print(f"          law questions correctly")
            print("="*60 + "\n")

            return {
                "model_path":  str(output_path),
                "accuracy":    acc_pct,
                "std_err":     err_pct,
                "task":        "mmlu_professional_law",
                "num_fewshot": 5,
                "timestamp":   datetime.now().isoformat()
            }
        else:
            print(f"[WARN] Could not find accuracy in results: {task_results}")
            return None

    except Exception as e:
        print(f"[WARN] Could not parse results file: {e}")
        return None


def save_summary(results, model_name, output_dir):
    """
    Saves a clean summary JSON alongside the full lm-eval output.
    This summary is what eval_report.py reads to build the
    comparison table showing improvement across all three models.
    """

    summary_file = os.path.join(
        output_dir,
        f"summary_{model_name}.json"
    )

    with open(summary_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Summary saved to: {summary_file}")
    return summary_file


def main():
    args = parse_args()

    # Step 1 — Run evaluation
    output_file = run_evaluation(
        model_path  = args.model_path,
        model_name  = args.model_name,
        output_dir  = args.output_dir,
        num_fewshot = args.num_fewshot,
        batch_size  = args.batch_size
    )

    # Step 2 — Parse and display results
    results = parse_results(output_file)

    # Step 3 — Save clean summary
    if results and args.model_name:
        save_summary(results, args.model_name, args.output_dir)

    print("\nDone. Run eval_report.py to compare all model results.")


if __name__ == "__main__":
    main()
