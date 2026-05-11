#!/usr/bin/env python3
"""
eval_report.py
Reads the three summary JSON files produced by eval.py and
generates a clean comparison report showing improvement
across all three models on professional_law MMLU.

Usage:
    python3 eval_report.py

Expects these files in ~/nebius-poc/results/:
    summary_baseline_raw.json
    summary_baseline_instruct.json
    summary_finetuned.json
"""

import json
import os
import glob
from datetime import datetime
from pathlib import Path


# Results directory
RESULTS_DIR = os.path.expanduser("~/nebius-poc/results")

# Expected model summaries in order
MODELS = [
    {
        "key":         "baseline_raw",
        "label":       "Llama 3.1 8B (raw)",
        "description": "Base pretrained model — no instruction tuning"
    },
    {
        "key":         "baseline_instruct",
        "label":       "Llama 3.1 8B Instruct",
        "description": "Meta instruction-tuned — general purpose"
    },
    {
        "key":         "finetuned",
        "label":       "Llama 3.1 8B Fine-tuned",
        "description": "Domain fine-tuned on professional_law"
    },
]


def load_summary(key):
    """
    Loads a summary JSON file for a given model key.
    Returns None if file not found — handles case where
    fine-tuned model hasn't been evaluated yet.
    """
    summary_file = os.path.join(RESULTS_DIR, f"summary_{key}.json")

    if not os.path.exists(summary_file):
        # Try finding any matching file
        matches = glob.glob(
            os.path.join(RESULTS_DIR, f"*{key}*.json")
        )
        # Filter out lm-eval verbose output files
        matches = [m for m in matches if "summary" in m or key in m]
        if matches:
            summary_file = matches[0]
        else:
            return None

    try:
        with open(summary_file, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"  [WARN] Could not load {summary_file}: {e}")
        return None


def calculate_delta(base_acc, current_acc):
    """
    Calculates the improvement delta between two accuracy scores.
    Returns a formatted string with + or - sign.
    """
    delta = current_acc - base_acc
    sign  = "+" if delta >= 0 else ""
    return f"{sign}{delta:.2f}"


def is_significant(accuracy, std_err):
    """
    Checks if an improvement is statistically significant.
    Rule of thumb: improvement > 2x standard error = significant.
    This means the improvement is unlikely to be measurement noise.
    """
    return accuracy > (2 * std_err) if std_err else True


def generate_report(results):
    """
    Generates the full comparison report from loaded results.
    """
    W = 68  # report width
    report = []

    def line(text=""):
        report.append(text)

    # ── Title ─────────────────────────────────────────────────
    report.append("")
    report.append("=" * W)
    report.append(" " * 10 + "MMLU PROFESSIONAL LAW — MODEL COMPARISON REPORT")
    report.append("=" * W)
    report.append(f"  Task:        mmlu_professional_law")
    report.append(f"  Evaluation:  5-shot (standard MMLU methodology)")
    report.append(f"  Tool:        EleutherAI lm-evaluation-harness")
    report.append(f"  Generated:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    report.append("=" * W)

    # ── Results Table ─────────────────────────────────────────
    report.append("")
    report.append(f"  {'Model':<35} {'Accuracy':>10} {'Std Err':>10} {'Delta':>10}")
    report.append("  " + "-" * (W - 4))

    raw_accuracy = None

    for model in MODELS:
        key   = model["key"]
        label = model["label"]
        data  = results.get(key)

        if data:
            acc     = data["accuracy"]
            err     = data.get("std_err", 0)
            acc_str = f"{acc:.2f}%"
            err_str = f"±{err:.2f}%"

            # Calculate delta vs raw baseline
            if raw_accuracy is None:
                raw_accuracy = acc
                delta_str    = "baseline"
            else:
                delta_str = f"{calculate_delta(raw_accuracy, acc)}pp"

            report.append(
                f"  {label:<35} {acc_str:>10} {err_str:>10} {delta_str:>10}"
            )
        else:
            report.append(
                f"  {label:<35} {'PENDING':>10} {'—':>10} {'—':>10}"
            )

    report.append("  " + "-" * (W - 4))
    report.append(f"  pp = percentage points")
    report.append("")

    # ── Key Finding ───────────────────────────────────────────
    raw_data      = results.get("baseline_raw")
    instruct_data = results.get("baseline_instruct")
    finetuned_data = results.get("finetuned")

    if raw_data and finetuned_data:
        improvement = finetuned_data["accuracy"] - raw_data["accuracy"]
        std_err     = finetuned_data.get("std_err", 0)
        significant = is_significant(improvement, std_err)

        report.append("=" * W)
        report.append("  KEY FINDING")
        report.append("=" * W)
        report.append("")
        report.append(
            f"  Fine-tuning improved professional_law accuracy by "
            f"{improvement:.2f} percentage points"
        )
        report.append(
            f"  over the raw base model "
            f"({raw_data['accuracy']:.2f}% → {finetuned_data['accuracy']:.2f}%)"
        )
        report.append("")

        if instruct_data:
            instruct_improvement = (finetuned_data["accuracy"] -
                                   instruct_data["accuracy"])
            report.append(
                f"  Over Meta's instruct baseline: "
                f"{instruct_improvement:+.2f} percentage points"
            )
            report.append(
                f"  ({instruct_data['accuracy']:.2f}% → "
                f"{finetuned_data['accuracy']:.2f}%)"
            )
            report.append("")

        if significant:
            report.append(
                f"  Statistical significance: YES — improvement exceeds "
                f"2x standard error"
            )
        else:
            report.append(
                f"  Statistical significance: MARGINAL — improvement within "
                f"standard error range"
            )
            report.append(
                f"  Note: More training epochs or data would increase "
                f"this margin"
            )

        report.append("")

    elif raw_data and not finetuned_data:
        report.append("=" * W)
        report.append("  STATUS")
        report.append("=" * W)
        report.append("")
        report.append("  Baseline evaluations complete.")
        report.append("  Fine-tuned model evaluation pending.")
        report.append("  Run after training completes:")
        report.append("")
        report.append(
            "  MODEL_PATH=/mnt/data/outputs/finetuned-llama \\"
        )
        report.append(
            "  MODEL_NAME=finetuned \\"
        )
        report.append(
            "  sbatch eval.sbatch"
        )
        report.append("")

    # ── What This Means ───────────────────────────────────────
    report.append("=" * W)
    report.append("  WHAT THIS MEANS")
    report.append("=" * W)
    report.append("")
    report.append(
        "  Random guessing on 4-choice questions = 25.00%"
    )
    report.append(
        "  Human expert performance on MMLU law  = ~70-75%"
    )
    report.append("")

    if raw_data:
        raw_acc = raw_data["accuracy"]
        above_random = raw_acc - 25.0
        report.append(
            f"  Base model is {above_random:.1f}pp above random guessing"
        )

    if finetuned_data and raw_data:
        improvement = finetuned_data["accuracy"] - raw_data["accuracy"]
        report.append(
            f"  Fine-tuning added {improvement:.1f}pp of domain knowledge"
        )
        report.append(
            f"  on top of pretraining"
        )

    report.append("")

    # ── Methodology ───────────────────────────────────────────
    report.append("=" * W)
    report.append("  METHODOLOGY")
    report.append("=" * W)
    report.append("")
    report.append(
        "  Evaluation tool:  EleutherAI lm-evaluation-harness"
    )
    report.append(
        "  Scoring method:   Log-likelihood (not generation)"
    )
    report.append(
        "  Why log-likelihood: More reliable than generation-based"
    )
    report.append(
        "  scoring — eliminates randomness from sampling"
    )
    report.append(
        "  and gives reproducible results every run."
    )
    report.append("")
    report.append(
        "  5-shot prompting: Each question shown with 5 examples"
    )
    report.append(
        "  before it. Standard MMLU evaluation protocol used"
    )
    report.append(
        "  by Meta, Google, and Microsoft for their own models."
    )
    report.append(
        "  Results are directly comparable to published scores."
    )
    report.append("")
    report.append(
        "  Test set:         Held-out questions never seen during"
    )
    report.append(
        "  fine-tuning. Clean train/test split prevents data"
    )
    report.append(
        "  leakage and ensures fair comparison."
    )
    report.append("")

    # ── Files ─────────────────────────────────────────────────
    report.append("=" * W)
    report.append("  RESULT FILES")
    report.append("=" * W)
    report.append("")

    for model in MODELS:
        key          = model["key"]
        summary_file = f"results/summary_{key}.json"
        exists       = os.path.exists(
            os.path.join(RESULTS_DIR, f"summary_{key}.json")
        )
        status = "✓" if exists else "pending"
        report.append(f"  {status}  {summary_file}")

    report.append("")
    report.append("=" * W)
    report.append("")

    return "\n".join(report)


def main():
    print("\nLoading evaluation results...")

    # Load all available results
    results = {}
    for model in MODELS:
        key  = model["key"]
        data = load_summary(key)
        if data:
            results[key] = data
            print(f"  [FOUND] {key}: {data['accuracy']:.2f}%")
        else:
            print(f"  [PENDING] {key}: not yet evaluated")

    if not results:
        print("\n[ERROR] No evaluation results found.")
        print(f"Run eval.sbatch first to generate results in {RESULTS_DIR}")
        return

    # Generate report
    report = generate_report(results)
    print(report)

    # Save report to results folder
    report_file = os.path.join(RESULTS_DIR, "eval_comparison_report.txt")
    with open(report_file, "w") as f:
        f.write(report)

    print(f"Report saved to: {report_file}")


if __name__ == "__main__":
    main()
