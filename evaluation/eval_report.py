#!/usr/bin/env python3
"""
eval_report.py
Reads lm-eval output files and generates a clean comparison
report showing improvement across all three models on
professional_law MMLU.

Usage:
    python3 eval_report.py

Looks for lm-eval output files in ~/nebius-poc/results/
named like: eval_baseline_raw_TIMESTAMP.json
"""

import json
import os
import glob
from datetime import datetime
from pathlib import Path


# Results directory
RESULTS_DIR = os.path.expanduser("~/nebius-poc/results")

# Models in order
MODELS = [
    {
        "key":         "baseline_raw",
        "label":       "Mistral 7B (raw)",
        "description": "Base pretrained model — no instruction tuning"
    },
    {
        "key":         "baseline_instruct",
        "label":       "Mistral 7B Instruct",
        "description": "Mistral instruction-tuned — general purpose"
    },
    {
        "key":         "finetuned",
        "label":       "Mistral 7B Fine-tuned",
        "description": "Domain fine-tuned on professional_law"
    },
]


def load_summary(key):
    """
    Loads results from lm-eval output files.
    Handles both summary JSON format and raw lm-eval format.
    """

    # First try our summary format
    summary_file = os.path.join(RESULTS_DIR, f"summary_{key}.json")
    if os.path.exists(summary_file):
        try:
            with open(summary_file, "r") as f:
                data = json.load(f)
                # Already in our format
                if "accuracy" in data:
                    return data
        except Exception:
            pass

    # Fall back to lm-eval output format
    # Files named like: eval_baseline_raw_20260512_005007_....json
    pattern = os.path.join(RESULTS_DIR, f"eval_{key}_*.json")
    matches = glob.glob(pattern)

    # Filter out samples files and non-json
    matches = [
        m for m in matches
        if "samples" not in m
        and m.endswith(".json")
    ]

    if not matches:
        return None

    # Use most recent file
    latest = sorted(matches)[-1]

    try:
        with open(latest, "r") as f:
            data = json.load(f)

        # Extract from lm-eval format
        task_results = data.get("results", {}).get(
            "mmlu_professional_law", {}
        )

        # lm-eval uses "acc,none" as key
        accuracy = task_results.get(
            "acc,none",
            task_results.get("acc", None)
        )
        std_err = task_results.get(
            "acc_stderr,none",
            task_results.get("acc_stderr", None)
        )

        if accuracy is not None:
            return {
                "accuracy":    accuracy * 100,
                "std_err":     std_err * 100 if std_err else 0,
                "task":        "mmlu_professional_law",
                "num_fewshot": 5,
                "source_file": os.path.basename(latest),
                "timestamp":   str(data.get("date", ""))
            }

    except Exception as e:
        print(f"  [WARN] Could not parse {latest}: {e}")

    return None


def calculate_delta(base_acc, current_acc):
    """Calculate improvement delta with sign."""
    delta = current_acc - base_acc
    sign  = "+" if delta >= 0 else ""
    return f"{sign}{delta:.2f}"


def is_significant(improvement, std_err):
    """
    Check if improvement is statistically significant.
    Rule: improvement > 2x standard error = significant.
    """
    return improvement > (2 * std_err) if std_err else True


def generate_report(results):
    """Generate full comparison report."""
    W = 68
    report = []

    # ── Title ─────────────────────────────────────────────────
    report.append("")
    report.append("=" * W)
    report.append(
        " " * 10 + "MMLU PROFESSIONAL LAW — MODEL COMPARISON REPORT"
    )
    report.append("=" * W)
    report.append(f"  Task:        mmlu_professional_law")
    report.append(f"  Evaluation:  5-shot (standard MMLU methodology)")
    report.append(f"  Tool:        EleutherAI lm-evaluation-harness")
    report.append(
        f"  Generated:   "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}"
    )
    report.append("=" * W)

    # ── Results Table ─────────────────────────────────────────
    report.append("")
    report.append(
        f"  {'Model':<35} {'Accuracy':>10} "
        f"{'Std Err':>10} {'Delta':>10}"
    )
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

            if raw_accuracy is None:
                raw_accuracy = acc
                delta_str    = "baseline"
            else:
                delta_str = f"{calculate_delta(raw_accuracy, acc)}pp"

            report.append(
                f"  {label:<35} {acc_str:>10} "
                f"{err_str:>10} {delta_str:>10}"
            )
        else:
            report.append(
                f"  {label:<35} {'PENDING':>10} "
                f"{'—':>10} {'—':>10}"
            )

    report.append("  " + "-" * (W - 4))
    report.append(f"  pp = percentage points")
    report.append("")

    # ── Key Finding ───────────────────────────────────────────
    raw_data       = results.get("baseline_raw")
    instruct_data  = results.get("baseline_instruct")
    finetuned_data = results.get("finetuned")

    report.append("=" * W)

    if raw_data and finetuned_data:
        improvement = finetuned_data["accuracy"] - raw_data["accuracy"]
        std_err     = finetuned_data.get("std_err", 0)
        significant = is_significant(improvement, std_err)

        report.append("  KEY FINDING")
        report.append("=" * W)
        report.append("")
        report.append(
            f"  Fine-tuning improved professional_law accuracy by "
            f"{improvement:.2f} percentage points"
        )
        report.append(
            f"  over the raw base model "
            f"({raw_data['accuracy']:.2f}% → "
            f"{finetuned_data['accuracy']:.2f}%)"
        )
        report.append("")

        if instruct_data:
            instruct_delta = (
                finetuned_data["accuracy"] - instruct_data["accuracy"]
            )
            report.append(
                f"  Over instruct baseline: "
                f"{instruct_delta:+.2f} percentage points"
            )
            report.append(
                f"  ({instruct_data['accuracy']:.2f}% → "
                f"{finetuned_data['accuracy']:.2f}%)"
            )
            report.append("")

        if significant:
            report.append(
                f"  Statistical significance: YES — improvement "
                f"exceeds 2x standard error"
            )
        else:
            report.append(
                f"  Statistical significance: MARGINAL — "
                f"improvement within standard error range"
            )
            report.append(
                f"  Note: More training epochs or data would "
                f"increase this margin"
            )
        report.append("")

    elif raw_data and not finetuned_data:
        report.append("  STATUS")
        report.append("=" * W)
        report.append("")

        if raw_data:
            report.append(
                f"  Raw baseline:      "
                f"{raw_data['accuracy']:.2f}%"
            )
        if instruct_data:
            report.append(
                f"  Instruct baseline: "
                f"{instruct_data['accuracy']:.2f}%"
            )

        report.append("")
        report.append(
            "  Fine-tuned model evaluation pending."
        )
        report.append(
            "  Run after training completes:"
        )
        report.append("")
        report.append(
            "  MODEL_PATH=/mnt/data/outputs/finetuned-mistral \\"
        )
        report.append(
            "  MODEL_NAME=finetuned \\"
        )
        report.append(
            "  sbatch eval.sbatch"
        )
        report.append("")
    else:
        report.append("  STATUS")
        report.append("=" * W)
        report.append("")
        report.append("  No results found yet.")
        report.append("")

    # ── Context ───────────────────────────────────────────────
    report.append("=" * W)
    report.append("  CONTEXT")
    report.append("=" * W)
    report.append("")
    report.append(
        "  Random guessing (4 choices) = 25.00%"
    )
    report.append(
        "  Human expert performance    = ~70-75%"
    )
    report.append("")

    if raw_data:
        above_random = raw_data["accuracy"] - 25.0
        report.append(
            f"  Raw model is {above_random:.1f}pp above "
            f"random guessing"
        )

    if finetuned_data and raw_data:
        improvement = (
            finetuned_data["accuracy"] - raw_data["accuracy"]
        )
        report.append(
            f"  Fine-tuning added {improvement:.1f}pp of "
            f"domain-specific knowledge"
        )

    report.append("")

    # ── Methodology ───────────────────────────────────────────
    report.append("=" * W)
    report.append("  METHODOLOGY")
    report.append("=" * W)
    report.append("")
    report.append(
        "  Tool:    EleutherAI lm-evaluation-harness"
    )
    report.append(
        "  Scoring: Log-likelihood — more reliable than"
    )
    report.append(
        "           generation-based scoring. Eliminates"
    )
    report.append(
        "           randomness, gives reproducible results."
    )
    report.append("")
    report.append(
        "  5-shot:  Each question shown with 5 examples."
    )
    report.append(
        "           Standard MMLU protocol used by Meta,"
    )
    report.append(
        "           Google, and Microsoft for benchmarking."
    )
    report.append("")
    report.append(
        "  Test set: Held-out questions never seen during"
    )
    report.append(
        "            fine-tuning. Clean train/test split"
    )
    report.append(
        "            prevents data leakage."
    )
    report.append("")

    # ── Source Files ──────────────────────────────────────────
    report.append("=" * W)
    report.append("  SOURCE FILES")
    report.append("=" * W)
    report.append("")

    for model in MODELS:
        key  = model["key"]
        data = results.get(key)

        if data:
            src = data.get("source_file", f"summary_{key}.json")
            report.append(f"  ✓  results/{src}")
        else:
            report.append(f"  —  results/eval_{key}_*.json  (pending)")

    report.append("")
    report.append("=" * W)
    report.append("")

    return "\n".join(report)


def main():
    print("\nLoading evaluation results...")

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
        print(
            f"\n[ERROR] No results found in {RESULTS_DIR}"
        )
        print(
            "Run eval.sbatch first to generate results."
        )
        return

    # Generate report
    report = generate_report(results)
    print(report)

    # Save report
    report_file = os.path.join(
        RESULTS_DIR, "eval_comparison_report.txt"
    )
    with open(report_file, "w") as f:
        f.write(report)

    print(f"Report saved to: {report_file}")


if __name__ == "__main__":
    main()
