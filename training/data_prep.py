#!/usr/bin/env python3
"""
data_prep.py
Loads the cais/mmlu professional_law dataset from HuggingFace
and formats it into a prompt template the model can learn from.

MMLU split structure:
  test:       1534 examples  <- we fine-tune on this (largest split)
  validation: 170 examples   <- we monitor training loss on this
  dev:        5 examples     <- few-shot examples used by lm-eval

The raw MMLU data looks like:
{
  "question": "Which of the following...",
  "choices": ["A...", "B...", "C...", "D..."],
  "answer": 1   <- index of correct choice (0-3)
}

We convert this into:
### Question:
Which of the following...

A) choice 1
B) choice 2
C) choice 3
D) choice 4

### Answer:
B
"""

import os
import sys
import yaml
from datasets import load_dataset

# Answer index to letter mapping
ANSWER_MAP = {0: "A", 1: "B", 2: "C", 3: "D"}


def load_config():
    """Load hyperparameters from config.yaml."""
    config_path = os.path.join(
        os.path.dirname(__file__),
        "config.yaml"
    )
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def format_example(example):
    """
    Converts one raw MMLU example into the prompt template.
    The model learns to predict the answer letter
    given the question and choices.
    """
    question      = example["question"]
    choices       = example["choices"]
    answer        = example["answer"]
    answer_letter = ANSWER_MAP[answer]

    choice_lines = "\n".join([
        f"{letter}) {text}"
        for letter, text in zip(["A", "B", "C", "D"], choices)
    ])

    prompt = f"""### Question:
{question}

{choice_lines}

### Answer:
{answer_letter}"""

    return {"text": prompt}


def load_and_prepare_dataset(config):
    """
    Loads professional_law split of cais/mmlu
    and formats every example into prompt template.

    Split usage:
      test split (1534)       -> fine-tuning training data
      validation split (170)  -> monitor training loss
      dev split (5)           -> used by lm-eval for few-shot
    """
    dataset_config = config["dataset"]

    print(f"\nLoading dataset: {dataset_config['name']}")
    print(f"Subset:          {dataset_config['subset']}")
    print(f"Cache dir:       {config['model']['cache_dir']}")
    sys.stdout.flush()

    dataset = load_dataset(
        dataset_config["name"],
        dataset_config["subset"],
        cache_dir=config["model"]["cache_dir"]
    )

    print(f"\nDataset splits:")
    for split_name, split_data in dataset.items():
        print(f"  {split_name}: {len(split_data)} examples")
    sys.stdout.flush()

    print(f"\nFormatting examples into prompt template...")
    sys.stdout.flush()

    # Use test split column names as reference
    # since that is our primary training split
    formatted = dataset.map(
        format_example,
        remove_columns=dataset["test"].column_names,
        desc="Formatting examples"
    )

    print(f"\nFormatting complete.")
    print(f"\nExample formatted sample:")
    print("-" * 50)
    print(formatted["test"][0]["text"])
    print("-" * 50)
    sys.stdout.flush()

    return formatted


def verify_format(dataset):
    """
    Sanity checks the formatted dataset.
    Verifies every example has the expected format
    before we start training on it.
    """
    print(f"\nVerifying dataset format...")
    sys.stdout.flush()

    issues = 0

    # Check first 100 examples from test split
    for i, example in enumerate(dataset["test"]):
        text = example["text"]

        if "### Question:" not in text:
            print(f"  [WARN] Example {i} missing ### Question:")
            issues += 1

        if "### Answer:" not in text:
            print(f"  [WARN] Example {i} missing ### Answer:")
            issues += 1

        answer_line = text.split("### Answer:")[-1].strip()
        if answer_line not in ["A", "B", "C", "D"]:
            print(
                f"  [WARN] Example {i} invalid answer: "
                f"'{answer_line}'"
            )
            issues += 1

        if i >= 100:
            break

    if issues == 0:
        print(f"  [PASS] All examples correctly formatted")
    else:
        print(f"  [FAIL] Found {issues} formatting issues")

    sys.stdout.flush()
    return issues == 0


def get_dataset_stats(dataset):
    """Prints useful statistics about the dataset."""
    print(f"\nDataset Statistics:")
    print("-" * 40)

    for split_name, split_data in dataset.items():
        texts      = split_data["text"]
        lengths    = [len(t.split()) for t in texts]
        avg_length = sum(lengths) / len(lengths)
        max_length = max(lengths)
        est_tokens = avg_length * 1.3

        print(f"\n  {split_name} split:")
        print(f"    Examples:   {len(split_data)}")
        print(
            f"    Avg length: {avg_length:.0f} words "
            f"(~{est_tokens:.0f} tokens)"
        )
        print(f"    Max length: {max_length} words")

    print("-" * 40)
    sys.stdout.flush()


def main():
    """
    Test data preparation independently.
    Run before submitting training job to verify
    dataset loads and formats correctly.
    """
    print("Starting data preparation check...")
    sys.stdout.flush()

    config  = load_config()
    dataset = load_and_prepare_dataset(config)
    valid   = verify_format(dataset)

    if not valid:
        print("\n[FAIL] Fix formatting issues before training")
        return

    get_dataset_stats(dataset)

    print("\n[PASS] Dataset ready for training")
    print(f"\nSplit usage during fine-tuning:")
    print(f"  Train on:    test split       ({len(dataset['test'])} examples)")
    print(f"  Monitor on:  validation split ({len(dataset['validation'])} examples)")
    print(f"  lm-eval uses dev split for    ({len(dataset['dev'])} few-shot examples)")
    print(f"\nNote: lm-eval final evaluation uses test split.")
    print(f"      Aware of overlap — acceptable for PoC scope.")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
