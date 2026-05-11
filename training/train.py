#!/usr/bin/env python3
"""
train.py
Fine-tunes Llama 3.1 8B Instruct on MMLU professional_law
using LoRA via TRL SFTTrainer with FSDP for multi-node
distributed training across 4 GPUs on 2 nodes.

Launch via torchrun (handled by train.sbatch):
    torchrun --nproc_per_node=2 --nnodes=2 train.py

Do NOT run directly with python3 — FSDP requires torchrun
to initialize the distributed process group correctly.
"""

import os
import sys
import yaml
import torch
import logging
from datetime import datetime
from pathlib import Path

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import torch.distributed as dist

# Import our data preparation functions
from data_prep import load_and_prepare_dataset, format_example

# ── Logging Setup ─────────────────────────────────────────────
logging.basicConfig(
    format  = "%(asctime)s - %(levelname)s - %(message)s",
    level   = logging.INFO,
    handlers= [logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def load_config():
    """Load hyperparameters from config.yaml."""
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_distributed():
    """
    Initialize the distributed training environment.
    Reads environment variables set by torchrun/Slurm.
    These tell each process where it is in the cluster.
    """
    rank       = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Set the GPU for this process
    torch.cuda.set_device(local_rank)

    logger.info(
        f"Distributed setup: rank={rank}, "
        f"world_size={world_size}, "
        f"local_rank={local_rank}"
    )

    return rank, world_size, local_rank


def load_tokenizer(config, rank):
    """
    Load the Llama tokenizer.
    The tokenizer converts text into token IDs that the model
    can process. Llama uses a BPE tokenizer with a 128k vocab.
    """
    if rank == 0:
        logger.info(f"Loading tokenizer: {config['model']['name']}")

    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["name"],
        cache_dir  = config["model"]["cache_dir"],
        token      = os.environ.get("HF_TOKEN"),
        use_fast   = True,   # use the fast Rust tokenizer
    )

    # Llama doesn't have a padding token by default
    # We set it to the end-of-sequence token
    # This is required for batched training where sequences
    # have different lengths and need to be padded to same length
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Pad on the right side
    # Important for causal LM — padding on left would shift
    # the position encodings and confuse the model
    tokenizer.padding_side = "right"

    if rank == 0:
        logger.info(
            f"Tokenizer loaded. Vocab size: {tokenizer.vocab_size}"
        )

    return tokenizer


def load_model(config, rank):
    """
    Load Llama 3.1 8B Instruct in bfloat16 precision.

    We load WITHOUT FSDP wrapping here — FSDP is applied
    by the Trainer automatically based on the fsdp argument
    in TrainingArguments. Loading the full model first then
    letting the Trainer shard it is the correct pattern with TRL.
    """
    if rank == 0:
        logger.info(f"Loading model: {config['model']['name']}")
        logger.info("This downloads 16GB on first run — may take a few minutes")

    model = AutoModelForCausalLM.from_pretrained(
        config["model"]["name"],
        torch_dtype = torch.bfloat16,
        cache_dir   = config["model"]["cache_dir"],
        token       = os.environ.get("HF_TOKEN"),
        # device_map is NOT set here
        # FSDP handles device placement — setting device_map
        # would conflict with FSDP and cause errors
    )

    # Disable model parallelism warning
    model.config.use_cache = False
    # use_cache=True is for inference (KV cache speeds up generation)
    # During training we don't need KV cache and it conflicts
    # with gradient checkpointing

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(
            f"Model loaded. Total parameters: "
            f"{total_params/1e9:.2f}B"
        )

    return model


def apply_lora(model, config, rank):
    """
    Apply LoRA adapters to the model.
    Freezes all original weights and adds trainable
    adapter matrices to the specified layers.

    After this function:
    - 8B original parameters: FROZEN (not updated)
    - ~20M adapter parameters: TRAINABLE (get gradients)
    """
    lora_config = config["lora"]

    peft_config = LoraConfig(
        r              = lora_config["r"],
        lora_alpha     = lora_config["lora_alpha"],
        lora_dropout   = lora_config["lora_dropout"],
        bias           = lora_config["bias"],
        task_type      = TaskType.CAUSAL_LM,
        target_modules = lora_config["target_modules"],
    )

    model = get_peft_model(model, peft_config)

    if rank == 0:
        # Print trainable parameter count
        # This confirms LoRA is applied correctly
        trainable     = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        total         = sum(p.numel() for p in model.parameters())
        trainable_pct = 100 * trainable / total

        logger.info(
            f"LoRA applied. Trainable parameters: "
            f"{trainable/1e6:.1f}M / {total/1e9:.2f}B "
            f"({trainable_pct:.2f}%)"
        )

    return model


def build_training_args(config, rank):
    """
    Build the TrainingArguments object.
    This controls everything about the training loop —
    batch size, learning rate, checkpointing, logging etc.

    The FSDP settings here tell the Trainer to wrap the model
    with FSDP automatically — we don't do it manually.
    """
    train_config = config["training"]

    # Create output directories
    os.makedirs(train_config["output_dir"],           exist_ok=True)
    os.makedirs(config["paths"]["checkpoint_dir"],    exist_ok=True)
    os.makedirs(config["paths"]["log_dir"],           exist_ok=True)

    training_args = SFTConfig(
        # Output
        output_dir              = train_config["output_dir"],

        # Training duration
        num_train_epochs        = train_config["num_train_epochs"],

        # Batch size and accumulation
        per_device_train_batch_size  = train_config["per_device_train_batch_size"],
        per_device_eval_batch_size   = train_config["per_device_eval_batch_size"],
        gradient_accumulation_steps  = train_config["gradient_accumulation_steps"],

        # Learning rate and schedule
        learning_rate           = train_config["learning_rate"],
        lr_scheduler_type       = train_config["lr_scheduler_type"],
        warmup_steps            = train_config["warmup_steps"],

        # Regularization
        weight_decay            = train_config["weight_decay"],
        max_grad_norm           = train_config["max_grad_norm"],

        # Precision
        bf16                    = train_config["bf16"],

        # Logging
        logging_steps           = train_config["logging_steps"],
        logging_dir             = config["paths"]["log_dir"],

        # Evaluation
        eval_strategy           = "steps",
        eval_steps              = train_config["eval_steps"],

        # Checkpointing
        save_strategy           = "steps",
        save_steps              = train_config["save_steps"],
        save_total_limit        = train_config["save_total_limit"],
        load_best_model_at_end  = train_config["load_best_model_at_end"],
        metric_for_best_model   = "eval_loss",

        # Sequence length
        max_seq_length          = config["dataset"]["max_seq_length"],

        # FSDP configuration
        # This tells the Trainer to wrap the model with FSDP
        # using the settings from our config
        fsdp                    = "full_shard",
        fsdp_config             = {
            "backward_prefetch":     "backward_pre",
            "forward_prefetch":      True,
            "cpu_offload":           False,
            "auto_wrap_policy":      "transformer_based_wrap",
            "transformer_layer_cls_to_wrap": "LlamaDecoderLayer",
        },

        # Reporting
        report_to               = train_config["report_to"],

        # Dataset field
        dataset_text_field      = "text",
    )

    return training_args


def main():
    """
    Main training function.
    Orchestrates the complete fine-tuning pipeline:
    1. Setup distributed environment
    2. Load tokenizer and model
    3. Apply LoRA
    4. Load and format dataset
    5. Configure training
    6. Train
    7. Save final model
    """

    # ── Step 1: Distributed Setup ─────────────────────────────
    rank, world_size, local_rank = setup_distributed()

    if rank == 0:
        logger.info("="*60)
        logger.info("LLAMA 3.1 8B INSTRUCT — PROFESSIONAL LAW FINE-TUNING")
        logger.info("="*60)
        logger.info(f"World size:  {world_size} GPUs across 2 nodes")
        logger.info(f"Strategy:    FSDP FULL_SHARD (ZeRO-3 equivalent)")
        logger.info(f"Technique:   LoRA (r=16, alpha=32)")
        logger.info(f"Dataset:     cais/mmlu professional_law")
        logger.info(f"Start time:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("="*60)

    # ── Step 2: Load Config ───────────────────────────────────
    config = load_config()

    # ── Step 3: Load Tokenizer ────────────────────────────────
    tokenizer = load_tokenizer(config, rank)

    # ── Step 4: Load Model ────────────────────────────────────
    model = load_model(config, rank)

    # ── Step 5: Apply LoRA ────────────────────────────────────
    model = apply_lora(model, config, rank)

    # ── Step 6: Load Dataset ──────────────────────────────────
    if rank == 0:
        logger.info("Loading and formatting dataset...")

    dataset = load_and_prepare_dataset(config)

    if rank == 0:
        logger.info(
            f"Dataset ready: "
            f"{len(dataset['train'])} train, "
            f"{len(dataset['validation'])} validation examples"
        )

    # ── Step 7: Build Training Arguments ─────────────────────
    training_args = build_training_args(config, rank)

    # ── Step 8: Initialize Trainer ────────────────────────────
    if rank == 0:
        logger.info("Initializing SFTTrainer...")

    trainer = SFTTrainer(
        model           = model,
        args            = training_args,
        train_dataset   = dataset["train"],
        eval_dataset    = dataset["validation"],
        tokenizer       = tokenizer,
    )

    # ── Step 9: Train ─────────────────────────────────────────
    if rank == 0:
        logger.info("Starting training...")
        logger.info(
            f"Effective batch size: "
            f"{config['training']['per_device_train_batch_size']} "
            f"x {config['training']['gradient_accumulation_steps']} "
            f"x {world_size} = "
            f"{config['training']['per_device_train_batch_size'] * config['training']['gradient_accumulation_steps'] * world_size}"
        )

    train_result = trainer.train()

    # ── Step 10: Save Final Model ─────────────────────────────
    if rank == 0:
        logger.info("Training complete. Saving final model...")

    trainer.save_model()

    # Save training metrics
    if rank == 0:
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)

        logger.info("="*60)
        logger.info("TRAINING COMPLETE")
        logger.info("="*60)
        logger.info(
            f"Final model saved to: "
            f"{config['training']['output_dir']}"
        )
        logger.info(
            f"Training loss:  {metrics.get('train_loss', 'N/A'):.4f}"
        )
        logger.info(
            f"Runtime:        "
            f"{metrics.get('train_runtime', 0)/3600:.2f} hours"
        )
        logger.info(
            f"Samples/second: "
            f"{metrics.get('train_samples_per_second', 'N/A'):.2f}"
        )
        logger.info("="*60)
        logger.info(
            "Next step: run evaluation on the fine-tuned model"
        )
        logger.info(
            f"  MODEL_PATH={config['training']['output_dir']} \\"
        )
        logger.info(
            f"  MODEL_NAME=finetuned \\"
        )
        logger.info(
            f"  sbatch eval.sbatch"
        )
        logger.info("="*60)


if __name__ == "__main__":
    main()
