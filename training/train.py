#!/usr/bin/env python3
"""
train.py
Fine-tunes Mistral 7B Instruct on MMLU professional_law
using LoRA via TRL SFTTrainer with FSDP for multi-node
distributed training across 4 GPUs on 2 nodes.

TRL 1.4.0 API:
  - max_length goes in SFTConfig (not max_seq_length)
  - processing_class replaces tokenizer in SFTTrainer
  - logging_dir stays in SFTConfig

Launch via torchrun (handled by train.sbatch):
    torchrun --nproc_per_node=2 --nnodes=2 train.py
"""

import os
import sys
import yaml
import torch
import logging
from datetime import datetime

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig

from data_prep import load_and_prepare_dataset

# ── Logging ───────────────────────────────────────────────────
logging.basicConfig(
    format   = "%(asctime)s - %(levelname)s - %(message)s",
    level    = logging.INFO,
    handlers = [logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def load_config():
    config_path = os.path.join(
        os.path.dirname(__file__), "config.yaml"
    )
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_distributed():
    rank       = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    logger.info(
        f"Distributed setup: rank={rank}, "
        f"world_size={world_size}, local_rank={local_rank}"
    )
    return rank, world_size, local_rank


def load_tokenizer(config, rank):
    if rank == 0:
        logger.info(f"Loading tokenizer: {config['model']['name']}")

    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["name"],
        cache_dir = config["model"]["cache_dir"],
        token     = os.environ.get("HF_TOKEN"),
        use_fast  = True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    tokenizer.padding_side = "right"

    if rank == 0:
        logger.info(f"Tokenizer loaded. Vocab size: {tokenizer.vocab_size}")

    return tokenizer


def load_model(config, rank):
    if rank == 0:
        logger.info(f"Loading model: {config['model']['name']}")

    model = AutoModelForCausalLM.from_pretrained(
        config["model"]["name"],
        dtype     = torch.bfloat16,
        cache_dir = config["model"]["cache_dir"],
        token     = os.environ.get("HF_TOKEN"),
        # No device_map — FSDP handles placement
    )

    model.config.use_cache = False

    if rank == 0:
        total = sum(p.numel() for p in model.parameters())
        logger.info(f"Model loaded. Parameters: {total/1e9:.2f}B")

    return model


def apply_lora(model, config, rank):
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
        trainable = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        total = sum(p.numel() for p in model.parameters())
        pct   = 100 * trainable / total
        logger.info(
            f"LoRA applied. Trainable: "
            f"{trainable/1e6:.1f}M / {total/1e9:.2f}B ({pct:.2f}%)"
        )

    return model


def build_training_args(config, rank):
    train_config = config["training"]

    os.makedirs(train_config["output_dir"],        exist_ok=True)
    os.makedirs(config["paths"]["checkpoint_dir"], exist_ok=True)
    os.makedirs(config["paths"]["log_dir"],        exist_ok=True)

    training_args = SFTConfig(
        # Output
        output_dir                   = train_config["output_dir"],

        # Duration
        num_train_epochs             = train_config["num_train_epochs"],

        # Batch size
        per_device_train_batch_size  = train_config["per_device_train_batch_size"],
        per_device_eval_batch_size   = train_config["per_device_eval_batch_size"],
        gradient_accumulation_steps  = train_config["gradient_accumulation_steps"],

        # Learning rate
        learning_rate                = train_config["learning_rate"],
        lr_scheduler_type            = train_config["lr_scheduler_type"],
        warmup_steps                 = train_config["warmup_steps"],

        # Regularization
        weight_decay                 = train_config["weight_decay"],
        max_grad_norm                = train_config["max_grad_norm"],

        # Precision
        bf16                         = train_config["bf16"],

        # Logging
        logging_steps                = train_config["logging_steps"],

        # Evaluation
        eval_strategy                = "steps",
        eval_steps                   = train_config["eval_steps"],

        # Checkpointing
        save_strategy                = "steps",
        save_steps                   = train_config["save_steps"],
        save_total_limit             = train_config["save_total_limit"],
        load_best_model_at_end       = train_config["load_best_model_at_end"],
        metric_for_best_model        = "eval_loss",

        # Sequence length — TRL 1.4.0 uses max_length not max_seq_length
        max_length                   = config["dataset"]["max_seq_length"],

        # Dataset
        dataset_text_field           = "text",

        # FSDP
        fsdp                         = "full_shard",
        fsdp_config                  = {
            "backward_prefetch":             "backward_pre",
            "forward_prefetch":              True,
            "cpu_offload":                   False,
            "auto_wrap_policy":              "transformer_based_wrap",
            "transformer_layer_cls_to_wrap": "MistralDecoderLayer",
        },

        # Reporting
        report_to                    = train_config["report_to"],
    )

    return training_args


def main():
    # ── Step 1: Distributed Setup ─────────────────────────────
    rank, world_size, local_rank = setup_distributed()

    if rank == 0:
        logger.info("=" * 60)
        logger.info("MISTRAL 7B INSTRUCT — PROFESSIONAL LAW FINE-TUNING")
        logger.info("=" * 60)
        logger.info(f"World size:  {world_size} GPUs across 2 nodes")
        logger.info(f"Strategy:    FSDP FULL_SHARD (ZeRO-3 equivalent)")
        logger.info(f"Technique:   LoRA (r=16, alpha=32)")
        logger.info(f"Dataset:     cais/mmlu professional_law")
        logger.info(f"Start time:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 60)

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
            f"{len(dataset['test'])} train, "
            f"{len(dataset['validation'])} validation examples"
        )

    # ── Step 7: Build Training Arguments ─────────────────────
    training_args = build_training_args(config, rank)

    # ── Step 8: Initialize Trainer ────────────────────────────
    # TRL 1.4.0: processing_class replaces tokenizer
    if rank == 0:
        logger.info("Initializing SFTTrainer...")

    trainer = SFTTrainer(
        model             = model,
        args              = training_args,
        train_dataset     = dataset["test"],
        eval_dataset      = dataset["validation"],
        processing_class  = tokenizer,
    )

    # ── Step 9: Train ─────────────────────────────────────────
    if rank == 0:
        eff_batch = (
            config["training"]["per_device_train_batch_size"] *
            config["training"]["gradient_accumulation_steps"] *
            world_size
        )
        logger.info("Starting training...")
        logger.info(f"Effective batch size: {eff_batch}")

    train_result = trainer.train()

    # ── Step 10: Save Final Model ─────────────────────────────
    if rank == 0:
        logger.info("Training complete. Saving final model...")

    trainer.save_model()

    if rank == 0:
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)

        logger.info("=" * 60)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 60)
        logger.info(
            f"Model saved to: {config['training']['output_dir']}"
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
        logger.info("=" * 60)
        logger.info("Next: run evaluation on fine-tuned model")
        logger.info(
            "  MODEL_PATH=/mnt/data/outputs/finetuned-mistral \\"
        )
        logger.info("  MODEL_NAME=finetuned \\")
        logger.info("  sbatch eval.sbatch")
        logger.info("=" * 60)


if __name__ == "__main__":
    main()
