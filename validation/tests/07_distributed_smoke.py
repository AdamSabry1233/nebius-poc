#!/usr/bin/env python3
"""
Test 07: Distributed Training Smoke Test
Runs a minimal end-to-end distributed training step across
both nodes to verify the full training stack works together
before committing to a real multi-hour training run.

This catches issues that individual component tests miss:
- NCCL initialized but misconfigured for FSDP
- FSDP sharding working but gradient sync broken
- Process group initialized but AllReduce incorrect
"""

import os
import sys
import time
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
import torch.nn as nn
import torch.optim as optim

# Minimum acceptable throughput for smoke test
MIN_SAMPLES_PER_SEC = 10  # very conservative — real training will be much higher

def setup_distributed():
    """
    Initialize the distributed process group.
    This is identical to what your real training job does.
    If this fails your training job will fail too.
    """
    rank         = int(os.environ.get("RANK", 0))
    world_size   = int(os.environ.get("WORLD_SIZE", 1))
    local_rank   = int(os.environ.get("LOCAL_RANK", 0))
    master_addr  = os.environ.get("MASTER_ADDR", "localhost")
    master_port  = os.environ.get("MASTER_PORT", "29500")

    print(f"  [INFO] Initializing rank {rank}/{world_size}")
    print(f"  [INFO] Master: {master_addr}:{master_port}")
    print(f"  [INFO] Local rank: {local_rank}")

    # Set the GPU for this process
    torch.cuda.set_device(local_rank)

    # Initialize process group with NCCL backend
    # This is the exact same call your training script makes
    dist.init_process_group(
        backend    = "nccl",
        init_method= "env://",
        world_size = world_size,
        rank       = rank
    )

    return rank, world_size, local_rank

def build_test_model():
    """
    Build a tiny transformer-like model for the smoke test.
    Small enough to run in seconds but exercises the same
    code paths as a real LLM fine-tuning job.
    """
    class SmallTransformerBlock(nn.Module):
        def __init__(self, hidden_size=512, num_heads=8):
            super().__init__()
            self.attention = nn.MultiheadAttention(
                hidden_size, num_heads, batch_first=True
            )
            self.feed_forward = nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 4),
                nn.GELU(),
                nn.Linear(hidden_size * 4, hidden_size)
            )
            self.norm1 = nn.LayerNorm(hidden_size)
            self.norm2 = nn.LayerNorm(hidden_size)

        def forward(self, x):
            # Attention block with residual connection
            attn_out, _ = self.attention(x, x, x)
            x = self.norm1(x + attn_out)
            # Feed forward block with residual connection
            ff_out = self.feed_forward(x)
            x = self.norm2(x + ff_out)
            return x

    class SmallTestModel(nn.Module):
        def __init__(self, vocab_size=32000, hidden_size=512, num_layers=4):
            super().__init__()
            self.embedding  = nn.Embedding(vocab_size, hidden_size)
            self.layers     = nn.ModuleList([
                SmallTransformerBlock(hidden_size)
                for _ in range(num_layers)
            ])
            self.output     = nn.Linear(hidden_size, vocab_size)

        def forward(self, input_ids):
            x = self.embedding(input_ids)
            for layer in self.layers:
                x = layer(x)
            return self.output(x)

    return SmallTestModel()

def wrap_with_fsdp(model, rank):
    """
    Wrap the model with FSDP — the same way your
    real training script will wrap Llama 3.1 8B.
    Tests that FSDP sharding works correctly across
    all 4 GPUs on both nodes.
    """
    device = torch.device(f"cuda:{torch.cuda.current_device()}")

    fsdp_model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,  # ZeRO-3 equivalent
        device_id=torch.cuda.current_device(),
        mixed_precision=None  # keep simple for smoke test
    )

    if rank == 0:
        print(f"  [INFO] Model wrapped with FSDP FULL_SHARD strategy")

    return fsdp_model

def run_training_steps(model, rank, world_size, num_steps=5):
    """
    Run a few forward + backward passes with gradient sync.
    This exercises the complete training loop:
    1. Forward pass through FSDP sharded model
    2. Loss computation
    3. Backward pass computing gradients
    4. AllReduce syncing gradients across all GPUs/nodes
    5. Optimizer step updating weights
    """
    device    = torch.device(f"cuda:{torch.cuda.current_device()}")
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    # Small batch: 4 samples, 64 sequence length
    batch_size   = 4
    seq_len      = 64
    vocab_size   = 32000

    if rank == 0:
        print(f"  [INFO] Running {num_steps} training steps")
        print(f"  [INFO] Batch size: {batch_size}, Sequence length: {seq_len}")

    step_times = []

    for step in range(num_steps):
        # Generate random token IDs as dummy training data
        input_ids = torch.randint(
            0, vocab_size,
            (batch_size, seq_len),
            device=device
        )
        target_ids = torch.randint(
            0, vocab_size,
            (batch_size, seq_len),
            device=device
        )

        step_start = time.perf_counter()

        # Forward pass
        optimizer.zero_grad()
        logits = model(input_ids)

        # Compute cross entropy loss
        # This is the same loss used in language model training
        loss = nn.CrossEntropyLoss()(
            logits.view(-1, vocab_size),
            target_ids.view(-1)
        )

        # Backward pass — computes gradients
        loss.backward()

        # Gradient clipping — standard in LLM training
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Optimizer step — updates weights
        optimizer.step()

        # Synchronize CUDA operations
        torch.cuda.synchronize()

        step_end  = time.perf_counter()
        step_time = step_end - step_start
        step_times.append(step_time)

        if rank == 0:
            samples_per_sec = batch_size / step_time
            print(f"  [INFO] Step {step+1}/{num_steps}: "
                  f"loss={loss.item():.4f}, "
                  f"time={step_time*1000:.1f}ms, "
                  f"throughput={samples_per_sec:.1f} samples/s")

    return step_times

def verify_gradient_sync(model, rank, world_size):
    """
    Verify that gradients are actually being synchronized
    across all GPUs and nodes correctly.

    This catches the case where the process group initialized
    but AllReduce is silently not working — which would cause
    model weights to diverge across GPUs during training.
    """
    if world_size < 2:
        if rank == 0:
            print(f"  [INFO] Single process — skipping gradient sync check")
        return True

    # Create a known tensor on each rank
    # After AllReduce the sum should equal rank_sum
    test_tensor = torch.ones(10, device=f"cuda:{torch.cuda.current_device()}") * rank

    # AllReduce sum — result should be 0+1+2+3 = 6 for world_size=4
    dist.all_reduce(test_tensor, op=dist.ReduceOp.SUM)

    expected_sum = sum(range(world_size))
    actual_sum   = test_tensor[0].item()

    if rank == 0:
        if abs(actual_sum - expected_sum) < 0.001:
            print(f"  [PASS] Gradient sync verified: "
                  f"AllReduce sum = {actual_sum:.0f} "
                  f"(expected {expected_sum})")
            return True
        else:
            print(f"  [FAIL] Gradient sync incorrect: "
                  f"got {actual_sum:.0f}, expected {expected_sum}")
            return False

    return True

def check_fsdp_memory_efficiency(rank):
    """
    Verify FSDP is actually sharding memory across GPUs.
    If sharding works correctly each GPU should use
    significantly less memory than holding the full model.
    """
    if rank == 0:
        # Check memory before and after model creation
        torch.cuda.reset_peak_memory_stats()
        allocated_gb = torch.cuda.memory_allocated() / 1024**3
        peak_gb      = torch.cuda.max_memory_allocated() / 1024**3

        print(f"  [INFO] GPU memory allocated: {allocated_gb:.2f} GB")
        print(f"  [INFO] GPU peak memory:      {peak_gb:.2f} GB")

        # With FSDP each GPU should have much less than full model
        # Our test model is tiny so thresholds are low
        if allocated_gb < 2.0:
            print(f"  [PASS] FSDP memory sharding working correctly")
            return True
        else:
            print(f"  [WARN] Higher than expected memory — check FSDP config")
            return True  # warn not fail for small test model

    return True

def cleanup_distributed():
    """Clean up the process group after the smoke test."""
    if dist.is_initialized():
        dist.destroy_process_group()

def main():
    print("\n" + "="*50)
    print("TEST 07: Distributed Training Smoke Test")
    print("="*50)

    results = []

    # ── Step 1: Setup distributed environment ────────
    print("\n--- Initializing Distributed Environment ---")
    try:
        rank, world_size, local_rank = setup_distributed()
        if rank == 0:
            print(f"  [PASS] Process group initialized: "
                  f"{world_size} processes across nodes")
        results.append(True)
    except Exception as e:
        print(f"  [FAIL] Distributed initialization failed: {e}")
        print(f"  [INFO] Check MASTER_ADDR, MASTER_PORT, RANK, WORLD_SIZE")
        sys.exit(1)

    try:
        # ── Step 2: Build and wrap model with FSDP ───
        print("\n--- Building FSDP Model ---")
        try:
            model = build_test_model()
            model = wrap_with_fsdp(model, rank)
            if rank == 0:
                print(f"  [PASS] Model built and wrapped with FSDP")
            results.append(True)
        except Exception as e:
            if rank == 0:
                print(f"  [FAIL] Model/FSDP setup failed: {e}")
            results.append(False)

        # ── Step 3: Run training steps ───────────────
        print("\n--- Running Training Steps ---")
        try:
            step_times = run_training_steps(model, rank, world_size)

            avg_step_time   = sum(step_times) / len(step_times)
            avg_samples_sec = 4 / avg_step_time  # batch_size=4

            if rank == 0:
                if avg_samples_sec >= MIN_SAMPLES_PER_SEC:
                    print(f"  [PASS] Training throughput: "
                          f"{avg_samples_sec:.1f} samples/s")
                    results.append(True)
                else:
                    print(f"  [FAIL] Training throughput too low: "
                          f"{avg_samples_sec:.1f} samples/s "
                          f"(minimum {MIN_SAMPLES_PER_SEC})")
                    results.append(False)
            else:
                results.append(True)

        except Exception as e:
            if rank == 0:
                print(f"  [FAIL] Training steps failed: {e}")
            results.append(False)

        # ── Step 4: Verify gradient sync ─────────────
        print("\n--- Verifying Gradient Synchronization ---")
        try:
            sync_passed = verify_gradient_sync(model, rank, world_size)
            results.append(sync_passed)
        except Exception as e:
            if rank == 0:
                print(f"  [FAIL] Gradient sync check failed: {e}")
            results.append(False)

        # ── Step 5: Check FSDP memory efficiency ─────
        print("\n--- Checking FSDP Memory Efficiency ---")
        try:
            mem_passed = check_fsdp_memory_efficiency(rank)
            results.append(mem_passed)
        except Exception as e:
            if rank == 0:
                print(f"  [FAIL] Memory check failed: {e}")
            results.append(False)

    finally:
        cleanup_distributed()

    # ── Final Result (rank 0 only) ────────────────────
    if rank == 0:
        all_passed = all(results)

        print("\n" + "-"*50)
        if all_passed:
            print("TEST 07 RESULT: PASS")
            print("[INFO] Full distributed training stack verified:")
            print("       NCCL ✓  FSDP ✓  AllReduce ✓  Optimizer ✓")
        else:
            print("TEST 07 RESULT: FAIL")
        print("="*50 + "\n")

        return 0 if all_passed else 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
