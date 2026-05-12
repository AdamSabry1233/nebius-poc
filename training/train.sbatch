#!/bin/bash
# =============================================================
# train.sbatch
# Slurm job file for fine-tuning Llama 3.1 8B Instruct
# on MMLU professional_law using LoRA + FSDP
# across 4 GPUs on 2 nodes.
#
# Submit with:
#   sbatch train.sbatch
# =============================================================

#SBATCH --job-name=llama_finetune
#SBATCH -N 2                          # 2 nodes
#SBATCH --partition=earlytalent
#SBATCH --qos=gpulimit
#SBATCH --gres=gpu:2                  # 2 GPUs per node = 4 total
#SBATCH --ntasks-per-node=2           # 1 task per GPU per node
#SBATCH --cpus-per-task=16            # 16 CPU cores per task
#SBATCH --mem=200G                    # 200GB RAM per node
#SBATCH --time=12:00:00               # 12 hours max
#SBATCH --output=train_%j.out         # all output to one file

# =============================================================
# Environment Setup
# =============================================================

echo "============================================================"
echo "LLAMA 3.1 8B INSTRUCT — PROFESSIONAL LAW FINE-TUNING"
echo "============================================================"
echo "Job ID:        $SLURM_JOB_ID"
echo "Nodes:         $SLURM_JOB_NODELIST"
echo "GPUs per node: $SLURM_GPUS_ON_NODE"
echo "Tasks/node:    $SLURM_NTASKS_PER_NODE"
echo "Submit time:   $(date '+%Y-%m-%d %H:%M:%S UTC')"
echo "============================================================"

# Activate virtual environment
source ~/nebius-env/bin/activate

# ── HuggingFace Authentication ────────────────────────────────
# Required to download gated Llama 3.1 model
export HF_TOKEN=$HF_TOKEN
export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN

# ── Model Cache ───────────────────────────────────────────────
# Cache to /mnt/data not /home
# Llama 3.1 8B = 16GB — only download once
export HF_HOME=/mnt/data/hf_cache
export TRANSFORMERS_CACHE=/mnt/data/hf_cache

# ── NCCL Configuration ────────────────────────────────────────
# These tell NCCL how to communicate between nodes
export NCCL_IB_DISABLE=0              # keep InfiniBand enabled
export NCCL_DEBUG=WARN                # show warnings not full debug
export NCCL_SOCKET_IFNAME=eth0        # use eth0 for OOB coordination

# ── FSDP Configuration ────────────────────────────────────────
export FSDP_SHARDING_STRATEGY=1       # 1 = FULL_SHARD
export FSDP_CPU_OFFLOAD=0             # no CPU offload — plenty of GPU mem

# ── Performance Tuning ────────────────────────────────────────
export OMP_NUM_THREADS=8              # OpenMP threads for CPU ops
export TOKENIZERS_PARALLELISM=false   # avoid tokenizer warning in workers

# ── Distributed Training Coordinates ─────────────────────────
# MASTER_ADDR is the IP of the first node
# All processes connect to this node to coordinate
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=4                   # 2 nodes x 2 GPUs = 4 total

echo ""
echo "Distributed environment:"
echo "  MASTER_ADDR: $MASTER_ADDR"
echo "  MASTER_PORT: $MASTER_PORT"
echo "  WORLD_SIZE:  $WORLD_SIZE"
echo "  NCCL_IB_DISABLE: $NCCL_IB_DISABLE"
echo ""

# Create required directories
mkdir -p /mnt/data/hf_cache
mkdir -p /mnt/data/outputs/finetuned-llama
mkdir -p /mnt/data/checkpoints
mkdir -p /mnt/data/logs

# =============================================================
# Launch Training
# =============================================================

echo "Launching distributed training..."
echo ""

# srun launches torchrun on both nodes simultaneously
# torchrun starts 2 processes per node (one per GPU)
# resulting in 4 total processes across both nodes
#
# --nproc_per_node=2    → 2 processes per node (2 GPUs per node)
# --nnodes=2            → total of 2 nodes
# --node_rank           → which node this is (0 or 1)
# --master_addr         → IP of rank 0 node
# --master_port         → port for process coordination
# --rdzv_backend=c10d   → rendezvous backend for process discovery
#                         c10d uses PyTorch's built-in coordination

srun --ntasks=2 \
     --ntasks-per-node=1 \
     bash -c "
         torchrun \
             --nproc_per_node=2 \
             --nnodes=2 \
             --node_rank=\$SLURM_NODEID \
             --master_addr=$MASTER_ADDR \
             --master_port=$MASTER_PORT \
             ~/nebius-poc/training/train.py
     "

TRAIN_EXIT=$?

# =============================================================
# Post-Training Summary
# =============================================================

echo ""
echo "============================================================"
echo "TRAINING JOB COMPLETE"
echo "============================================================"
echo "Job ID:      $SLURM_JOB_ID"
echo "End time:    $(date '+%Y-%m-%d %H:%M:%S UTC')"
echo "Exit code:   $TRAIN_EXIT"
echo ""

if [ $TRAIN_EXIT -eq 0 ]; then
    echo "STATUS: SUCCESS"
    echo ""
    echo "Model saved to: /mnt/data/outputs/finetuned-llama"
    echo ""
    echo "============================================================"
    echo "NEXT STEPS"
    echo "============================================================"
    echo ""
    echo "1. Run evaluation on fine-tuned model:"
    echo ""
    echo "   MODEL_PATH=/mnt/data/outputs/finetuned-llama \\"
    echo "   MODEL_NAME=finetuned \\"
    echo "   sbatch eval.sbatch"
    echo ""
    echo "2. Generate comparison report:"
    echo ""
    echo "   python3 ~/nebius-poc/evaluation/eval_report.py"
    echo ""
    echo "3. Run throughput benchmark:"
    echo ""
    echo "   sbatch benchmark_throughput.sbatch"
    echo ""
else
    echo "STATUS: FAILED — check output above for errors"
    echo ""
    echo "Common issues:"
    echo "  - HF_TOKEN not set or expired"
    echo "  - Model not yet approved on HuggingFace"
    echo "  - Out of memory — reduce per_device_train_batch_size"
    echo "  - NCCL timeout — check InfiniBand with ibstat"
fi

echo "============================================================"

exit $TRAIN_EXIT
