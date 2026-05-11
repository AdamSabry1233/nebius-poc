#!/bin/bash
# =============================================================
# eval.sbatch
# Slurm job file for running MMLU professional_law evaluation
# Runs lm-evaluation-harness on a specified model
#
# Usage:
#   sbatch eval.sbatch --model_path meta-llama/Llama-3.1-8B \
#                      --model_name baseline_raw
#
# Or set variables directly:
#   MODEL_PATH=meta-llama/Llama-3.1-8B \
#   MODEL_NAME=baseline_raw \
#   sbatch eval.sbatch
# =============================================================

#SBATCH --job-name=mmlu_eval
#SBATCH -N 1                          # evaluation only needs 1 node
#SBATCH --partition=earlytalent
#SBATCH --qos=gpulimit
#SBATCH --gres=gpu:2                  # 2 GPUs — enough for 8B model
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=02:00:00               # 2 hours max per eval run
#SBATCH --output=eval_%j.out

# =============================================================
# Environment
# =============================================================

echo "============================================================"
echo "MMLU PROFESSIONAL LAW EVALUATION"
echo "============================================================"
echo "Job ID:      $SLURM_JOB_ID"
echo "Node:        $SLURMD_NODENAME"
echo "GPUs:        $CUDA_VISIBLE_DEVICES"
echo "Start time:  $(date '+%Y-%m-%d %H:%M:%S UTC')"
echo "============================================================"

# Activate virtual environment
source ~/nebius-env/bin/activate

# HuggingFace token — needed to download Llama models
export HF_TOKEN=$HF_TOKEN
export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN

# Disable tokenizer parallelism warning
export TOKENIZERS_PARALLELISM=false

# Tell HuggingFace to cache models in /mnt/data
# so we don't re-download 16GB every time
export HF_HOME=/mnt/data/hf_cache
export TRANSFORMERS_CACHE=/mnt/data/hf_cache

echo ""
echo "HF cache directory: $HF_HOME"
echo "Model to evaluate:  ${MODEL_PATH:-meta-llama/Llama-3.1-8B-Instruct}"
echo "Model name:         ${MODEL_NAME:-unnamed}"
echo ""

# Create cache and results directories
mkdir -p /mnt/data/hf_cache
mkdir -p ~/nebius-poc/results

# =============================================================
# Run Evaluation
# =============================================================

echo "Starting evaluation..."
echo ""

python3 ~/nebius-poc/evaluation/eval.py \
    --model_path  "${MODEL_PATH:-meta-llama/Llama-3.1-8B-Instruct}" \
    --model_name  "${MODEL_NAME:-unnamed}" \
    --output_dir  ~/nebius-poc/results \
    --num_fewshot 5 \
    --batch_size  auto

EVAL_EXIT=$?

# =============================================================
# Summary
# =============================================================

echo ""
echo "============================================================"
echo "EVALUATION COMPLETE"
echo "============================================================"
echo "Job ID:      $SLURM_JOB_ID"
echo "Model:       ${MODEL_PATH:-meta-llama/Llama-3.1-8B-Instruct}"
echo "End time:    $(date '+%Y-%m-%d %H:%M:%S UTC')"
echo "Exit code:   $EVAL_EXIT"
echo ""

if [ $EVAL_EXIT -eq 0 ]; then
    echo "STATUS: PASS — evaluation completed successfully"
    echo ""
    echo "Results saved to: ~/nebius-poc/results/"
    echo ""
    echo "Next steps:"
    echo "  1. Run eval_report.py to compare all model scores"
    echo "  2. Submit training job: sbatch train.sbatch"
else
    echo "STATUS: FAIL — check output above for errors"
fi

echo "============================================================"

exit $EVAL_EXIT
