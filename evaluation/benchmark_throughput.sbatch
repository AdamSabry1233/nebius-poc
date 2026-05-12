#!/bin/bash
#SBATCH --job-name=throughput_bench
#SBATCH -N 1
#SBATCH --partition=earlytalent
#SBATCH --qos=gpulimit
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=01:00:00
#SBATCH --output=benchmark_%j.out

echo "============================================================"
echo "INFERENCE THROUGHPUT BENCHMARK"
echo "============================================================"
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURMD_NODENAME"
echo "Start:     $(date '+%Y-%m-%d %H:%M:%S UTC')"
echo "============================================================"

source ~/nebius-env/bin/activate
export HF_HOME=/mnt/data/hf_cache
export TOKENIZERS_PARALLELISM=false

python3 ~/nebius-poc/evaluation/benchmark_throughput.py \
    --model_path  /mnt/data/outputs/finetuned-mistral \
    --output_dir  ~/nebius-poc/results \
    --batch_sizes 1 2 4 8 16 32 64 \
    --input_length  128 \
    --output_length 128 \
    --num_requests  100

echo ""
echo "Benchmark complete."
echo "Results: ~/nebius-poc/results/throughput_report.txt"
