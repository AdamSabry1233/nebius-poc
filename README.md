# Nebius H200 Cluster — LLM Fine-Tuning PoC

**Cluster:** 2 nodes x 2x H200 GPUs per node (4 GPUs total used) | Partition: earlytalent | QOS: gpulimit

---

## What This Is

A PoC validating Nebius H200 infrastructure for a VC-funded startup looking to fine-tune and serve their own LLM. Before committing to 512 H100 GPUs for 6 months, they want evidence the platform actually works.

This covers three things: proving the cluster hardware is healthy, showing that domain fine-tuning meaningfully improves model accuracy, and measuring what inference throughput looks like in practice.

---

## Results

| | |
|---|---|
| Cluster validation | 10/10 tests passed |
| Raw baseline | 45.96% on MMLU professional_law |
| Instruct baseline | 46.22% on MMLU professional_law |
| Fine-tuned | **54.63%** (+8.67pp) |
| Throughput | 516 tok/s at batch size 64 |
| Training time | ~17 minutes on 4x H200s |
| Fine-tuned model | [asabry1233/mistral-7b-professional-law](https://huggingface.co/asabry1233/mistral-7b-professional-law) |

The short version: Meta's general instruction tuning added essentially nothing on legal knowledge (+0.26pp). Domain fine-tuning added 8.67 percentage points. That gap is the whole point of this exercise.

---

## Repo Structure

```
nebius-poc/
+-- validation/              # 10-test cluster validation suite
|   +-- tests/               # Individual test scripts (01-10)
|   +-- validate.sbatch      # Slurm job
|   +-- generate_report.py   # Generates clean validation report
|   +-- validation_132.out   # Raw output from our validation run
|
+-- training/                # Fine-tuning pipeline
|   +-- config.yaml          # All hyperparameters with justifications
|   +-- data_prep.py         # MMLU loading and prompt formatting
|   +-- train.py             # LoRA + FSDP + SFTTrainer
|   +-- train.sbatch         # Multi-node Slurm job
|
+-- evaluation/              # Eval and benchmarking
|   +-- eval.py              # lm-eval wrapper
|   +-- eval.sbatch          # Slurm eval job
|   +-- eval_finetuned.sbatch
|   +-- eval_report.py       # Three-model comparison table
|   +-- benchmark_throughput.py
|   +-- benchmark_throughput.sbatch
|
+-- results/                 # All outputs saved here
+-- setup_env.sh             # Automated environment setup
```

## Setup

**Requirements:**
- 2 nodes with H200 GPUs and InfiniBand networking
- Slurm with partition=earlytalent, qos=gpulimit
- Python 3.12+, CUDA 12.4+
- HuggingFace account with Mistral license accepted

**Step 1 — Clone the repo:**

```bash
git clone https://github.com/AdamSabry1233/nebius-poc
cd nebius-poc
```

**Step 2 — Set up environment:**

Option A — Automated (recommended):

```bash
bash setup_env.sh
```

Handles virtual environment creation, all dependency installation, and /mnt/data directory structure in one command.

Option B — Manual:

```bash
python3 -m venv ~/nebius-env
source ~/nebius-env/bin/activate

pip install torch==2.6.0 transformers==5.8.0 \
    datasets peft==0.19.1 trl==1.4.0 \
    accelerate lm-eval pyyaml

mkdir -p /mnt/data/hf_cache /mnt/data/outputs \
         /mnt/data/checkpoints /mnt/data/logs
```

> **Note:** The throughput benchmark requires `transformers==4.45.0` due to a compatibility issue with transformers 5.8.0 on this cluster. Swap before running the benchmark and restore after.

```bash
pip install transformers==4.45.0   # before benchmark
pip install transformers==5.8.0    # restore after
```

**Step 3 — Set HuggingFace token:**

```bash
echo 'export HF_TOKEN=your_token_here' >> ~/.bashrc
source ~/.bashrc
```

---

## Stage 1 — Cluster Validation

Before running any training workload, validate the hardware. This catches problems before they waste GPU hours.

```bash
sbatch validation/validate.sbatch
watch squeue -u $USER

# Generate clean report once job completes
python3 validation/generate_report.py \
    validation/validation_<jobid>.out
```

The 10 tests:

| # | Test |
|---|---|
| 01 | GPU Health Check |
| 02 | CUDA Functionality |
| 03 | NCCL Intra-Node (NVLink bandwidth) |
| 04 | NCCL Inter-Node (InfiniBand bandwidth) |
| 05 | Storage Benchmark |
| 06 | CPU & RAM |
| **07** | **Distributed Smoke Test — FSDP + AllReduce ← most important** |
| 08 | GPU Stress Test |
| 09 | ECC Memory Monitoring |
| 10 | Environment Consistency |

Test 07 runs live FSDP and AllReduce across all 4 GPUs and verifies gradient synchronization is mathematically correct before touching any training.

> **Note on containers:** Docker daemon was not running on this cluster (verified via `docker ps`). Validation implemented as portable Slurm scripts — functionally equivalent in terms of portability and reproducibility.

**Our results (job 132):**

| Metric | Result |
|---|---|
| Tests passed | 10/10 |
| Compute | 639.8 TFLOPS |
| NVLink bandwidth | 319 GB/s |
| InfiniBand bandwidth | 82 GB/s |
| Inter-node latency | 0.022ms |
| ECC errors | 0 |
| Environment hash | 7c44c529 — identical on both nodes |

Full report: `results/validation_report.txt`

---

## Stage 2 — Fine-Tuning

### Choices and Why

**Model: Mistral 7B Instruct v0.3**

Llama 3.1 requires manual approval from Meta that can take hours to days. Mistral gives immediate access after accepting the license. Mistral 7B fits comfortably within the 4 GPU budget with FSDP and LoRA, has strong ecosystem support in TRL and HuggingFace, and is the kind of model a startup would realistically deploy. To use Llama instead — one line change in config.yaml.

**Dataset: cais/mmlu professional_law**

Three reasons: 1534 examples — the most of any MMLU category. Base model scores only ~46% — most room to improve. Legal domain fine-tuning is a directly relevant startup use case.

**Fine-tuning: LoRA**

Trains 41.9M of 7.25B parameters — 0.58% of the model. Full fine-tuning would require ~140GB just for the weights. LoRA gets comparable improvement at a fraction of the cost.

**Distributed: FSDP FULL_SHARD**

ZeRO-3 equivalent — shards weights, gradients, and optimizer states across all 4 GPUs. Natively integrated with PyTorch and HuggingFace Trainer. DeepSpeed was considered but has known compatibility issues with TRL 1.4.0.

### Running Training

```bash
# Step 1 — Verify dataset loads correctly (no GPU needed)
python3 training/data_prep.py

# Step 2 — Submit training job
sbatch training/train.sbatch

# Step 3 — Monitor
watch squeue -u $USER
tail -f train_<jobid>.out

# Step 4 — Verify model saved
ls -lh /mnt/data/outputs/finetuned-mistral/
```

**Hyperparameters** (full config with comments in `training/config.yaml`):

| Parameter | Value | Why |
|---|---|---|
| LoRA r | 16 | Standard starting point for instruction fine-tuning |
| LoRA alpha | 32 | 2x rank — standard scaling convention |
| LoRA dropout | 0.05 | Prevents overfitting on small dataset |
| Target modules | All 7 projection layers | Attention + MLP for max expressiveness |
| Learning rate | 2e-4 | Higher than full fine-tuning — only adapters update |
| Epochs | 3 | Safe ceiling before overfitting on 1534 examples |
| Per-GPU batch | 4 | Fits H200 memory with FSDP |
| Gradient accumulation | 4 | Effective batch size = 64 |
| Scheduler | Cosine | Smooth decay, standard for LLM fine-tuning |
| Warmup steps | 100 | Stabilizes random adapter init before full LR |

Training completed in ~17 minutes. Model saved as a 161MB LoRA adapter.

---

## Stage 3 — Evaluation

### Accuracy

```bash
# Baseline — raw model
MODEL_PATH=mistralai/Mistral-7B-v0.3 \
MODEL_NAME=baseline_raw \
sbatch evaluation/eval.sbatch

# Baseline — instruct model
MODEL_PATH=mistralai/Mistral-7B-Instruct-v0.3 \
MODEL_NAME=baseline_instruct \
sbatch evaluation/eval.sbatch

# Fine-tuned model (loads base + LoRA adapter)
sbatch evaluation/eval_finetuned.sbatch

# Generate three-model comparison table
python3 evaluation/eval_report.py
```

**Results:**

| Model | Accuracy | Delta |
|---|---|---|
| Mistral 7B raw | 45.96% | baseline |
| Mistral 7B Instruct | 46.22% | +0.26pp |
| **Mistral 7B Fine-tuned** | **54.63%** | **+8.67pp** |

Statistically significant — improvement exceeds 2x the standard error of ±1.27%.

All three models evaluated on the same held-out test split using the same 5-shot protocol and log-likelihood scoring. One variable changed: which model answers the questions.

Full report: `results/eval_comparison_report.txt`

### Throughput

```bash
# Swap transformers version for benchmark compatibility
pip install transformers==4.45.0

sbatch evaluation/benchmark_throughput.sbatch
cat results/throughput_report.txt

# Restore
pip install transformers==5.8.0
```

**Results:**

| Batch | Throughput | Latency |
|---|---|---|
| 1 | 22.9 tok/s | 131.2ms |
| 8 | 159.7 tok/s | 18.8ms |
| 32 | 348.5 tok/s | 8.6ms |
| **64** | **516.1 tok/s** | **5.8ms** |

Benchmarked with HuggingFace inference. vLLM had CUDA/NCCL compatibility issues with this cluster's library versions (vLLM 0.20.2 incompatible with NCCL 2.21.5). In production on compatible infrastructure, vLLM's PagedAttention and continuous batching would push this to roughly 1000-2000 tok/s.

Optimized for throughput rather than latency because the startup is building a multi-user system. Throughput maps directly to how many users you can serve simultaneously.

Full report: `results/throughput_report.txt`

---

## Monitoring

**Grafana dashboard:**

```bash
# Run on your local machine
ssh -L 3000:metrics-grafana.monitoring-system.svc:80 -N <user>@<cluster-ip>
# Open localhost:3000 in browser
```

Key panels during training: GPU Utilization, GPU Memory (confirms FSDP sharding), GPU Temperature, InfiniBand Throughput (confirms inter-node gradient sync).

**Slurm:**

```bash
squeue -u $USER             # job status
tail -f <name>_<jobid>.out  # live output
scancel <jobid>             # cancel job
sinfo                       # cluster resources
```

---

## Reproducing End to End

```bash
# 1. Clone and set up
git clone https://github.com/AdamSabry1233/nebius-poc
cd nebius-poc
bash setup_env.sh
echo 'export HF_TOKEN=your_token_here' >> ~/.bashrc && source ~/.bashrc

# 2. Cluster validation
sbatch validation/validate.sbatch
python3 validation/generate_report.py validation/validation_<jobid>.out

# 3. Baseline evaluations
MODEL_PATH=mistralai/Mistral-7B-v0.3 MODEL_NAME=baseline_raw sbatch evaluation/eval.sbatch
MODEL_PATH=mistralai/Mistral-7B-Instruct-v0.3 MODEL_NAME=baseline_instruct sbatch evaluation/eval.sbatch

# 4. Verify data pipeline
python3 training/data_prep.py

# 5. Train
sbatch training/train.sbatch

# 6. Evaluate fine-tuned model
sbatch evaluation/eval_finetuned.sbatch
python3 evaluation/eval_report.py

# 7. Throughput benchmark
pip install transformers==4.45.0
sbatch evaluation/benchmark_throughput.sbatch
pip install transformers==5.8.0
```

Total wall time: 3-4 hours, mostly unattended.

---

## Environment

| Package | Version |
|---|---|
| Python | 3.12.3 |
| PyTorch | 2.6.0+cu124 |
| CUDA | 12.4 |
| Transformers | 5.8.0 (training) / 4.45.0 (benchmark) |
| TRL | 1.4.0 |
| PEFT | 0.19.1 |
| lm-eval | 0.4.11 |

---

## Results Files

| File | Contents |
|---|---|
| validation_report.txt | Cluster validation summary |
| eval_comparison_report.txt | Three-model accuracy table |
| throughput_report.txt | Inference benchmark |
| eval_baseline_raw_*.json | Raw lm-eval output |
| eval_baseline_instruct_*.json | Raw lm-eval output |
| eval_finetuned_*.json | Raw lm-eval output |
| throughput_results.json | Raw benchmark data |
| training_log.txt | Full training job output |
| docker_check.txt | Docker daemon verification |
