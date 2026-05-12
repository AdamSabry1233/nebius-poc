# Nebius H200 Cluster — LLM Fine-Tuning PoC

**Cluster:** 2 nodes x 8x H200 GPUs | Partition: earlytalent | QOS: gpulimit

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

    nebius-poc/
    +-- validation/          # 10-test cluster validation suite
    |   +-- tests/           # Individual test scripts (01-10)
    |   +-- validate.sbatch  # Slurm job
    |   +-- generate_report.py
    |
    +-- training/            # Fine-tuning pipeline
    |   +-- config.yaml      # All hyperparameters with justifications
    |   +-- data_prep.py     # MMLU loading and prompt formatting
    |   +-- train.py         # LoRA + FSDP + SFTTrainer
    |   +-- train.sbatch     # Multi-node Slurm job
    |
    +-- evaluation/          # Eval and benchmarking
    |   +-- eval.py          # lm-eval wrapper
    |   +-- eval.sbatch      # Slurm eval job
    |   +-- eval_finetuned.sbatch
    |   +-- eval_report.py   # Three-model comparison table
    |   +-- benchmark_throughput.py
    |   +-- benchmark_throughput.sbatch
    |
    +-- results/             # All outputs saved here

---

## Setup

Requirements:
- 2 nodes with H200 GPUs and InfiniBand
- Slurm with partition=earlytalent, qos=gpulimit
- Python 3.12+, CUDA 12.4+
- HuggingFace account with Mistral license accepted

    git clone https://github.com/AdamSabry1233/nebius-poc
    cd nebius-poc

    python3 -m venv ~/nebius-env
    source ~/nebius-env/bin/activate

    pip install torch==2.6.0 transformers==4.45.0 \
        datasets peft trl accelerate lm-eval pyyaml

    echo 'export HF_TOKEN=your_token_here' >> ~/.bashrc
    source ~/.bashrc

    mkdir -p /mnt/data/hf_cache /mnt/data/outputs \
             /mnt/data/checkpoints /mnt/data/logs

  For automated environment setup:
    bash setup_env.sh

Creates the virtual environment, installs CUDA-enabled PyTorch and all required dependencies automatically.

---

## Stage 1 — Cluster Validation

Before running any training workload, validate the hardware. This catches problems before they waste GPU hours.

    sbatch validation/validate.sbatch
    watch squeue -u $USER

    python3 validation/generate_report.py \
        validation/validation_<jobid>.out

The 10 tests cover GPU health, CUDA functionality, intra-node NVLink bandwidth, inter-node InfiniBand bandwidth, storage throughput, CPU/RAM, a live distributed smoke test with FSDP and AllReduce across all 4 GPUs, GPU stress testing, ECC memory error monitoring, and environment consistency between both nodes.

Our results (job 132):

    All 10 tests passed
    639.8 TFLOPS confirmed
    0 ECC errors
    0.022ms InfiniBand latency
    Environment hash: 7c44c529 — identical on both nodes

The distributed smoke test (test 07) is the most important one. It runs actual FSDP and AllReduce across all 4 GPUs on both nodes and verifies gradient synchronization is mathematically correct before touching any real training.

---

## Stage 2 — Fine-Tuning

### Choices and Why

**Model: Mistral 7B Instruct v0.3**

Llama 3.1 requires manual approval from Meta that can take hours to days. Mistral gives immediate access after accepting the license. Beyond that, Mistral 7B fits comfortably within the 4 GPU budget with FSDP and LoRA, has strong ecosystem support in TRL and HuggingFace, and is the kind of model a startup would realistically deploy. The pipeline works identically for Llama — one line change in config.yaml.

**Dataset: cais/mmlu professional_law**

Three reasons: it has the most training examples of any MMLU category (~1534), the base model scores only ~46% on it leaving real room to improve, and legal domain fine-tuning is a directly relevant startup use case. A category where the model already scores 85% would give nothing to prove.

**Fine-tuning: LoRA**

Trains 41.9M of 7.25B parameters — 0.58% of the model. Full fine-tuning would require ~140GB just for the weights before gradients or optimizer states. LoRA gets comparable improvement at a fraction of the cost, which is what a budget-conscious startup actually needs.

**Distributed: FSDP FULL_SHARD**

ZeRO-3 equivalent — shards everything across all 4 GPUs. Natively integrated with PyTorch and HuggingFace Trainer. DeepSpeed was considered but has known compatibility issues with newer TRL versions.

### Running Training

    python3 training/data_prep.py

    sbatch training/train.sbatch

    watch squeue -u $USER
    tail -f train_<jobid>.out

Hyperparameters (full config with comments in training/config.yaml):

| Parameter | Value | Why |
|---|---|---|
| LoRA r | 16 | Standard starting point for instruction fine-tuning |
| LoRA alpha | 32 | 2x rank — standard scaling convention |
| Learning rate | 2e-4 | Higher than full fine-tuning because only adapters update |
| Epochs | 3 | Safe ceiling before overfitting on 1534 examples |
| Per-GPU batch | 4 | Fits H200 memory comfortably with FSDP |
| Gradient accumulation | 4 | Effective batch size = 64 |
| Scheduler | Cosine | Smooth decay, standard for LLM fine-tuning |

Training completed in ~17 minutes. Model saved to /mnt/data/outputs/finetuned-mistral/ as a 161MB LoRA adapter.

---

## Stage 3 — Evaluation

### Accuracy

    # Baselines
    MODEL_PATH=mistralai/Mistral-7B-v0.3 \
    MODEL_NAME=baseline_raw \
    sbatch evaluation/eval.sbatch

    MODEL_PATH=mistralai/Mistral-7B-Instruct-v0.3 \
    MODEL_NAME=baseline_instruct \
    sbatch evaluation/eval.sbatch

    # Fine-tuned model
    sbatch evaluation/eval_finetuned.sbatch

    # Comparison table
    python3 evaluation/eval_report.py

Results:

| Model | Accuracy | Delta |
|---|---|---|
| Mistral 7B raw | 45.96% | baseline |
| Mistral 7B Instruct | 46.22% | +0.26pp |
| Mistral 7B Fine-tuned | **54.63%** | **+8.67pp** |

Statistically significant — improvement exceeds 2x the standard error of +/-1.27%.

All three models evaluated on the same held-out test split using the same 5-shot protocol and log-likelihood scoring. One variable changed: which model answers the questions.

### Throughput

    sbatch evaluation/benchmark_throughput.sbatch
    cat results/throughput_report.txt

Results:

| Batch | Throughput | Latency |
|---|---|---|
| 1 | 22.9 tok/s | 131.2ms |
| 8 | 159.7 tok/s | 18.8ms |
| 32 | 348.5 tok/s | 8.6ms |
| 64 | **516.1 tok/s** | 5.8ms |

Benchmarked with HuggingFace inference. vLLM had CUDA/NCCL compatibility issues with this cluster's library versions. In production on compatible infrastructure, vLLM's PagedAttention and continuous batching would push this to roughly 1000-2000 tok/s.

Optimized for throughput rather than latency because the startup is building a multi-user system. Throughput maps directly to how many users you can serve simultaneously.

---

## Monitoring

Grafana:

    ssh -L 3000:metrics-grafana.monitoring-system.svc:80 -N user@ip_address
    # Open localhost:3000 in browser

Key panels during training: GPU Utilization, GPU Memory (confirms FSDP sharding), GPU Temperature, InfiniBand Throughput (confirms inter-node gradient sync).

Slurm:

    squeue -u $USER             # job status
    tail -f <name>_<jobid>.out  # live output
    scancel <jobid>             # cancel job
    sinfo                       # cluster resources

---

## Reproducing End to End

    # 1. Environment setup (see Setup section above)

    # 2. Cluster validation
    sbatch validation/validate.sbatch
    python3 validation/generate_report.py validation/validation_<jobid>.out

    # 3. Baselines
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
    sbatch evaluation/benchmark_throughput.sbatch

Total wall time: 3-4 hours, mostly unattended.

---

## Environment

    Python:       3.12.3
    PyTorch:      2.6.0+cu124
    CUDA:         12.4
    Transformers: 4.45.0 (benchmark) / 5.8.0 (training)
    TRL:          1.4.0
    PEFT:         0.19.1
    lm-eval:      0.4.11

---

## Results Files

    results/
    +-- validation_report.txt           cluster validation summary
    +-- eval_comparison_report.txt      three-model accuracy table
    +-- throughput_report.txt           inference benchmark
    +-- eval_baseline_raw_*.json        raw lm-eval output
    +-- eval_baseline_instruct_*.json   raw lm-eval output
    +-- eval_finetuned_*.json           raw lm-eval output
    +-- throughput_results.json         raw benchmark data
