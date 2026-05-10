#!/bin/bash
# =============================================================
# validate.sbatch
# Slurm job file for cluster validation suite
# Runs all 10 validation tests across both nodes
# Submit with: sbatch validate.sbatch
# =============================================================

#SBATCH --job-name=cluster_validation
#SBATCH -N 2                          # both nodes
#SBATCH --partition=earlytalent       # your partition
#SBATCH --qos=gpulimit                # your QOS
#SBATCH --gres=gpu:2                  # 2 GPUs per node = 4 total
#SBATCH --ntasks-per-node=2           # 1 task per node
#SBATCH --cpus-per-task=16            # 16 CPU cores per task
#SBATCH --mem=200G                    # 200GB RAM per node
#SBATCH --time=01:30:00               # 1.5 hours max
#SBATCH --output=validation_%j.out    # all output to one file


# =============================================================
# Environment Setup
# =============================================================

echo "============================================================"
echo "NEBIUS CLUSTER VALIDATION JOB"
echo "============================================================"
echo "Job ID:        $SLURM_JOB_ID"
echo "Nodes:         $SLURM_JOB_NODELIST"
echo "GPUs per node: $SLURM_GPUS_ON_NODE"
echo "Submit time:   $(date '+%Y-%m-%d %H:%M:%S UTC')"
echo "============================================================"
echo ""

# Activate virtual environment
source ~/nebius-env/bin/activate

# =============================================================
# Distributed Training Environment Variables
# These are the same variables your training job will use
# Setting them here lets tests 04 and 07 test the exact
# same configuration as your real training run
# =============================================================

# Master node = first node in the allocation
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=4          # 2 nodes x 2 GPUs = 4 total processes
export NCCL_IB_DISABLE=0    # keep InfiniBand enabled
export NCCL_DEBUG=WARN       # show NCCL warnings but not full debug spam

echo "Distributed environment:"
echo "  MASTER_ADDR: $MASTER_ADDR"
echo "  MASTER_PORT: $MASTER_PORT"
echo "  WORLD_SIZE:  $WORLD_SIZE"
echo "  NCCL_IB_DISABLE: $NCCL_IB_DISABLE"
echo ""

# =============================================================
# Phase 1: Per-Node Tests (runs independently on each node)
# Tests 01-06, 08-10 run on each node separately
# They test that node's own hardware and software
# =============================================================

echo "============================================================"
echo "PHASE 1: Per-Node Validation (both nodes simultaneously)"
echo "============================================================"

srun --ntasks=2 --ntasks-per-node=1 bash -c '
    # Set per-process rank variables
    export RANK=$SLURM_PROCID
    export LOCAL_RANK=0
    export NODE_RANK=$SLURM_NODEID

    echo ""
    echo "Starting per-node tests on: $(hostname) (rank $RANK)"

    # Run all tests except distributed ones (04 and 07)
    # Those need special multi-process launch
    source ~/nebius-env/bin/activate

    python3 ~/nebius-poc/validation/tests/01_gpu_health.py
    python3 ~/nebius-poc/validation/tests/02_cuda_check.py
    bash    ~/nebius-poc/validation/tests/03_nccl_intranode.sh
    python3 ~/nebius-poc/validation/tests/05_storage_bench.py
    python3 ~/nebius-poc/validation/tests/06_cpu_ram_check.py
    python3 ~/nebius-poc/validation/tests/08_gpu_stress.py
    python3 ~/nebius-poc/validation/tests/09_ecc_monitoring.py
    python3 ~/nebius-poc/validation/tests/10_env_consistency.py
'

PHASE1_EXIT=$?
echo ""
echo "Phase 1 complete — exit code: $PHASE1_EXIT"

# =============================================================
# Phase 2: Multi-Node Tests (requires both nodes together)
# Tests 04 and 07 need all 4 processes across both nodes
# to test inter-node communication and distributed training
# =============================================================

echo ""
echo "============================================================"
echo "PHASE 2: Multi-Node Validation (inter-node communication)"
echo "============================================================"

# Test 04: Inter-Node NCCL Bandwidth
echo ""
echo "--- Running Test 04: NCCL Inter-Node ---"

srun --ntasks=4 \
     --ntasks-per-node=2 \
     bash -c '
    export RANK=$SLURM_PROCID
    export LOCAL_RANK=$((SLURM_PROCID % 2))
    export WORLD_SIZE=4
    export NODE_RANK=$SLURM_NODEID
    source ~/nebius-env/bin/activate
    bash ~/nebius-poc/validation/tests/04_nccl_internode.sh
'

TEST04_EXIT=$?
echo "Test 04 exit code: $TEST04_EXIT"

# Test 07: Distributed Smoke Test
echo ""
echo "--- Running Test 07: Distributed Smoke Test ---"

srun --ntasks=4 \
     --ntasks-per-node=2 \
     bash -c '
    export RANK=$SLURM_PROCID
    export LOCAL_RANK=$((SLURM_PROCID % 2))
    export WORLD_SIZE=4
    export NODE_RANK=$SLURM_NODEID
    source ~/nebius-env/bin/activate
    python3 ~/nebius-poc/validation/tests/07_distributed_smoke.py
'

TEST07_EXIT=$?
echo "Test 07 exit code: $TEST07_EXIT"

# =============================================================
# Final Summary
# =============================================================

echo ""
echo "============================================================"
echo "VALIDATION JOB COMPLETE"
echo "============================================================"
echo "Job ID:       $SLURM_JOB_ID"
echo "Completed:    $(date '+%Y-%m-%d %H:%M:%S UTC')"
echo "Phase 1:      $([ $PHASE1_EXIT -eq 0 ] && echo 'PASS' || echo 'FAIL')"
echo "Test 04:      $([ $TEST04_EXIT -eq 0 ] && echo 'PASS' || echo 'FAIL')"
echo "Test 07:      $([ $TEST07_EXIT -eq 0 ] && echo 'PASS' || echo 'FAIL')"
echo ""

# Overall result
if [ $PHASE1_EXIT -eq 0 ] && \
   [ $TEST04_EXIT -eq 0 ] && \
   [ $TEST07_EXIT -eq 0 ]; then
    echo "OVERALL RESULT: ALL TESTS PASSED"
    echo "Cluster is validated and ready for training"
    FINAL_EXIT=0
else
    echo "OVERALL RESULT: SOME TESTS FAILED"
    echo "Review the output above for failure details"
    FINAL_EXIT=1
fi

echo "============================================================"
echo ""
echo "Output saved to: validation_${SLURM_JOB_ID}.out"
echo "Copy to results: cp validation_${SLURM_JOB_ID}.out"
echo "                    ~/nebius-poc/results/validation_report.txt"
echo ""

exit $FINAL_EXIT
