#!/bin/bash
# Test 04: NCCL Inter-Node Communication Check
# Validates InfiniBand bandwidth between the two nodes
# H200 InfiniBand expected bandwidth: ~400 GB/s

echo ""
echo "=================================================="
echo "TEST 04: NCCL Inter-Node Communication Check"
echo "=================================================="

source ~/nebius-env/bin/activate

# ── Step 1: Check InfiniBand interfaces ──────────────
echo ""
echo "--- InfiniBand Interface Status ---"

if command -v ibstat &> /dev/null; then
    ibstat_output=$(ibstat 2>&1)
    if echo "$ibstat_output" | grep -q "Active"; then
        echo "  [PASS] InfiniBand interface is active"
        echo "$ibstat_output" | grep -E "State|Physical|Rate" | head -10
        IB_PASS=true
    else
        echo "  [WARN] InfiniBand interface not active"
        IB_PASS=false
    fi
else
    echo "  [INFO] ibstat not available — testing bandwidth directly"
    IB_PASS=true
fi

# ── Step 2: Check NCCL environment variables ─────────
echo ""
echo "--- NCCL Environment Configuration ---"

if [ "${NCCL_IB_DISABLE}" = "1" ]; then
    echo "  [WARN] NCCL_IB_DISABLE=1 — InfiniBand disabled for NCCL"
else
    echo "  [PASS] NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0} — InfiniBand enabled"
fi

echo "  [INFO] MASTER_ADDR=${MASTER_ADDR:-not set}"
echo "  [INFO] MASTER_PORT=${MASTER_PORT:-not set}"
echo "  [INFO] WORLD_SIZE=${WORLD_SIZE:-not set}"
echo "  [INFO] RANK=${RANK:-not set}"

# ── Step 3: Inter-node bandwidth test ────────────────
echo ""
echo "--- Inter-Node AllReduce Bandwidth Test ---"

if [ "${SLURM_NNODES}" -lt 2 ] 2>/dev/null; then
    echo "  [WARN] Only 1 node allocated — skipping inter-node test"
    echo "--------------------------------------------------"
    echo "TEST 04 RESULT: SKIP (single node)"
    echo "=================================================="
    exit 0
fi

python3 << 'EOF'
import torch
import torch.distributed as dist
import os
import time
import sys

MIN_BANDWIDTH_GBS = 50

def run_internode_bandwidth_test():
    rank        = int(os.environ.get("RANK", 0))
    world_size  = int(os.environ.get("WORLD_SIZE", 1))
    master_addr = os.environ.get("MASTER_ADDR", "localhost")
    master_port = os.environ.get("MASTER_PORT", "29500")
    local_rank  = int(os.environ.get("LOCAL_RANK", 0))

    print(f"  [INFO] Rank {rank}/{world_size} on {master_addr}")

    if world_size < 2:
        print("  [WARN] World size < 2 — cannot test inter-node")
        return True

    dist.init_process_group(
        backend     = "nccl",
        init_method = "env://",
        world_size  = world_size,
        rank        = rank
    )

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    print(f"  [INFO] Process group initialized — running warmup")

    # ── Warmup — essential for accurate inter-node measurement ──
    # InfiniBand connection setup happens on first transfer
    # Without warmup the first timed result is always slow
    warmup_elements = (100 * 1024 * 1024) // 4
    warmup_tensor   = torch.ones(warmup_elements, dtype=torch.float32, device=device)
    dist.all_reduce(warmup_tensor, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    del warmup_tensor
    torch.cuda.empty_cache()
    print(f"  [INFO] Warmup complete — starting timed bandwidth tests")

    # ── Timed tests ──
    test_sizes_mb = [100, 500, 1024]
    all_pass      = True

    for size_mb in test_sizes_mb:
        num_elements = (size_mb * 1024 * 1024) // 4
        tensor       = torch.ones(num_elements, dtype=torch.float32, device=device)

        # Run 3 times, take best result
        times = []
        for _ in range(3):
            start = time.perf_counter()
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            torch.cuda.synchronize()
            end = time.perf_counter()
            times.append(end - start)

        best_time      = min(times)
        data_moved_gb  = size_mb / 1024
        bandwidth_gbs  = data_moved_gb / best_time

        # Only rank 0 prints to avoid duplicate output
        if rank == 0:
            if bandwidth_gbs >= MIN_BANDWIDTH_GBS:
                print(f"  [PASS] {size_mb}MB AllReduce: {bandwidth_gbs:.1f} GB/s")
            else:
                print(f"  [FAIL] {size_mb}MB AllReduce: {bandwidth_gbs:.1f} GB/s "
                      f"(minimum {MIN_BANDWIDTH_GBS} GB/s)")
                all_pass = False

        del tensor
        torch.cuda.empty_cache()

    dist.destroy_process_group()
    return all_pass

try:
    passed = run_internode_bandwidth_test()
    if passed:
        print("  Inter-node bandwidth: PASS")
        sys.exit(0)
    else:
        print("  Inter-node bandwidth: FAIL")
        sys.exit(1)
except Exception as e:
    print(f"  [FAIL] Inter-node test error: {e}")
    sys.exit(1)
EOF

NCCL_EXIT=$?

# ── Step 4: Network latency ───────────────────────────
echo ""
echo "--- Network Latency Check ---"

OTHER_NODE=$(scontrol show hostnames $SLURM_JOB_NODELIST | tail -n 1)

if [ -n "$OTHER_NODE" ]; then
    ping_output=$(ping -c 10 $OTHER_NODE 2>&1)
    if echo "$ping_output" | grep -q "avg"; then
        avg_latency=$(echo "$ping_output" | grep "avg" | awk -F'/' '{print $5}')
        echo "  [INFO] Average latency to $OTHER_NODE: ${avg_latency}ms"
        if (( $(echo "$avg_latency < 1.0" | bc -l) )); then
            echo "  [PASS] Sub-millisecond latency confirmed"
        else
            echo "  [WARN] Latency ${avg_latency}ms — expected <1ms for InfiniBand"
        fi
    else
        echo "  [WARN] Could not measure latency"
    fi
fi

# ── Final Result ──────────────────────────────────────
echo ""
echo "--------------------------------------------------"

if [ $NCCL_EXIT -eq 0 ]; then
    echo "TEST 04 RESULT: PASS"
    EXIT_CODE=0
else
    echo "TEST 04 RESULT: FAIL"
    EXIT_CODE=1
fi

echo "=================================================="
echo ""
exit $EXIT_CODE
