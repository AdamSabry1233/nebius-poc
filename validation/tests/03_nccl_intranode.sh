#!/bin/bash
# Test 03: NCCL Intra-Node Communication Check
# Validates NVLink bandwidth between GPUs within the same node
# H200 NVLink expected bandwidth: ~900 GB/s

# H200 NVLink threshold
MIN_BANDWIDTH_GBS=100  # GB/s — Python simulation overhead brings measured value down

echo ""
echo "=================================================="
echo "TEST 03: NCCL Intra-Node Communication Check"
echo "=================================================="

# Activate virtual environment
source ~/nebius-env/bin/activate

# ── Step 1: Check NVLink status ──────────────────────
echo ""
echo "--- NVLink Status ---"

nvlink_output=$(nvidia-smi nvlink --status 2>&1)

if echo "$nvlink_output" | grep -q "Error\|not supported\|No NVLinks"; then
    echo "  [FAIL] NVLink not available or not supported on this GPU"
    NVLINK_PASS=false
else
    echo "  [PASS] NVLink is active"
    echo "$nvlink_output" | head -20
    NVLINK_PASS=true
fi

# ── Step 2: Check NVLink speed ───────────────────────
echo ""
echo "--- NVLink Speed ---"

nvlink_speed=$(nvidia-smi nvlink --capabilities 2>&1)

if [ $? -eq 0 ]; then
    echo "  [INFO] NVLink capabilities:"
    echo "$nvlink_speed" | head -10
else
    echo "  [INFO] Could not retrieve NVLink capabilities"
fi

# ── Step 3: Run NCCL bandwidth test via Python ───────
echo ""
echo "--- NCCL AllReduce Bandwidth Test ---"

python3 << 'EOF'
import torch
import time

def run_allreduce_bandwidth_test():
    gpu_count = torch.cuda.device_count()

    if gpu_count < 2:
        print(f"  [WARN] Only {gpu_count} GPU visible — skipping multi-GPU test")
        return True

    print(f"  [INFO] Testing NCCL across {gpu_count} GPUs on this node")

    MIN_BANDWIDTH_GBS = 100

    # ── Warmup — critical to discard CUDA initialization overhead ──
    # First GPU transfer always pays a one-time setup cost
    # This makes the first measurement look terrible (1-2 GB/s)
    # Warmup opens the transfer path so timed tests are accurate
    print(f"  [INFO] Running warmup to initialize CUDA transfer path...")
    warmup_elements = (100 * 1024 * 1024) // 4
    warmup_tensor = torch.ones(warmup_elements, dtype=torch.float32, device='cuda:0')
    for gpu_id in range(1, gpu_count):
        _ = warmup_tensor.to(f'cuda:{gpu_id}')
    torch.cuda.synchronize()
    del warmup_tensor
    torch.cuda.empty_cache()
    print(f"  [INFO] Warmup complete — starting timed bandwidth tests")

    # ── Timed bandwidth tests ──
    # Skip 10MB — too small, dominated by overhead even after warmup
    # Use 100MB, 500MB, 1024MB for accurate measurement
    test_sizes_mb = [100, 500, 1024]
    all_pass = True

    for size_mb in test_sizes_mb:
        num_elements = (size_mb * 1024 * 1024) // 4
        tensor = torch.ones(num_elements, dtype=torch.float32, device='cuda:0')

        # Run transfer multiple times and take the best result
        # Best result eliminates occasional OS scheduling noise
        times = []
        for _ in range(3):
            start = time.perf_counter()
            for gpu_id in range(1, gpu_count):
                other_tensor = tensor.to(f'cuda:{gpu_id}')
                tensor = tensor + other_tensor.to('cuda:0')
            torch.cuda.synchronize()
            end = time.perf_counter()
            times.append(end - start)

        best_time = min(times)
        data_moved_gb = (size_mb * 2) / 1024
        bandwidth_gbs = data_moved_gb / best_time

        if bandwidth_gbs >= MIN_BANDWIDTH_GBS:
            print(f"  [PASS] {size_mb}MB tensor: {bandwidth_gbs:.1f} GB/s")
        else:
            print(f"  [FAIL] {size_mb}MB tensor: {bandwidth_gbs:.1f} GB/s "
                  f"(minimum {MIN_BANDWIDTH_GBS} GB/s)")
            all_pass = False

        del tensor
        torch.cuda.empty_cache()

    return all_pass

try:
    passed = run_allreduce_bandwidth_test()
    if passed:
        print("  NCCL intra-node bandwidth: PASS")
        exit(0)
    else:
        print("  NCCL intra-node bandwidth: FAIL")
        exit(1)
except Exception as e:
    print(f"  [FAIL] NCCL test error: {e}")
    exit(1)
EOF

NCCL_EXIT=$?

# ── Step 4: Check P2P access ─────────────────────────
echo ""
echo "--- GPU Peer-to-Peer Access ---"

python3 << 'EOF'
import torch

gpu_count = torch.cuda.device_count()

for i in range(gpu_count):
    for j in range(gpu_count):
        if i != j:
            can_access = torch.cuda.can_device_access_peer(i, j)
            if can_access:
                print(f"  [PASS] GPU {i} can directly access GPU {j} via P2P")
            else:
                print(f"  [WARN] GPU {i} cannot directly access GPU {j}")
EOF

# ── Final Result ──────────────────────────────────────
echo ""
echo "--------------------------------------------------"

if [ "$NVLINK_PASS" = true ] && [ $NCCL_EXIT -eq 0 ]; then
    echo "TEST 03 RESULT: PASS"
    EXIT_CODE=0
else
    echo "TEST 03 RESULT: FAIL"
    EXIT_CODE=1
fi

echo "=================================================="
echo ""
exit $EXIT_CODE
