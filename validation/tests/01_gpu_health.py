#!/usr/bin/env python3
"""
Test 01: GPU Health Check
Validates each GPU is recognized, has correct memory,
and is running at expected specs for H200 nodes.
"""

import subprocess
import sys

# H200 thresholds — hardcoded for this cluster
MIN_GPU_MEMORY_GB = 140        # H200 has 141GB
EXPECTED_GPU_NAME = "H200"
MAX_GPU_TEMP_C = 85            # above this is dangerous
MIN_GPUS_PER_NODE = 2          # we request 2 per node in sbatch

def run_command(cmd):
    """Run a shell command and return output."""
    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True
    )
    return result.stdout.strip()

def check_gpu_count():
    """Check that expected number of GPUs are visible."""
    output = run_command("nvidia-smi --list-gpus")
    gpu_count = len(output.strip().split("\n")) if output else 0

    if gpu_count >= MIN_GPUS_PER_NODE:
        print(f"  [PASS] GPU count: {gpu_count} GPUs visible")
        return True
    else:
        print(f"  [FAIL] GPU count: expected >={MIN_GPUS_PER_NODE}, got {gpu_count}")
        return False

def check_gpu_model():
    """Check that GPUs are the expected H200 model."""
    output = run_command(
        "nvidia-smi --query-gpu=name --format=csv,noheader"
    )
    gpu_names = output.strip().split("\n") if output else []
    all_pass = True

    for i, name in enumerate(gpu_names):
        if EXPECTED_GPU_NAME in name:
            print(f"  [PASS] GPU {i} model: {name}")
        else:
            print(f"  [FAIL] GPU {i} model: expected {EXPECTED_GPU_NAME}, got {name}")
            all_pass = False

    return all_pass

def check_gpu_memory():
    """Check each GPU has at least 140GB memory."""
    output = run_command(
        "nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits"
    )
    memory_values = output.strip().split("\n") if output else []
    all_pass = True

    for i, mem_mb in enumerate(memory_values):
        mem_gb = int(mem_mb.strip()) / 1024
        if mem_gb >= MIN_GPU_MEMORY_GB:
            print(f"  [PASS] GPU {i} memory: {mem_gb:.1f} GB")
        else:
            print(f"  [FAIL] GPU {i} memory: expected >={MIN_GPU_MEMORY_GB}GB, got {mem_gb:.1f}GB")
            all_pass = False

    return all_pass

def check_gpu_temperature():
    """Check GPUs are not overheating."""
    output = run_command(
        "nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits"
    )
    temps = output.strip().split("\n") if output else []
    all_pass = True

    for i, temp in enumerate(temps):
        temp_c = int(temp.strip())
        if temp_c <= MAX_GPU_TEMP_C:
            print(f"  [PASS] GPU {i} temperature: {temp_c}C")
        else:
            print(f"  [FAIL] GPU {i} temperature: {temp_c}C exceeds max {MAX_GPU_TEMP_C}C")
            all_pass = False

    return all_pass

def check_driver_version():
    """Check NVIDIA driver is installed and report version."""
    output = run_command(
        "nvidia-smi --query-gpu=driver_version --format=csv,noheader"
    )
    versions = output.strip().split("\n") if output else []

    if versions and versions[0]:
        print(f"  [PASS] Driver version: {versions[0]}")
        return True
    else:
        print(f"  [FAIL] Could not retrieve driver version")
        return False

def check_gpu_utilization():
    """Check baseline GPU utilization before training."""
    output = run_command(
        "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits"
    )
    utils = output.strip().split("\n") if output else []

    for i, util in enumerate(utils):
        print(f"  [INFO] GPU {i} current utilization: {util.strip()}%")

    # just informational — not a pass/fail
    return True

def main():
    print("\n" + "="*50)
    print("TEST 01: GPU Health Check")
    print("="*50)

    results = []
    results.append(check_gpu_count())
    results.append(check_gpu_model())
    results.append(check_gpu_memory())
    results.append(check_gpu_temperature())
    results.append(check_driver_version())
    results.append(check_gpu_utilization())

    all_passed = all(results)

    print("-"*50)
    if all_passed:
        print("TEST 01 RESULT: PASS")
    else:
        print("TEST 01 RESULT: FAIL")
    print("="*50 + "\n")

    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
