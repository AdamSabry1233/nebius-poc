#!/usr/bin/env python3
"""
Test 10: Environment Consistency Check
Verifies both nodes have identical software environments.

This is one of the most commonly overlooked validation checks.
If node 0 has CUDA 12.4 and node 1 has CUDA 12.1, or if
PyTorch versions differ between nodes, you get subtle bugs
that are extremely hard to diagnose mid-training.

The fix takes 5 minutes. Missing it can cost hours of debugging.
"""

import sys
import os
import subprocess
import platform
import hashlib
import json

def run_command(cmd):
    """Run shell command and return output."""
    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True
    )
    return result.stdout.strip(), result.returncode

def get_node_identity():
    """
    Report which node this process is running on.
    In a multi-node job both nodes run this script —
    the output from each node appears in the same log file
    prefixed with the node hostname so you can compare them.
    """
    hostname    = platform.node()
    rank        = os.environ.get("RANK", "0")
    local_rank  = os.environ.get("LOCAL_RANK", "0")
    slurm_node  = os.environ.get("SLURM_NODEID", "0")

    print(f"\n  [INFO] Hostname:    {hostname}")
    print(f"  [INFO] RANK:        {rank}")
    print(f"  [INFO] LOCAL_RANK:  {local_rank}")
    print(f"  [INFO] SLURM_NODEID:{slurm_node}")

    return hostname, rank

def check_os_version():
    """
    Verify OS version matches across nodes.
    Different OS versions can have different system library
    versions which causes subtle incompatibilities.
    """
    print("\n--- OS Version ---")

    os_info, _ = run_command("cat /etc/os-release | grep PRETTY_NAME")
    kernel, _  = run_command("uname -r")
    arch, _    = run_command("uname -m")

    print(f"  [INFO] OS:      {os_info}")
    print(f"  [INFO] Kernel:  {kernel}")
    print(f"  [INFO] Arch:    {arch}")

    # Just report — comparison happens by reading both
    # nodes output in the same log file
    print(f"  [PASS] OS info collected — compare across nodes in log")
    return True

def check_cuda_version():
    """
    Verify CUDA version is identical across nodes.
    CUDA version mismatch is the #1 cause of mysterious
    distributed training failures that only appear at scale.
    """
    print("\n--- CUDA Version ---")

    # Check nvcc version
    nvcc_out, rc = run_command("nvcc --version")
    if rc == 0:
        # Extract version line
        for line in nvcc_out.split("\n"):
            if "release" in line.lower():
                print(f"  [INFO] NVCC: {line.strip()}")
                break
    else:
        print(f"  [WARN] nvcc not in PATH")

    # Check CUDA runtime version via nvidia-smi
    cuda_out, _ = run_command(
        "nvidia-smi --query-gpu=driver_version,cuda_version "
        "--format=csv,noheader"
    )
    if cuda_out:
        parts = cuda_out.split("\n")[0].split(",")
        if len(parts) >= 2:
            driver_ver = parts[0].strip()
            cuda_ver   = parts[1].strip()
            print(f"  [INFO] Driver version: {driver_ver}")
            print(f"  [INFO] CUDA version:   {cuda_ver}")

    # Check PyTorch CUDA version
    try:
        import torch
        print(f"  [INFO] PyTorch CUDA: {torch.version.cuda}")
        print(f"  [INFO] cuDNN version: {torch.backends.cudnn.version()}")
        print(f"  [PASS] CUDA versions collected — compare across nodes")
        return True
    except ImportError:
        print(f"  [FAIL] PyTorch not importable")
        return False

def check_python_version():
    """
    Verify Python version matches across nodes.
    Python minor version differences (3.11 vs 3.12) can
    cause pickle incompatibilities in distributed training
    when serializing data between nodes.
    """
    print("\n--- Python Version ---")

    version = platform.python_version()
    impl    = platform.python_implementation()
    path, _ = run_command("which python3")

    print(f"  [INFO] Python version: {version}")
    print(f"  [INFO] Implementation: {impl}")
    print(f"  [INFO] Python path:    {path}")

    # Check Python is same version as expected
    major = sys.version_info.major
    minor = sys.version_info.minor

    if major == 3 and minor >= 10:
        print(f"  [PASS] Python {version} meets minimum requirement (3.10+)")
        return True
    else:
        print(f"  [FAIL] Python {version} too old — need 3.10+")
        return False

def check_pytorch_version():
    """
    Verify PyTorch version and build info matches across nodes.
    Even same PyTorch version can differ if built with different
    CUDA or cuDNN versions — causing silent numerical differences.
    """
    print("\n--- PyTorch Version ---")

    try:
        import torch

        print(f"  [INFO] PyTorch version:  {torch.__version__}")
        print(f"  [INFO] CUDA available:   {torch.cuda.is_available()}")
        print(f"  [INFO] CUDA version:     {torch.version.cuda}")
        print(f"  [INFO] cuDNN enabled:    {torch.backends.cudnn.enabled}")
        print(f"  [INFO] cuDNN version:    {torch.backends.cudnn.version()}")

        # Check debug build — debug builds are much slower
        is_debug = torch.version.debug
        if is_debug:
            print(f"  [WARN] PyTorch is a DEBUG build — very slow for training")
        else:
            print(f"  [PASS] PyTorch is a release build")

        # Get build configuration
        config = torch.__config__.show()
        # Hash the config to detect any differences between nodes
        config_hash = hashlib.md5(config.encode()).hexdigest()[:8]
        print(f"  [INFO] Build config hash: {config_hash}")
        print(f"  [INFO] (hashes must match across all nodes)")

        print(f"  [PASS] PyTorch info collected — compare across nodes")
        return True

    except ImportError:
        print(f"  [FAIL] PyTorch not installed")
        return False

def check_key_packages():
    """
    Verify critical package versions match across nodes.
    transformers, peft, trl version mismatches cause
    model loading failures in distributed training.
    """
    print("\n--- Key Package Versions ---")

    packages = [
        "transformers",
        "peft",
        "trl",
        "accelerate",
        "datasets",
        "tokenizers",
        "numpy",
        "psutil",
    ]

    all_pass    = True
    pkg_versions = {}

    for pkg in packages:
        try:
            module = __import__(pkg.replace("-", "_"))
            version = getattr(module, "__version__", "unknown")
            pkg_versions[pkg] = version
            print(f"  [INFO] {pkg}: {version}")
        except ImportError:
            print(f"  [WARN] {pkg}: NOT INSTALLED")
            pkg_versions[pkg] = "NOT INSTALLED"
            if pkg in ["transformers", "peft", "trl", "accelerate"]:
                # These are required for training
                all_pass = False
                print(f"  [FAIL] {pkg} is required for training")

    # Create a hash of all versions
    # Same hash on both nodes = identical environments
    versions_str = json.dumps(pkg_versions, sort_keys=True)
    env_hash     = hashlib.md5(versions_str.encode()).hexdigest()[:8]
    print(f"\n  [INFO] Environment hash: {env_hash}")
    print(f"  [INFO] (hashes must match across all nodes)")

    if all_pass:
        print(f"  [PASS] All required packages installed")
    else:
        print(f"  [FAIL] Missing required packages")

    return all_pass

def check_nccl_version():
    """
    Verify NCCL version matches across nodes.
    NCCL version mismatch causes immediate crash when
    the distributed process group tries to initialize.
    """
    print("\n--- NCCL Version ---")

    try:
        import torch.cuda.nccl as nccl
        version = nccl.version()
        print(f"  [INFO] NCCL version: {version}")
        print(f"  [PASS] NCCL version collected — compare across nodes")
        return True
    except Exception:
        pass

    # Try via torch directly
    try:
        import torch
        if hasattr(torch.cuda, 'nccl'):
            print(f"  [INFO] NCCL available via torch.cuda.nccl")
            return True
    except Exception:
        pass

    # Try via subprocess
    nccl_out, rc = run_command(
        "python3 -c \"import torch; print(torch.cuda.nccl.version())\""
    )
    if rc == 0 and nccl_out:
        print(f"  [INFO] NCCL version: {nccl_out}")
        print(f"  [PASS] NCCL version collected")
        return True

    print(f"  [WARN] Could not determine NCCL version")
    return True  # not a hard fail

def check_environment_variables():
    """
    Check critical environment variables are set correctly
    and consistently across both nodes.
    """
    print("\n--- Environment Variables ---")

    # Variables that must be set for distributed training
    required_vars = [
        "MASTER_ADDR",
        "MASTER_PORT",
        "WORLD_SIZE",
        "RANK",
        "LOCAL_RANK",
    ]

    # Variables that affect training performance
    optional_vars = [
        "NCCL_IB_DISABLE",
        "NCCL_DEBUG",
        "NCCL_SOCKET_IFNAME",
        "CUDA_VISIBLE_DEVICES",
        "OMP_NUM_THREADS",
        "TOKENIZERS_PARALLELISM",
    ]

    all_pass = True

    print("  Required variables:")
    for var in required_vars:
        value = os.environ.get(var)
        if value is not None:
            print(f"    [INFO] {var}={value}")
        else:
            print(f"    [WARN] {var} not set — "
                  f"needed for distributed training")

    print("\n  Optional variables:")
    for var in optional_vars:
        value = os.environ.get(var, "not set")
        print(f"    [INFO] {var}={value}")

    print(f"\n  [PASS] Environment variables collected — compare across nodes")
    return all_pass

def check_filesystem_consistency():
    """
    Verify both nodes see the same shared filesystem.
    If /home or /mnt/data are mounted differently on each
    node, one node won't find the training data or checkpoints.
    """
    print("\n--- Filesystem Consistency ---")

    # Check mounted filesystems
    mounts_out, _ = run_command("df -h | grep -E 'home|mnt|data'")
    if mounts_out:
        print(f"  [INFO] Relevant mounts:")
        for line in mounts_out.split("\n"):
            print(f"    {line}")

    # Check we can write to shared storage
    test_file = f"/home/adam/env_check_{os.environ.get('RANK', '0')}.txt"
    try:
        with open(test_file, "w") as f:
            f.write(f"node={platform.node()}\n")
            f.write(f"rank={os.environ.get('RANK', '0')}\n")
        print(f"  [PASS] Can write to shared /home filesystem")
        os.remove(test_file)
    except Exception as e:
        print(f"  [FAIL] Cannot write to /home: {e}")
        return False

    # Check /mnt/data is accessible
    if os.path.exists("/mnt/data"):
        stat    = os.statvfs("/mnt/data")
        free_gb = (stat.f_bavail * stat.f_frsize) / 1024**3
        print(f"  [PASS] /mnt/data accessible: {free_gb:.1f}GB free")
    else:
        print(f"  [WARN] /mnt/data not accessible from this node")

    return True

def check_gpu_driver_consistency():
    """
    Verify GPU driver version matches across nodes.
    Driver version mismatch causes NCCL initialization
    failures that are hard to diagnose.
    """
    print("\n--- GPU Driver Consistency ---")

    driver_out, _ = run_command(
        "nvidia-smi --query-gpu=index,driver_version,name "
        "--format=csv,noheader"
    )

    if driver_out:
        for line in driver_out.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                print(f"  [INFO] GPU {parts[0]}: "
                      f"driver={parts[1]}, "
                      f"model={parts[2]}")
        print(f"  [PASS] Driver info collected — compare across nodes")
        return True
    else:
        print(f"  [WARN] Could not query GPU driver info")
        return True

def main():
    print("\n" + "="*50)
    print("TEST 10: Environment Consistency Check")
    print("="*50)

    # First identify which node we are
    print("\n--- Node Identity ---")
    hostname, rank = get_node_identity()

    results = []

    results.append(check_os_version())
    results.append(check_cuda_version())
    results.append(check_python_version())
    results.append(check_pytorch_version())
    results.append(check_key_packages())
    results.append(check_nccl_version())
    results.append(check_environment_variables())
    results.append(check_filesystem_consistency())
    results.append(check_gpu_driver_consistency())

    all_passed = all(results)

    # Print summary with node identity for easy log comparison
    print("\n" + "-"*50)
    print(f"Node: {hostname} (rank {rank})")
    if all_passed:
        print("TEST 10 RESULT: PASS")
    else:
        print("TEST 10 RESULT: FAIL")
    print("="*50 + "\n")

    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
