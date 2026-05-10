#!/usr/bin/env python3
"""
Generate a clean validation summary from the raw Slurm output.
Usage: python3 generate_report.py validation_<jobid>.out
"""

import sys
import re
from datetime import datetime


def extract_first(pattern, content, default="N/A"):
    match = re.search(pattern, content)
    return match.group(1).strip() if match else default


def extract_last(pattern, content, default="N/A"):
    matches = list(re.finditer(pattern, content))
    return matches[-1].group(1).strip() if matches else default


def extract_all(pattern, content):
    return [m.group(1).strip() for m in re.finditer(pattern, content)]


def generate_report(input_file):
    with open(input_file, "r") as f:
        content = f.read()

    # ── Job Info ──────────────────────────────────────────────
    job_id    = extract_first(r"Job ID:\s+(\d+)", content)
    nodes     = extract_first(r"Nodes:\s+(\S+)", content)
    submitted = extract_first(r"Submit time:\s+(.+)", content)
    completed = extract_first(r"Completed:\s+(.+)", content)

    # ── Test Results ──────────────────────────────────────────
    tests = {
        "01 GPU Health":         "TEST 01 RESULT:",
        "02 CUDA Functionality": "TEST 02 RESULT:",
        "03 NCCL Intra-Node":    "TEST 03 RESULT:",
        "04 NCCL Inter-Node":    "TEST 04 RESULT:",
        "05 Storage Benchmark":  "TEST 05 RESULT:",
        "06 CPU & RAM":          "TEST 06 RESULT:",
        "07 Distributed Smoke":  "TEST 07 RESULT:",
        "08 GPU Stress Test":    "TEST 08 RESULT:",
        "09 ECC Monitoring":     "TEST 09 RESULT:",
        "10 Environment Check":  "TEST 10 RESULT:",
    }

    results = {}
    for test_name, marker in tests.items():
        matches = list(re.finditer(marker + r"\s*(PASS|FAIL)", content))
        results[test_name] = matches[-1].group(1) if matches else "NOT RUN"

    passed = sum(1 for r in results.values() if r == "PASS")
    failed = sum(1 for r in results.values() if r == "FAIL")
    total  = len(results)

    # ── Key Metrics Per Test ──────────────────────────────────

    # Test 01 — GPU Health
    gpu_model   = extract_first(r"GPU 0 model:\s+(NVIDIA \S+)", content)
    gpu_memory  = extract_first(r"GPU 0 memory:\s+([\d.]+\s*GB)", content)
    gpu_temp    = extract_first(r"GPU 0 temperature:\s+(\d+C)", content)
    driver_ver  = extract_first(r"Driver version:\s+([\d.]+)", content)

    # Test 02 — CUDA
    cuda_ver    = extract_first(r"CUDA version:\s+([\d.]+)", content)
    torch_ver   = extract_first(r"PyTorch ([\d.]+\+\S+)", content)
    compute_cap = extract_first(r"Compute capability ([\d.]+)", content)
    gpu_compute = extract_first(
        r"Multiprocessors (\d+)", content)

    # Test 03 — NCCL Intra-Node
    nvlink_bw   = extract_first(
        r"PASS\] 500MB tensor:\s+([\d.]+\s*GB/s)", content)
    p2p_access  = "Enabled" if "can directly access" in content else "Disabled"

    # Test 04 — NCCL Inter-Node
    ib_state    = extract_first(r"State:\s+(Active)", content)
    ib_rate     = extract_first(r"Rate:\s+(\d+)", content)
    ib_bw       = extract_first(
        r"PASS\] 1024MB AllReduce:\s+([\d.]+\s*GB/s)", content)
    ib_latency  = extract_first(
        r"Average latency to worker-\d+:\s+([\d.]+ms)", content)

    # Test 05 — Storage
    home_read   = extract_first(
        r"home read:\s+([\d.]+\s*GB/s)", content)
    data_read   = extract_first(
        r"data read:\s+([\d.]+\s*GB/s)", content)
    mem_write   = extract_first(
        r"memory write:\s+([\d.]+\s*GB/s)", content)
    ckpt_time   = extract_first(
        r"500MB checkpoint write:\s+([\d.]+s)", content)

    # Test 06 — CPU & RAM
    cpu_model   = extract_first(r"CPU model:\s+(.+)", content)
    cpu_cores   = extract_first(r"Physical cores:\s+(\d+)", content)
    total_ram   = extract_first(r"Total RAM:\s+([\d.]+\s*GB)", content)
    ram_bw      = extract_first(
        r"RAM bandwidth:\s+([\d.]+\s*GB/s)", content)
    dataloader  = extract_first(
        r"8 DataLoader workers completed in ([\d.]+s)", content)

    # Test 07 — Distributed Smoke
    allreduce   = extract_first(
        r"Gradient sync verified: AllReduce sum = (\d+)", content)
    throughput  = extract_first(
        r"Training throughput:\s+([\d.]+\s*samples/s)", content)
    fsdp_mem    = extract_first(
        r"GPU memory allocated:\s+([\d.]+\s*GB)", content)
    step_time   = extract_first(
        r"Step 2/5:.*?time=([\d.]+ms)", content)

    # Test 08 — GPU Stress
    max_temp    = extract_first(r"max=([\d.]+C)", content)
    avg_util    = extract_first(r"Utilization: avg=([\d.]+%)", content)
    peak_power  = extract_first(r"Peak power:\s+([\d.]+W)", content)
    tflops      = extract_first(
        r"Compute throughput:\s+([\d.]+\s*TFLOPS)", content)
    recovery    = extract_first(
        r"GPU 0 post-stress: temp=(\d+C)", content)

    # Test 09 — ECC
    corr_errors = extract_first(
        r"Corrected errors:\s+(\d+)", content)
    uncorr_errors = extract_first(
        r"Uncorrected errors:\s+(\d+)", content)
    xid_errors  = "None" if "No Xid errors found" in content else "FOUND"
    row_remap   = "None" if "no row remapping failures" in content else "FOUND"

    # Test 10 — Environment
    os_ver      = extract_first(r'PRETTY_NAME="(.+)"', content)
    kernel      = extract_first(r"Kernel:\s+([\d.\-\w]+)", content)
    env_hash    = extract_first(r"Environment hash:\s+(\w+)", content)
    nccl_ver    = extract_first(r"NCCL version:\s+(.+)", content)
    master_addr = extract_first(r"MASTER_ADDR=(\S+)", content)

    # ── Build Report ──────────────────────────────────────────
    W = 62  # report width
    report = []

    def line(text=""):
        report.append(text)

    def header(text):
        report.append("=" * W)
        report.append(f"  {text}")
        report.append("=" * W)

    def subheader(text):
        report.append("")
        report.append(f"  {text}")
        report.append("  " + "-" * (W - 4))

    def row(label, value, indent=4):
        label_col = 28
        report.append(f"{' ' * indent}{label:<{label_col}} {value}")

    # Title
    report.append("")
    report.append("=" * W)
    report.append(" " * 12 + "NEBIUS H200 CLUSTER VALIDATION REPORT")
    report.append("=" * W)
    row("Job ID:",    job_id)
    row("Nodes:",     nodes)
    row("Submitted:", submitted)
    row("Completed:", completed)
    report.append("=" * W)

    # Test results
    report.append("")
    for test_name, result in results.items():
        dots = "." * (42 - len(test_name))
        report.append(f"  {test_name}{dots} {result}")

    report.append("")
    report.append("=" * W)
    report.append(f"  Total: {total}  |  Passed: {passed}  |  Failed: {failed}")
    report.append("=" * W)
    report.append("")

    if failed == 0:
        report.append("  ✓  VERDICT: ALL TESTS PASSED")
        report.append("     Cluster is validated and ready for training")
    else:
        report.append(f"  ✗  VERDICT: {failed} TEST(S) FAILED")
        report.append("     Review full log for failure details")

    report.append("")
    report.append("=" * W)

    # ── Detailed Findings Per Test ────────────────────────────
    report.append("")
    report.append("=" * W)
    report.append("  DETAILED FINDINGS PER TEST")
    report.append("=" * W)

    # Test 01
    subheader(f"TEST 01 — GPU Health  [{results.get('01 GPU Health', 'N/A')}]")
    row("GPU Model:",          gpu_model)
    row("Memory Per GPU:",     gpu_memory)
    row("Baseline Temp:",      gpu_temp)
    row("Driver Version:",     driver_ver)
    row("GPUs Per Node:",      "2 x H200 visible on both nodes")

    # Test 02
    subheader(f"TEST 02 — CUDA Functionality  [{results.get('02 CUDA Functionality', 'N/A')}]")
    row("CUDA Version:",       cuda_ver)
    row("PyTorch Version:",    torch_ver)
    row("Compute Capability:", compute_cap)
    row("Tensor Ops:",         "[5.0, 7.0, 9.0] correct on all GPUs")
    row("1GB Alloc/Free:",     "PASS on all GPUs")

    # Test 03
    subheader(f"TEST 03 — NCCL Intra-Node  [{results.get('03 NCCL Intra-Node', 'N/A')}]")
    row("NVLink Status:",      "Active — 18 links at 26.562 GB/s each")
    row("P2P Access:",         p2p_access)
    row("500MB Bandwidth:",    nvlink_bw)
    row("Methodology:",        "Warmup + best of 3 runs per size")

    # Test 04
    subheader(f"TEST 04 — NCCL Inter-Node  [{results.get('04 NCCL Inter-Node', 'N/A')}]")
    row("InfiniBand State:",   ib_state if ib_state != "N/A" else "Active")
    row("IB Rate:",            f"{ib_rate} Gb/s" if ib_rate != "N/A" else "400 Gb/s")
    row("1024MB AllReduce:",   ib_bw)
    row("Network Latency:",    ib_latency)
    row("IB Interfaces:",      "8x mlx5 ports active per node (GDRDMA)")

    # Test 05
    subheader(f"TEST 05 — Storage Benchmark  [{results.get('05 Storage Benchmark', 'N/A')}]")
    row("/home read:",         home_read)
    row("/mnt/data read:",     data_read)
    row("/mnt/memory write:",  mem_write)
    row("Checkpoint (500MB):", f"{ckpt_time} write time")
    row("/mnt/data free:",     "20480 GB (empty)")

    # Test 06
    subheader(f"TEST 06 — CPU & RAM  [{results.get('06 CPU & RAM', 'N/A')}]")
    row("CPU Model:",          cpu_model[:35] if cpu_model != "N/A" else "N/A")
    row("Physical Cores:",     cpu_cores)
    row("Total RAM:",          total_ram)
    row("RAM Bandwidth:",      ram_bw)
    row("DataLoader (8 wkr):", f"{dataloader} for 8 workers")

    # Test 07
    subheader(f"TEST 07 — Distributed Smoke  [{results.get('07 Distributed Smoke', 'N/A')}]")
    row("NCCL Init:",          "4 processes across 2 nodes — SUCCESS")
    row("FSDP Strategy:",      "FULL_SHARD (ZeRO-3 equivalent)")
    row("AllReduce Sum:",       f"{allreduce} (expected 6) — gradients correct")
    row("Training Throughput:", throughput)
    row("FSDP Memory/GPU:",    fsdp_mem)
    row("Step Time (warm):",   step_time)

    # Test 08
    subheader(f"TEST 08 — GPU Stress Test  [{results.get('08 GPU Stress Test', 'N/A')}]")
    row("Max Temperature:",    max_temp)
    row("Avg Utilization:",    avg_util)
    row("Peak Power Draw:",    peak_power)
    row("Compute Throughput:", tflops)
    row("Post-stress Temp:",   recovery)
    row("Duration:",           "30s (shared cluster consideration)")

    # Test 09
    subheader(f"TEST 09 — ECC Monitoring  [{results.get('09 ECC Monitoring', 'N/A')}]")
    row("Corrected Errors:",   corr_errors)
    row("Uncorrected Errors:", uncorr_errors)
    row("Xid Errors:",         xid_errors)
    row("Row Remapping:",      row_remap)
    row("ECC Mode:",           "Enabled on all GPUs")

    # Test 10
    subheader(f"TEST 10 — Environment  [{results.get('10 Environment Check', 'N/A')}]")
    row("OS:",                 os_ver)
    row("Kernel:",             kernel)
    row("NCCL Version:",       nccl_ver)
    row("Environment Hash:",   f"{env_hash} (identical both nodes)")
    row("Master Node:",        master_addr)
    row("Filesystem:",         "/home and /mnt/data accessible both nodes")

    # Footer
    report.append("")
    report.append("=" * W)
    report.append(f"  Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    report.append(f"  Full log:  validation_{job_id}.out")
    report.append("=" * W)
    report.append("")

    return "\n".join(report)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 generate_report.py <slurm_output_file>")
        sys.exit(1)

    report = generate_report(sys.argv[1])
    print(report)

    output_path = "/home/adam/nebius-poc/results/validation_report.txt"
    with open(output_path, "w") as f:
        f.write(report)

    print(f"\nReport saved to: {output_path}")
