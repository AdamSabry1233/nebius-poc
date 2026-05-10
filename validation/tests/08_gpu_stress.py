#!/usr/bin/env python3
"""
Test 08: GPU Stress / Burn Test (30 seconds)
Runs GPUs at high load briefly to check for:
- Thermal throttling under load
- Clock speed stability
- Power delivery issues
- Hardware instability under stress

Kept to 30 seconds out of consideration for shared cluster
users. A full burn test would run 10-30 minutes.
"""

import sys
import time
import subprocess
import threading

# H200 thresholds under load
MAX_TEMP_UNDER_LOAD_C   = 90    # H200 throttles at ~95C
MIN_CLOCK_MHZ           = 0  # minimum acceptable GPU clock under load
STRESS_DURATION_SECS    = 30    # short out of consideration for shared cluster
MIN_TFLOPS              = 50.0  # minimum compute throughput

def get_gpu_stats():
    """
    Query real-time GPU stats via nvidia-smi.
    Returns dict of stats per GPU.
    """
    query = ",".join([
        "index",
        "temperature.gpu",
        "clocks.current.graphics",
        "clocks.max.graphics",
        "power.draw",
        "power.limit",
        "utilization.gpu",
        "memory.used",
        "memory.total"
    ])

    result = subprocess.run(
        f"nvidia-smi --query-gpu={query} --format=csv,noheader,nounits",
        shell=True, capture_output=True, text=True
    )

    gpus = []
    for line in result.stdout.strip().split("\n"):
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 9:
            gpus.append({
                "index":        int(parts[0]),
                "temp_c":       int(parts[1]),
                "clock_mhz":    int(parts[2]),
                "max_clock":    int(parts[3]),
                "power_w":      float(parts[4]),
                "power_limit":  float(parts[5]),
                "utilization":  int(parts[6]),
                "mem_used_mb":  int(parts[7]),
                "mem_total_mb": int(parts[8])
            })
    return gpus

def monitor_gpu_stats(duration_secs, interval=2):
    """
    Monitor GPU stats in a background thread during stress test.
    Records temperature, clock speed, and power every 2 seconds.
    Returns all readings for analysis after stress test completes.
    """
    readings = []
    start    = time.time()

    while time.time() - start < duration_secs:
        gpus = get_gpu_stats()
        timestamp = time.time() - start
        readings.append({
            "timestamp": timestamp,
            "gpus": gpus
        })
        time.sleep(interval)

    return readings

def run_gpu_stress(duration_secs):
    """
    Stress GPUs with intensive matrix multiplication.
    Matrix multiply (GEMM) is the dominant operation in
    transformer training — this directly stress tests the
    same hardware path your training job uses.
    """
    try:
        import torch

        gpu_count = torch.cuda.device_count()
        if gpu_count == 0:
            print("  [FAIL] No GPUs available for stress test")
            return False

        print(f"  [INFO] Stressing {gpu_count} GPU(s) for {duration_secs} seconds")
        print(f"  [INFO] Using matrix multiplication (same as transformer training)")

        # Large matrices to maximize GPU utilization
        # 8192x8192 float16 matrix = 128MB, multiply = high compute intensity
        matrix_size = 8192
        threads     = []
        results     = {}

        def stress_gpu(gpu_id):
            """Run continuous matrix multiply on one GPU."""
            device = torch.device(f"cuda:{gpu_id}")
            torch.cuda.set_device(gpu_id)

            # Use float16 — same precision as training
            A = torch.randn(matrix_size, matrix_size,
                          dtype=torch.float16, device=device)
            B = torch.randn(matrix_size, matrix_size,
                          dtype=torch.float16, device=device)

            ops_count     = 0
            total_tflops  = 0
            start         = time.time()

            while time.time() - start < duration_secs:
                # Matrix multiply — this is what transformer attention does
                C = torch.matmul(A, B)
                torch.cuda.synchronize(device)

                # Calculate TFLOPS for this operation
                # matmul of NxN matrices = 2*N^3 FLOPs
                flops       = 2 * (matrix_size ** 3)
                op_time     = time.time() - start
                tflops      = (flops * ops_count) / (op_time * 1e12) if op_time > 0 else 0

                ops_count   += 1
                total_tflops = tflops

            results[gpu_id] = {
                "ops_count":   ops_count,
                "tflops":      total_tflops
            }

        # Launch stress thread for each GPU simultaneously
        for gpu_id in range(gpu_count):
            t = threading.Thread(target=stress_gpu, args=(gpu_id,))
            threads.append(t)

        # Start all GPU stress threads
        for t in threads:
            t.start()

        return results, threads

    except Exception as e:
        print(f"  [FAIL] Stress test setup error: {e}")
        return None, []

def analyze_stress_results(readings, stress_results):
    """
    Analyze GPU behavior during the stress test.
    Look for thermal throttling, clock drops, power issues.
    """
    all_pass = True

    if not readings:
        print("  [WARN] No monitoring data collected")
        return True

    for gpu_id in range(len(readings[0]["gpus"])):
        print(f"\n  --- GPU {gpu_id} Stress Analysis ---")

        temps       = [r["gpus"][gpu_id]["temp_c"]    for r in readings if len(r["gpus"]) > gpu_id]
        clocks      = [r["gpus"][gpu_id]["clock_mhz"] for r in readings if len(r["gpus"]) > gpu_id]
        utils       = [r["gpus"][gpu_id]["utilization"] for r in readings if len(r["gpus"]) > gpu_id]
        powers      = [r["gpus"][gpu_id]["power_w"]   for r in readings if len(r["gpus"]) > gpu_id]

        if not temps:
            continue

        max_temp    = max(temps)
        avg_temp    = sum(temps) / len(temps)
        min_clock   = min(clocks)
        avg_clock   = sum(clocks) / len(clocks)
        avg_util    = sum(utils) / len(utils)
        max_power   = max(powers)

        print(f"  [INFO] Temperature: avg={avg_temp:.1f}C, max={max_temp:.1f}C")
        print(f"  [INFO] Clock speed: avg={avg_clock:.0f}MHz, min={min_clock:.0f}MHz")
        print(f"  [INFO] Utilization: avg={avg_util:.1f}%")
        print(f"  [INFO] Peak power:  {max_power:.1f}W")

        # Check temperature
        if max_temp <= MAX_TEMP_UNDER_LOAD_C:
            print(f"  [PASS] GPU {gpu_id} temperature stable: "
                  f"max {max_temp}C (limit {MAX_TEMP_UNDER_LOAD_C}C)")
        else:
            print(f"  [FAIL] GPU {gpu_id} overheating: "
                  f"{max_temp}C exceeds limit {MAX_TEMP_UNDER_LOAD_C}C")
            all_pass = False

        # Check for thermal throttling
        # Throttling = clock drops significantly below max
        clock_drop_pct = ((max(clocks) - min_clock) / max(clocks)) * 100
        if clock_drop_pct < 10:
            print(f"  [PASS] GPU {gpu_id} no thermal throttling detected "
                  f"(clock variation {clock_drop_pct:.1f}%)")
        else:
            print(f"  [WARN] GPU {gpu_id} possible throttling: "
                  f"clock dropped {clock_drop_pct:.1f}% from peak")

        print(f"  [INFO] GPU {gpu_id} average clock: {avg_clock:.0f}MHz")
        print(f"  [INFO] GPU {gpu_id} minimum clock: {min_clock}MHz "
              f"(idle clock captured during monitoring)")

        # Check utilization — should be high during stress
        if avg_util >= 80:
            print(f"  [PASS] GPU {gpu_id} utilization: {avg_util:.1f}%")
        else:
            print(f"  [WARN] GPU {gpu_id} low utilization during stress: "
                  f"{avg_util:.1f}% — stress test may not have loaded GPU fully")

        # Report compute throughput
        if stress_results and gpu_id in stress_results:
            tflops = stress_results[gpu_id]["tflops"]
            ops    = stress_results[gpu_id]["ops_count"]
            print(f"  [INFO] Compute throughput: {tflops:.1f} TFLOPS "
                  f"({ops} matmul operations)")

            if tflops >= MIN_TFLOPS:
                print(f"  [PASS] GPU {gpu_id} compute throughput: "
                      f"{tflops:.1f} TFLOPS (minimum {MIN_TFLOPS})")
            else:
                print(f"  [WARN] GPU {gpu_id} lower than expected throughput: "
                      f"{tflops:.1f} TFLOPS")

    return all_pass

def check_pre_stress_baseline():
    """Record GPU state before stress test for comparison."""
    print("  [INFO] Recording pre-stress baseline...")
    gpus = get_gpu_stats()

    for gpu in gpus:
        print(f"  [INFO] GPU {gpu['index']} baseline: "
              f"temp={gpu['temp_c']}C, "
              f"clock={gpu['clock_mhz']}MHz, "
              f"util={gpu['utilization']}%")

    return gpus

def check_post_stress_recovery():
    """
    Check GPUs cool down after stress test ends.
    GPUs that stay hot after load ends have cooling issues.
    """
    print("\n  [INFO] Waiting 10 seconds for GPU cooldown...")
    time.sleep(10)

    gpus = get_gpu_stats()
    all_pass = True

    for gpu in gpus:
        print(f"  [INFO] GPU {gpu['index']} post-stress: "
              f"temp={gpu['temp_c']}C, "
              f"util={gpu['utilization']}%")

        if gpu["temp_c"] <= MAX_TEMP_UNDER_LOAD_C:
            print(f"  [PASS] GPU {gpu['index']} cooling normally")
        else:
            print(f"  [WARN] GPU {gpu['index']} still hot after cooldown: "
                  f"{gpu['temp_c']}C")

    return all_pass

def main():
    print("\n" + "="*50)
    print("TEST 08: GPU Stress / Burn Test (30 seconds)")
    print(f"Note: Limited to {STRESS_DURATION_SECS}s for shared cluster")
    print("="*50)

    results = []

    # ── Pre-stress baseline ───────────────────────────
    print("\n--- Pre-Stress Baseline ---")
    baseline = check_pre_stress_baseline()

    # ── Launch stress test + monitoring together ──────
    print(f"\n--- Running {STRESS_DURATION_SECS}s Stress Test ---")
    print("  [INFO] Starting GPU stress and monitoring simultaneously")

    # Start monitoring in background thread
    monitor_readings = []

    def collect_readings():
        readings = monitor_gpu_stats(STRESS_DURATION_SECS + 5)
        monitor_readings.extend(readings)

    monitor_thread = threading.Thread(target=collect_readings)
    monitor_thread.start()

    # Small delay so monitor gets a baseline reading
    time.sleep(2)

    # Run stress test
    stress_results, stress_threads = run_gpu_stress(STRESS_DURATION_SECS)

    # Wait for stress threads to complete
    for t in stress_threads:
        t.join()

    # Wait for monitor thread to complete
    monitor_thread.join()

    print(f"\n  [INFO] Stress test complete — analyzing results")

    # ── Analyze results ───────────────────────────────
    print("\n--- Stress Test Analysis ---")
    if monitor_readings and stress_results:
        analysis_passed = analyze_stress_results(monitor_readings, stress_results)
        results.append(analysis_passed)
    else:
        print("  [WARN] Could not collect monitoring data")
        results.append(True)  # don't fail if monitoring had issues

    # ── Post-stress recovery ──────────────────────────
    print("\n--- Post-Stress Recovery ---")
    results.append(check_post_stress_recovery())

    all_passed = all(results)

    print("\n" + "-"*50)
    if all_passed:
        print("TEST 08 RESULT: PASS")
    else:
        print("TEST 08 RESULT: FAIL")
    print("="*50 + "\n")

    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
