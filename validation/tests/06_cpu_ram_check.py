#!/usr/bin/env python3
"""
Test 06: CPU & RAM Check
Validates CPU cores and RAM are sufficient for data preprocessing,
tokenization, and data loading during training.
Slow CPU or insufficient RAM = data starvation problem where
GPUs sit idle waiting for the CPU to feed them batches.
"""

import sys
import os
import time
import psutil
import multiprocessing

# H200 node thresholds
MIN_CPU_CORES        = 32      # physical cores minimum
MIN_RAM_GB           = 400     # GB minimum total RAM
MIN_FREE_RAM_GB      = 200     # GB minimum free RAM
MIN_RAM_BANDWIDTH    = 60     # GB/s minimum memory bandwidth

def check_cpu_count():
    """
    Check physical CPU core count.
    More cores = faster data preprocessing and tokenization.
    Data loading uses multiple workers — need enough cores
    to keep all GPU workers fed simultaneously.
    """
    physical_cores  = psutil.cpu_count(logical=False)
    logical_cores   = psutil.cpu_count(logical=True)

    print(f"  [INFO] Physical cores: {physical_cores}")
    print(f"  [INFO] Logical cores (with hyperthreading): {logical_cores}")

    # Also check via os and multiprocessing for verification
    os_cpu_count = os.cpu_count()
    mp_cpu_count = multiprocessing.cpu_count()
    print(f"  [INFO] os.cpu_count(): {os_cpu_count}")
    print(f"  [INFO] multiprocessing.cpu_count(): {mp_cpu_count}")

    if physical_cores >= MIN_CPU_CORES:
        print(f"  [PASS] CPU cores: {physical_cores} "
              f"(minimum {MIN_CPU_CORES})")
        return True
    else:
        print(f"  [FAIL] CPU cores: {physical_cores} "
              f"(minimum {MIN_CPU_CORES})")
        return False

def check_cpu_frequency():
    """
    Check CPU clock speed.
    Higher frequency = faster tokenization per core.
    """
    try:
        freq = psutil.cpu_freq()
        if freq:
            current_mhz  = freq.current
            max_mhz      = freq.max
            print(f"  [INFO] CPU frequency: {current_mhz:.0f} MHz current")
            print(f"  [INFO] CPU frequency: {max_mhz:.0f} MHz maximum")

            if current_mhz >= 1000:  # at least 1GHz
                print(f"  [PASS] CPU frequency acceptable")
                return True
            else:
                print(f"  [FAIL] CPU frequency too low: {current_mhz:.0f} MHz")
                return False
        else:
            print(f"  [INFO] CPU frequency info not available")
            return True  # not a hard fail
    except Exception as e:
        print(f"  [INFO] Could not check CPU frequency: {e}")
        return True

def check_cpu_utilization():
    """
    Check current CPU load before training starts.
    High baseline CPU usage means fewer cores available
    for data loading workers.
    """
    # Measure over 2 seconds for accurate reading
    cpu_percent = psutil.cpu_percent(interval=2, percpu=False)
    per_cpu     = psutil.cpu_percent(interval=0, percpu=True)

    print(f"  [INFO] Overall CPU utilization: {cpu_percent:.1f}%")

    # Check if any cores are maxed out
    maxed_cores = sum(1 for c in per_cpu if c > 90)
    if maxed_cores > 0:
        print(f"  [WARN] {maxed_cores} CPU cores at >90% utilization")
    
    if cpu_percent < 50:
        print(f"  [PASS] CPU has sufficient headroom for data loading")
        return True
    else:
        print(f"  [WARN] CPU already at {cpu_percent:.1f}% — "
              f"may impact data loading performance")
        return True  # warn but don't fail — other jobs may be running

def check_total_ram():
    """
    Check total RAM meets minimum requirement.
    During training RAM holds:
    - The dataset in memory for fast access
    - DataLoader worker buffers
    - OS and system processes
    - CPU optimizer states if using ZeRO offloading
    """
    vm = psutil.virtual_memory()

    total_gb     = vm.total     / 1024**3
    available_gb = vm.available / 1024**3
    used_gb      = vm.used      / 1024**3
    used_pct     = vm.percent

    print(f"  [INFO] Total RAM:     {total_gb:.1f} GB")
    print(f"  [INFO] Used RAM:      {used_gb:.1f} GB ({used_pct:.1f}%)")
    print(f"  [INFO] Available RAM: {available_gb:.1f} GB")

    total_pass = total_gb >= MIN_RAM_GB
    free_pass  = available_gb >= MIN_FREE_RAM_GB

    if total_pass:
        print(f"  [PASS] Total RAM: {total_gb:.1f}GB "
              f"(minimum {MIN_RAM_GB}GB)")
    else:
        print(f"  [FAIL] Total RAM: {total_gb:.1f}GB "
              f"(minimum {MIN_RAM_GB}GB)")

    if free_pass:
        print(f"  [PASS] Free RAM: {available_gb:.1f}GB "
              f"(minimum {MIN_FREE_RAM_GB}GB)")
    else:
        print(f"  [FAIL] Free RAM: {available_gb:.1f}GB "
              f"(minimum {MIN_FREE_RAM_GB}GB)")

    return total_pass and free_pass

def check_swap():
    """
    Check swap usage.
    Heavy swap usage during training severely degrades performance
    because swap is disk-backed and much slower than RAM.
    Training should never be hitting swap.
    """
    swap = psutil.swap_memory()
    swap_total_gb = swap.total / 1024**3
    swap_used_gb  = swap.used  / 1024**3
    swap_pct      = swap.percent

    print(f"  [INFO] Swap total: {swap_total_gb:.1f} GB")
    print(f"  [INFO] Swap used:  {swap_used_gb:.1f} GB ({swap_pct:.1f}%)")

    if swap_pct < 10:
        print(f"  [PASS] Swap usage acceptable: {swap_pct:.1f}%")
        return True
    else:
        print(f"  [WARN] Swap usage high: {swap_pct:.1f}% — "
              f"may indicate memory pressure")
        return True  # warn but don't fail

def check_memory_bandwidth():
    """
    Measure RAM read bandwidth.
    Data loading workers read from RAM into GPU memory —
    RAM bandwidth is the ceiling on how fast you can feed GPUs.
    """
    try:
        import numpy as np

        # Allocate a 2GB array
        size_gb     = 2
        num_elements = (size_gb * 1024**3) // 8  # float64 = 8 bytes

        print(f"  [INFO] Allocating {size_gb}GB array for bandwidth test...")

        arr = np.ones(num_elements, dtype=np.float64)

        # Measure read bandwidth
        iterations = 3
        times = []

        for i in range(iterations):
            start  = time.perf_counter()
            total  = arr.sum()           # forces full array read
            end    = time.perf_counter()
            times.append(end - start)

        avg_time      = sum(times) / len(times)
        bandwidth_gbs = (size_gb * 8) / avg_time  # x8 because float64

        del arr

        if bandwidth_gbs >= MIN_RAM_BANDWIDTH:
            print(f"  [PASS] RAM bandwidth: {bandwidth_gbs:.1f} GB/s "
                  f"(minimum {MIN_RAM_BANDWIDTH} GB/s)")
            return True
        else:
            print(f"  [FAIL] RAM bandwidth: {bandwidth_gbs:.1f} GB/s "
                  f"(minimum {MIN_RAM_BANDWIDTH} GB/s)")
            return False

    except ImportError:
        print(f"  [INFO] numpy not available — skipping bandwidth test")
        return True
    except Exception as e:
        print(f"  [FAIL] Memory bandwidth test error: {e}")
        return False

def check_dataloader_simulation():
    """
    Simulate PyTorch DataLoader worker behavior.
    Spawns multiple worker processes to verify the system
    can handle concurrent data preprocessing the same way
    PyTorch DataLoader does during training.
    """
    try:
        import numpy as np

        def worker_task(worker_id, result_dict):
            """Simulates one DataLoader worker preprocessing a batch."""
            # Simulate tokenizing 32 text samples
            batch_size    = 32
            sequence_len  = 512
            start         = time.perf_counter()

            # Create random token IDs like a tokenizer would
            batch = np.random.randint(0, 32000, (batch_size, sequence_len))
            # Simulate attention mask creation
            mask  = np.ones_like(batch)
            # Simulate basic preprocessing
            _     = batch * mask

            end = time.perf_counter()
            result_dict[worker_id] = end - start

        # Test with 8 workers — typical DataLoader num_workers setting
        num_workers  = 8
        manager      = multiprocessing.Manager()
        result_dict  = manager.dict()
        processes    = []

        start = time.perf_counter()

        for i in range(num_workers):
            p = multiprocessing.Process(
                target=worker_task,
                args=(i, result_dict)
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        end = time.perf_counter()
        total_time = end - start

        avg_worker_time = sum(result_dict.values()) / len(result_dict)

        print(f"  [INFO] {num_workers} DataLoader workers completed in {total_time:.3f}s")
        print(f"  [INFO] Average worker preprocessing time: {avg_worker_time*1000:.1f}ms")

        if total_time < 5.0:
            print(f"  [PASS] DataLoader simulation: {num_workers} workers "
                  f"completed in {total_time:.2f}s")
            return True
        else:
            print(f"  [FAIL] DataLoader simulation too slow: {total_time:.2f}s")
            return False

    except Exception as e:
        print(f"  [FAIL] DataLoader simulation error: {e}")
        return False

def check_cpu_info():
    """Report CPU model and architecture for documentation."""
    try:
        # Read from /proc/cpuinfo
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if "model name" in line:
                    cpu_model = line.split(":")[1].strip()
                    print(f"  [INFO] CPU model: {cpu_model}")
                    break
    except Exception:
        pass

    # Check architecture
    import platform
    print(f"  [INFO] Architecture: {platform.machine()}")
    print(f"  [INFO] OS: {platform.platform()}")

    return True

def main():
    print("\n" + "="*50)
    print("TEST 06: CPU & RAM Check")
    print("="*50)

    results = []

    print("\n--- CPU Checks ---")
    results.append(check_cpu_info())
    results.append(check_cpu_count())
    results.append(check_cpu_frequency())
    results.append(check_cpu_utilization())

    print("\n--- RAM Checks ---")
    results.append(check_total_ram())
    results.append(check_swap())
    results.append(check_memory_bandwidth())

    print("\n--- DataLoader Simulation ---")
    results.append(check_dataloader_simulation())

    all_passed = all(results)

    print("\n" + "-"*50)
    if all_passed:
        print("TEST 06 RESULT: PASS")
    else:
        print("TEST 06 RESULT: FAIL")
    print("="*50 + "\n")

    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
