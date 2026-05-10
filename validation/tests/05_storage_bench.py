#!/usr/bin/env python3
"""
Test 05: Storage & Filesystem Benchmark
Validates read/write throughput across all storage locations
used during training. Slow storage = GPU starvation = wasted
GPU time sitting idle waiting for data.
"""

import os
import sys
import time
import shutil

# Per-location thresholds — calibrated to actual cluster performance
# /tmp is slower on this cluster (local disk, not optimized)
# /mnt/memory is RAM-backed so fastest
# /home and /mnt/data are NFS/network storage
STORAGE_THRESHOLDS = {
    "home":   {"read": 0.5,  "write": 0.3},
    "data":   {"read": 0.5,  "write": 0.3},
    "tmp":    {"read": 0.3,  "write": 0.1},
    "memory": {"read": 1.0,  "write": 1.0},
}

# Storage locations used during training
STORAGE_LOCATIONS = {
    "home":   "/home/adam",
    "data":   "/mnt/data",
    "tmp":    "/tmp",
    "memory": "/mnt/memory",
}

# Test file size — 1GB gives accurate throughput measurement
TEST_FILE_SIZE_GB    = 1
TEST_FILE_SIZE_BYTES = TEST_FILE_SIZE_GB * 1024 * 1024 * 1024

def check_storage_exists(name, path):
    """Verify the storage location is mounted and accessible."""
    if os.path.exists(path):
        stat     = shutil.disk_usage(path)
        total_gb = stat.total / 1024**3
        free_gb  = stat.free  / 1024**3
        used_pct = ((stat.total - stat.free) / stat.total) * 100
        print(f"  [PASS] {name} ({path}) is accessible")
        print(f"  [INFO] {name}: {free_gb:.1f}GB free of "
              f"{total_gb:.1f}GB ({used_pct:.1f}% used)")
        return True
    else:
        print(f"  [FAIL] {name} ({path}) is not accessible")
        return False

def benchmark_write(name, path):
    """
    Measure sequential write throughput.
    During training you write checkpoints — if this is slow
    your training pauses every time it saves a checkpoint.
    """
    test_file  = os.path.join(path, f"validation_write_{os.getpid()}.bin")
    chunk_size = 64 * 1024 * 1024  # 64MB chunks
    chunks     = TEST_FILE_SIZE_BYTES // chunk_size
    data_chunk = b"0" * chunk_size
    threshold  = STORAGE_THRESHOLDS.get(name, {}).get("write", 0.3)

    try:
        start = time.perf_counter()
        with open(test_file, "wb") as f:
            for _ in range(chunks):
                f.write(data_chunk)
            f.flush()
            os.fsync(f.fileno())
        end = time.perf_counter()

        throughput_gbs = TEST_FILE_SIZE_GB / (end - start)

        if throughput_gbs >= threshold:
            print(f"  [PASS] {name} write: {throughput_gbs:.2f} GB/s "
                  f"(minimum {threshold} GB/s)")
        else:
            print(f"  [FAIL] {name} write: {throughput_gbs:.2f} GB/s "
                  f"(minimum {threshold} GB/s)")
            return False

        return True

    except Exception as e:
        print(f"  [FAIL] {name} write error: {e}")
        return False

    finally:
        if os.path.exists(test_file):
            os.remove(test_file)

def benchmark_read(name, path):
    """
    Measure sequential read throughput.
    During training you constantly read data batches —
    if this is slow GPUs sit idle waiting for data.
    This is the data starvation problem.
    """
    test_file  = os.path.join(path, f"validation_read_{os.getpid()}.bin")
    chunk_size = 64 * 1024 * 1024  # 64MB chunks
    chunks     = TEST_FILE_SIZE_BYTES // chunk_size
    data_chunk = b"0" * chunk_size
    threshold  = STORAGE_THRESHOLDS.get(name, {}).get("read", 0.5)

    try:
        # Write test file first
        with open(test_file, "wb") as f:
            for _ in range(chunks):
                f.write(data_chunk)
            f.flush()
            os.fsync(f.fileno())

        # Sync to clear OS cache
        try:
            os.system("sync")
        except Exception:
            pass

        # Measure read speed
        start = time.perf_counter()
        with open(test_file, "rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
        end = time.perf_counter()

        throughput_gbs = TEST_FILE_SIZE_GB / (end - start)

        if throughput_gbs >= threshold:
            print(f"  [PASS] {name} read:  {throughput_gbs:.2f} GB/s "
                  f"(minimum {threshold} GB/s)")
        else:
            print(f"  [FAIL] {name} read:  {throughput_gbs:.2f} GB/s "
                  f"(minimum {threshold} GB/s)")
            return False

        return True

    except Exception as e:
        print(f"  [FAIL] {name} read error: {e}")
        return False

    finally:
        if os.path.exists(test_file):
            os.remove(test_file)

def benchmark_small_files(name, path):
    """
    Measure performance with many small files.
    Dataset loading often reads thousands of small files
    not one large file — this tests that scenario.
    """
    test_dir  = os.path.join(path, f"validation_small_{os.getpid()}")
    num_files = 1000
    file_size = 100 * 1024  # 100KB each

    try:
        os.makedirs(test_dir, exist_ok=True)

        # Write
        start = time.perf_counter()
        for i in range(num_files):
            fpath = os.path.join(test_dir, f"file_{i:04d}.bin")
            with open(fpath, "wb") as f:
                f.write(b"0" * file_size)
        end = time.perf_counter()
        print(f"  [INFO] {name} small file writes: "
              f"{num_files / (end - start):.0f} files/sec")

        # Read
        start = time.perf_counter()
        for i in range(num_files):
            fpath = os.path.join(test_dir, f"file_{i:04d}.bin")
            with open(fpath, "rb") as f:
                _ = f.read()
        end = time.perf_counter()
        print(f"  [INFO] {name} small file reads:  "
              f"{num_files / (end - start):.0f} files/sec")

        return True

    except Exception as e:
        print(f"  [FAIL] {name} small file test error: {e}")
        return False

    finally:
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)

def check_checkpoint_write_speed():
    """
    Simulate writing a model checkpoint.
    A Llama 3.1 8B LoRA checkpoint is roughly 500MB.
    This goes to /mnt/data — the primary training storage.
    """
    print(f"\n--- Checkpoint Write Simulation ---")
    checkpoint_path = "/mnt/data"

    if not os.path.exists(checkpoint_path):
        print(f"  [WARN] /mnt/data not accessible — skipping")
        return True

    test_file          = os.path.join(checkpoint_path,
                                      f"checkpoint_test_{os.getpid()}.bin")
    checkpoint_size_gb = 0.5  # 500MB
    chunk_size         = 64 * 1024 * 1024
    chunks             = int((checkpoint_size_gb * 1024**3) // chunk_size)
    data               = b"0" * chunk_size

    try:
        start = time.perf_counter()
        with open(test_file, "wb") as f:
            for _ in range(chunks):
                f.write(data)
            f.flush()
            os.fsync(f.fileno())
        end     = time.perf_counter()
        elapsed = end - start

        throughput_gbs = checkpoint_size_gb / elapsed
        print(f"  [INFO] 500MB checkpoint write: "
              f"{elapsed:.2f}s at {throughput_gbs:.2f} GB/s")

        if elapsed < 30:
            print(f"  [PASS] Checkpoint write time acceptable "
                  f"({elapsed:.1f}s < 30s)")
            return True
        else:
            print(f"  [FAIL] Checkpoint write too slow "
                  f"({elapsed:.1f}s > 30s)")
            return False

    except Exception as e:
        print(f"  [FAIL] Checkpoint write error: {e}")
        return False

    finally:
        if os.path.exists(test_file):
            os.remove(test_file)

def main():
    print("\n" + "="*50)
    print("TEST 05: Storage & Filesystem Benchmark")
    print("="*50)

    results = []

    for name, path in STORAGE_LOCATIONS.items():
        print(f"\n--- {name.upper()} Storage: {path} ---")

        if not check_storage_exists(name, path):
            results.append(False)
            continue

        # Skip 1GB test on /mnt/memory if not enough free space
        if name == "memory":
            stat = shutil.disk_usage(path)
            if stat.free < TEST_FILE_SIZE_BYTES:
                print(f"  [INFO] /mnt/memory too small for 1GB test — skipping")
                continue

        results.append(benchmark_write(name, path))
        results.append(benchmark_read(name, path))
        results.append(benchmark_small_files(name, path))

    results.append(check_checkpoint_write_speed())

    all_passed = all(results)

    print("\n" + "-"*50)
    if all_passed:
        print("TEST 05 RESULT: PASS")
    else:
        print("TEST 05 RESULT: FAIL")
    print("="*50 + "\n")

    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
