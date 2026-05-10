#!/usr/bin/env python3
"""
Test 09: ECC Memory & Xid Error Monitoring
Checks for GPU memory errors and hardware faults.

ECC (Error Correcting Code) memory detects and corrects
single-bit memory errors silently. But uncorrected multi-bit
errors corrupt data — in training this means corrupted
gradients which silently produces a wrong model.

Xid errors are NVIDIA's internal error codes for GPU
hardware faults — driver crashes, memory faults, etc.
"""

import sys
import subprocess
import os

# Thresholds
MAX_CORRECTED_ECC_ERRORS   = 100  # corrected errors are fixed but indicate aging hardware
MAX_UNCORRECTED_ECC_ERRORS = 0    # any uncorrected error = immediate fail
MAX_XID_ERRORS             = 0    # any Xid error = hardware fault

def run_command(cmd):
    """Run shell command and return output."""
    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True
    )
    return result.stdout.strip(), result.returncode

def check_ecc_mode():
    """
    Verify ECC mode is enabled on all GPUs.
    ECC must be ON to detect and correct memory errors.
    H200s should have ECC enabled by default.
    """
    print("\n--- ECC Mode Check ---")

    output, rc = run_command(
        "nvidia-smi --query-gpu=index,ecc.mode.current "
        "--format=csv,noheader"
    )

    if rc != 0 or not output:
        print("  [WARN] Could not query ECC mode")
        return True  # not a hard fail

    all_pass = True
    for line in output.strip().split("\n"):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2:
            gpu_id   = parts[0]
            ecc_mode = parts[1]

            if ecc_mode.lower() == "enabled":
                print(f"  [PASS] GPU {gpu_id}: ECC mode enabled")
            else:
                print(f"  [WARN] GPU {gpu_id}: ECC mode {ecc_mode} "
                      f"— memory errors will not be corrected")
                all_pass = False

    return all_pass

def check_volatile_ecc_errors():
    """
    Check ECC errors since last driver reload (volatile).
    These are errors that happened in the current session.

    Corrected errors: single-bit flips fixed by ECC hardware
                      a few is normal, many indicates degrading memory
    Uncorrected errors: multi-bit flips ECC could not fix
                        ANY uncorrected error = data corruption = fail
    """
    print("\n--- Volatile ECC Errors (current session) ---")

    # Query corrected errors
    corrected_out, _ = run_command(
        "nvidia-smi --query-gpu=index,"
        "ecc.errors.corrected.volatile.total "
        "--format=csv,noheader,nounits"
    )

    # Query uncorrected errors
    uncorrected_out, _ = run_command(
        "nvidia-smi --query-gpu=index,"
        "ecc.errors.uncorrected.volatile.total "
        "--format=csv,noheader,nounits"
    )

    all_pass     = True
    corrected    = {}
    uncorrected  = {}

    # Parse corrected errors
    if corrected_out:
        for line in corrected_out.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                try:
                    gpu_id = int(parts[0])
                    errors = int(parts[1])
                    corrected[gpu_id] = errors
                except ValueError:
                    pass

    # Parse uncorrected errors
    if uncorrected_out:
        for line in uncorrected_out.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                try:
                    gpu_id = int(parts[0])
                    errors = int(parts[1])
                    uncorrected[gpu_id] = errors
                except ValueError:
                    pass

    # Report and evaluate
    all_gpu_ids = set(list(corrected.keys()) + list(uncorrected.keys()))

    for gpu_id in sorted(all_gpu_ids):
        corr  = corrected.get(gpu_id, 0)
        uncorr = uncorrected.get(gpu_id, 0)

        print(f"\n  GPU {gpu_id}:")
        print(f"    Corrected errors:   {corr}")
        print(f"    Uncorrected errors: {uncorr}")

        # Corrected errors check
        if corr == 0:
            print(f"    [PASS] No corrected ECC errors")
        elif corr <= MAX_CORRECTED_ECC_ERRORS:
            print(f"    [WARN] {corr} corrected ECC errors — "
                  f"monitor for increase over time")
        else:
            print(f"    [FAIL] {corr} corrected ECC errors exceeds "
                  f"threshold {MAX_CORRECTED_ECC_ERRORS} — "
                  f"GPU memory may be degrading")
            all_pass = False

        # Uncorrected errors check — zero tolerance
        if uncorr == 0:
            print(f"    [PASS] No uncorrected ECC errors")
        else:
            print(f"    [FAIL] {uncorr} uncorrected ECC errors — "
                  f"DATA CORRUPTION RISK — do not train on this GPU")
            all_pass = False

    return all_pass

def check_aggregate_ecc_errors():
    """
    Check ECC errors since last system reboot (aggregate).
    Aggregate errors show the GPU's error history over time.
    High aggregate errors indicate a GPU that has been
    degrading over its lifetime.
    """
    print("\n--- Aggregate ECC Errors (since reboot) ---")

    corrected_out, _ = run_command(
        "nvidia-smi --query-gpu=index,"
        "ecc.errors.corrected.aggregate.total "
        "--format=csv,noheader,nounits"
    )

    uncorrected_out, _ = run_command(
        "nvidia-smi --query-gpu=index,"
        "ecc.errors.uncorrected.aggregate.total "
        "--format=csv,noheader,nounits"
    )

    all_pass = True

    if corrected_out:
        for line in corrected_out.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                try:
                    gpu_id = int(parts[0])
                    errors = int(parts[1])
                    print(f"  [INFO] GPU {gpu_id} aggregate corrected errors: {errors}")
                    if errors > MAX_CORRECTED_ECC_ERRORS * 10:
                        print(f"  [WARN] GPU {gpu_id} high lifetime corrected "
                              f"errors: {errors}")
                except ValueError:
                    pass

    if uncorrected_out:
        for line in uncorrected_out.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                try:
                    gpu_id = int(parts[0])
                    errors = int(parts[1])
                    if errors == 0:
                        print(f"  [PASS] GPU {gpu_id}: "
                              f"no aggregate uncorrected errors")
                    else:
                        print(f"  [FAIL] GPU {gpu_id}: "
                              f"{errors} aggregate uncorrected errors — "
                              f"GPU has history of data corruption")
                        all_pass = False
                except ValueError:
                    pass

    return all_pass

def check_retired_pages():
    """
    Check GPU memory page retirement.
    When GPU memory cells fail permanently NVIDIA retires
    those memory pages so they are no longer used.
    Too many retired pages means the GPU is significantly
    degraded and should be replaced.
    """
    print("\n--- Retired Memory Pages ---")

    # Single bit retirement
    sbe_out, _ = run_command(
        "nvidia-smi --query-gpu=index,"
        "retired_pages.single_bit_ecc.count "
        "--format=csv,noheader,nounits"
    )

    # Double bit retirement
    dbe_out, _ = run_command(
        "nvidia-smi --query-gpu=index,"
        "retired_pages.double_bit.count "
        "--format=csv,noheader,nounits"
    )

    all_pass = True

    if sbe_out:
        for line in sbe_out.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                try:
                    gpu_id = int(parts[0])
                    pages  = int(parts[1])

                    if pages == 0:
                        print(f"  [PASS] GPU {gpu_id}: "
                              f"no single-bit retired pages")
                    elif pages < 60:
                        print(f"  [WARN] GPU {gpu_id}: "
                              f"{pages} single-bit retired pages")
                    else:
                        print(f"  [FAIL] GPU {gpu_id}: "
                              f"{pages} single-bit retired pages — "
                              f"GPU significantly degraded")
                        all_pass = False
                except ValueError:
                    pass

    if dbe_out:
        for line in dbe_out.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                try:
                    gpu_id = int(parts[0])
                    pages  = int(parts[1])

                    if pages == 0:
                        print(f"  [PASS] GPU {gpu_id}: "
                              f"no double-bit retired pages")
                    else:
                        print(f"  [FAIL] GPU {gpu_id}: "
                              f"{pages} double-bit retired pages — "
                              f"serious hardware degradation")
                        all_pass = False
                except ValueError:
                    pass

    return all_pass

def check_xid_errors():
    """
    Check for Xid errors in the system kernel log.
    Xid errors are NVIDIA's internal error codes written
    to dmesg when GPU hardware faults occur.

    Common critical Xid codes:
    Xid 13  = graphics engine exception
    Xid 31  = GPU memory page fault
    Xid 48  = DBE (double bit error) ECC
    Xid 63  = row remapping failure
    Xid 74  = NVLINK error
    Xid 79  = GPU has fallen off the bus
    Xid 94  = contained channel error
    Xid 95  = uncontained error — most serious
    """
    print("\n--- Xid Error Check (kernel log) ---")

    # Critical Xid codes that indicate serious hardware problems
    critical_xids = {
        13: "Graphics engine exception",
        31: "GPU memory page fault",
        48: "Double-bit ECC error",
        63: "Row remapping failure",
        74: "NVLink error",
        79: "GPU fallen off bus",
        94: "Contained channel error",
        95: "Uncontained error — most critical"
    }

    # Read kernel log for NVIDIA Xid errors
    dmesg_out, rc = run_command(
        "dmesg 2>/dev/null | grep -i 'Xid' | tail -20"
    )

    if rc != 0 or not dmesg_out:
        print("  [PASS] No Xid errors found in kernel log")
        return True

    # Parse Xid codes from dmesg output
    found_xids   = []
    critical_found = []

    for line in dmesg_out.strip().split("\n"):
        if "Xid" in line:
            found_xids.append(line)
            # Check if any critical Xid codes are present
            for xid_code, description in critical_xids.items():
                if f"Xid {xid_code}" in line or f"Xid={xid_code}" in line:
                    critical_found.append((xid_code, description, line))

    if not found_xids:
        print("  [PASS] No Xid errors found")
        return True

    print(f"  [WARN] Found {len(found_xids)} Xid entries in kernel log:")
    for line in found_xids[-5:]:  # show last 5
        print(f"    {line}")

    if critical_found:
        print(f"\n  [FAIL] Critical Xid errors detected:")
        for xid_code, description, line in critical_found:
            print(f"    Xid {xid_code}: {description}")
        return False
    else:
        print(f"  [WARN] Non-critical Xid entries found — monitor during training")
        return True

def check_gpu_remapped_rows():
    """
    Check for remapped memory rows.
    When memory cells fail NVIDIA remaps them to spare rows.
    Many remapped rows indicates serious memory degradation.
    """
    print("\n--- GPU Memory Row Remapping ---")

    output, rc = run_command(
        "nvidia-smi --query-remapped-rows="
        "gpu_uuid,remapped_rows.correctable,"
        "remapped_rows.uncorrectable,"
        "remapped_rows.pending,"
        "remapped_rows.failure "
        "--format=csv,noheader"
    )

    if rc != 0 or not output:
        print("  [INFO] Row remapping info not available on this system")
        return True

    all_pass = True
    for i, line in enumerate(output.strip().split("\n")):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 5:
            correctable   = parts[1]
            uncorrectable = parts[2]
            pending       = parts[3]
            failure       = parts[4]

            print(f"  [INFO] GPU {i}: correctable={correctable}, "
                  f"uncorrectable={uncorrectable}, "
                  f"pending={pending}, failure={failure}")

            if failure.lower() == "true":
                print(f"  [FAIL] GPU {i}: row remapping failure detected")
                all_pass = False
            elif pending.lower() == "true":
                print(f"  [WARN] GPU {i}: row remapping pending — "
                      f"reboot required to apply")
            else:
                print(f"  [PASS] GPU {i}: no row remapping failures")

    return all_pass

def main():
    print("\n" + "="*50)
    print("TEST 09: ECC Memory & Xid Error Monitoring")
    print("="*50)

    results = []

    results.append(check_ecc_mode())
    results.append(check_volatile_ecc_errors())
    results.append(check_aggregate_ecc_errors())
    results.append(check_retired_pages())
    results.append(check_xid_errors())
    results.append(check_gpu_remapped_rows())

    all_passed = all(results)

    print("\n" + "-"*50)
    if all_passed:
        print("TEST 09 RESULT: PASS")
    else:
        print("TEST 09 RESULT: FAIL")
    print("="*50 + "\n")

    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
