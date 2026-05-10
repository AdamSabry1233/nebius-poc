#!/usr/bin/env python3
"""
Test 02: CUDA Functionality Check
Verifies CUDA is installed, version is compatible with PyTorch,
and basic GPU computation works correctly on each GPU.
"""

import sys

# CUDA version threshold
MIN_CUDA_VERSION = "12.0"

def check_pytorch_cuda():
    """Verify PyTorch can see CUDA."""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  [PASS] PyTorch {torch.__version__} — CUDA available")
            return True
        else:
            print(f"  [FAIL] PyTorch {torch.__version__} — CUDA not available")
            return False
    except ImportError:
        print("  [FAIL] PyTorch not installed")
        return False

def check_cuda_version():
    """Verify CUDA version meets minimum requirement."""
    try:
        import torch
        cuda_version = torch.version.cuda

        if cuda_version is None:
            print("  [FAIL] CUDA version could not be determined")
            return False

        # Compare versions
        current = tuple(int(x) for x in cuda_version.split("."))
        minimum = tuple(int(x) for x in MIN_CUDA_VERSION.split("."))

        if current >= minimum:
            print(f"  [PASS] CUDA version: {cuda_version} (minimum {MIN_CUDA_VERSION})")
            return True
        else:
            print(f"  [FAIL] CUDA version: {cuda_version} below minimum {MIN_CUDA_VERSION}")
            return False

    except Exception as e:
        print(f"  [FAIL] Error checking CUDA version: {e}")
        return False

def check_gpu_count():
    """Report how many GPUs PyTorch can see."""
    try:
        import torch
        count = torch.cuda.device_count()

        if count > 0:
            print(f"  [PASS] PyTorch sees {count} GPU(s)")
            return True
        else:
            print(f"  [FAIL] PyTorch sees 0 GPUs")
            return False

    except Exception as e:
        print(f"  [FAIL] Error counting GPUs: {e}")
        return False

def check_tensor_computation():
    """
    Run a basic tensor operation on each GPU and verify
    the result is correct. This proves CUDA computation
    actually works — not just that CUDA is detected.
    """
    try:
        import torch
        all_pass = True
        gpu_count = torch.cuda.device_count()

        for i in range(gpu_count):
            device = torch.device(f"cuda:{i}")

            # Create two tensors on this GPU
            a = torch.tensor([1.0, 2.0, 3.0], device=device)
            b = torch.tensor([4.0, 5.0, 6.0], device=device)

            # Perform addition
            result = a + b

            # Expected result
            expected = torch.tensor([5.0, 7.0, 9.0], device=device)

            # Verify result is correct
            if torch.allclose(result, expected):
                print(f"  [PASS] GPU {i} tensor computation correct: {result.tolist()}")
            else:
                print(f"  [FAIL] GPU {i} tensor computation wrong: got {result.tolist()}")
                all_pass = False

        return all_pass

    except Exception as e:
        print(f"  [FAIL] Tensor computation error: {e}")
        return False

def check_memory_allocation():
    """
    Verify we can allocate and free GPU memory correctly.
    Silent memory allocation failures cause training crashes.
    """
    try:
        import torch
        all_pass = True
        gpu_count = torch.cuda.device_count()

        for i in range(gpu_count):
            device = torch.device(f"cuda:{i}")

            # Allocate a 1GB tensor
            try:
                tensor = torch.zeros(
                    1024, 1024, 256,
                    dtype=torch.float32,
                    device=device
                )
                allocated_gb = tensor.element_size() * tensor.nelement() / 1024**3
                del tensor
                torch.cuda.empty_cache()
                print(f"  [PASS] GPU {i} memory allocation: {allocated_gb:.2f}GB allocated and freed")

            except RuntimeError as e:
                print(f"  [FAIL] GPU {i} memory allocation failed: {e}")
                all_pass = False

        return all_pass

    except Exception as e:
        print(f"  [FAIL] Memory allocation check error: {e}")
        return False

def check_cuda_device_properties():
    """Report key device properties for documentation."""
    try:
        import torch
        gpu_count = torch.cuda.device_count()

        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            print(f"  [INFO] GPU {i}: {props.name}")
            print(f"  [INFO] GPU {i}: Compute capability {props.major}.{props.minor}")
            print(f"  [INFO] GPU {i}: Total memory {props.total_memory / 1024**3:.1f}GB")
            print(f"  [INFO] GPU {i}: Multiprocessors {props.multi_processor_count}")

        return True

    except Exception as e:
        print(f"  [FAIL] Could not retrieve device properties: {e}")
        return False

def main():
    print("\n" + "="*50)
    print("TEST 02: CUDA Functionality Check")
    print("="*50)

    results = []
    results.append(check_pytorch_cuda())
    results.append(check_cuda_version())
    results.append(check_gpu_count())
    results.append(check_tensor_computation())
    results.append(check_memory_allocation())
    results.append(check_cuda_device_properties())

    all_passed = all(results)

    print("-"*50)
    if all_passed:
        print("TEST 02 RESULT: PASS")
    else:
        print("TEST 02 RESULT: FAIL")
    print("="*50 + "\n")

    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
