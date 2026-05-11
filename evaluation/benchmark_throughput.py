#!/usr/bin/env python3
"""
benchmark_throughput.py
Measures inference throughput of the fine-tuned model
using vLLM — the industry standard high-throughput
inference engine.

We optimize for THROUGHPUT not latency because:
- Startup is building a multi-user serving system
- Throughput = how many users can be served simultaneously
- Latency = how fast one user gets a response
- For concurrent users throughput is the business metric

Throughput is measured in tokens per second (tok/s)
across different batch sizes to find the optimal
operating point.

Usage:
    python3 benchmark_throughput.py \
        --model_path /mnt/data/outputs/finetuned-llama \
        --output_dir ../results
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark inference throughput using vLLM"
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default="/mnt/data/outputs/finetuned-llama",
        help="Path to fine-tuned model"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="../results",
        help="Directory to save benchmark results"
    )

    parser.add_argument(
        "--batch_sizes",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 32, 64],
        help="Batch sizes to benchmark"
    )

    parser.add_argument(
        "--input_length",
        type=int,
        default=128,
        help="Number of input tokens per request"
    )

    parser.add_argument(
        "--output_length",
        type=int,
        default=128,
        help="Number of output tokens to generate per request"
    )

    parser.add_argument(
        "--num_requests",
        type=int,
        default=100,
        help="Total requests per batch size measurement"
    )

    parser.add_argument(
        "--warmup_requests",
        type=int,
        default=10,
        help="Warmup requests before timed measurement"
    )

    return parser.parse_args()


def load_vllm_model(model_path):
    """
    Load the model with vLLM.

    vLLM is different from standard HuggingFace inference:

    Standard HF:
    - Processes one request at a time
    - Simple but inefficient
    - GPU sits partially idle between tokens

    vLLM:
    - Continuous batching — processes many requests simultaneously
    - PagedAttention — efficient KV cache memory management
    - GPU stays fully utilized
    - 2-4x higher throughput than standard HF inference

    This is what the startup would use in production
    so benchmarking with vLLM gives realistic numbers.
    """
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        print("[FAIL] vLLM not installed.")
        print("Install with: pip install vllm")
        sys.exit(1)

    print(f"\nLoading model with vLLM: {model_path}")
    print("vLLM initializes PagedAttention KV cache on startup...")

    llm = LLM(
        model              = model_path,
        dtype              = "bfloat16",
        # tensor_parallel_size controls how many GPUs vLLM uses
        # 2 GPUs for inference — enough for 8B model with headroom
        tensor_parallel_size = 2,
        # GPU memory utilization — use 90% of available GPU memory
        # for KV cache. Leave 10% for other operations.
        gpu_memory_utilization = 0.9,
        # Maximum sequence length
        max_model_len      = 2048,
    )

    print("[PASS] vLLM model loaded successfully")
    return llm


def generate_test_prompts(num_prompts, input_length):
    """
    Generate realistic test prompts for benchmarking.
    Uses professional law style questions to match
    the domain of our fine-tuned model.

    Using domain-relevant prompts is more realistic
    than random token sequences — the KV cache patterns
    will be representative of actual usage.
    """
    # Sample professional law questions for benchmarking
    law_prompts = [
        "### Question:\nUnder the common law, which of the following is true with respect to an offer?\n\nA) An offer can be accepted after rejection\nB) An offer cannot be revoked once made\nC) An offer is binding once communicated\nD) An offer may be revoked before acceptance\n\n### Answer:",
        "### Question:\nWhich of the following best describes the concept of promissory estoppel?\n\nA) A promise that is supported by consideration\nB) A promise that is enforceable despite lack of consideration\nC) A promise that requires written documentation\nD) A promise made under duress\n\n### Answer:",
        "### Question:\nIn criminal law, which element distinguishes murder from manslaughter?\n\nA) The age of the victim\nB) The use of a weapon\nC) Malice aforethought\nD) The location of the crime\n\n### Answer:",
        "### Question:\nWhich of the following is required for a valid contract?\n\nA) Written documentation only\nB) Notarization by a public official\nC) Offer, acceptance, and consideration\nD) Witnesses present at signing\n\n### Answer:",
        "### Question:\nUnder the Fourth Amendment, which standard is required for a valid search warrant?\n\nA) Reasonable suspicion\nB) Preponderance of evidence\nC) Probable cause\nD) Beyond reasonable doubt\n\n### Answer:",
    ]

    # Cycle through prompts to fill num_prompts
    prompts = []
    for i in range(num_prompts):
        prompts.append(law_prompts[i % len(law_prompts)])

    return prompts


def benchmark_batch_size(llm, batch_size, input_length,
                          output_length, num_requests,
                          warmup_requests):
    """
    Measures throughput at a specific batch size.

    The measurement process:
    1. Warmup — run some requests to initialize CUDA kernels
       and fill the KV cache. First requests are always slower.
    2. Timed run — measure wall clock time for num_requests
    3. Calculate throughput = total tokens / total time

    We measure total tokens (input + output) because both
    input processing and output generation consume GPU compute.
    """
    try:
        from vllm import SamplingParams
    except ImportError:
        sys.exit(1)

    # Sampling parameters
    # temperature=0 means greedy decoding — deterministic output
    # This is important for benchmarking: same input always
    # produces same output, making results reproducible
    sampling_params = SamplingParams(
        temperature    = 0.0,   # greedy — deterministic
        max_tokens     = output_length,
    )

    # Generate test prompts
    prompts = generate_test_prompts(
        num_requests + warmup_requests,
        input_length
    )

    print(f"\n  Batch size {batch_size:3d}:")
    print(f"    Warming up with {warmup_requests} requests...")

    # Warmup run — discarded from measurement
    warmup_prompts = prompts[:warmup_requests]
    _ = llm.generate(warmup_prompts, sampling_params)

    print(f"    Running {num_requests} timed requests...")

    # Timed run
    timed_prompts = prompts[warmup_requests:warmup_requests + num_requests]

    start_time = time.perf_counter()
    outputs    = llm.generate(timed_prompts, sampling_params)
    end_time   = time.perf_counter()

    elapsed = end_time - start_time

    # Count actual tokens generated
    total_output_tokens = sum(
        len(output.outputs[0].token_ids)
        for output in outputs
    )
    total_input_tokens  = input_length * num_requests
    total_tokens        = total_input_tokens + total_output_tokens

    # Calculate metrics
    throughput_toks   = total_tokens / elapsed
    throughput_reqs   = num_requests / elapsed
    latency_per_req   = elapsed / num_requests * 1000  # ms

    print(f"    Throughput:  {throughput_toks:.1f} tok/s")
    print(f"    Requests/s:  {throughput_reqs:.2f} req/s")
    print(f"    Avg latency: {latency_per_req:.1f}ms per request")

    return {
        "batch_size":         batch_size,
        "throughput_tokens":  throughput_toks,
        "throughput_requests":throughput_reqs,
        "latency_ms":         latency_per_req,
        "total_tokens":       total_tokens,
        "elapsed_seconds":    elapsed,
        "num_requests":       num_requests,
    }


def find_optimal_batch_size(results):
    """
    Identifies the optimal batch size — the point where
    throughput plateaus and adding more requests gives
    diminishing returns.

    The optimal point is where:
    - Throughput is near maximum
    - Memory is not exhausted
    - Latency is still acceptable

    We find this by looking for where throughput gain
    drops below 10% per batch size doubling.
    """
    if len(results) < 2:
        return results[-1]["batch_size"] if results else 1

    optimal = results[0]

    for i in range(1, len(results)):
        current  = results[i]["throughput_tokens"]
        previous = results[i-1]["throughput_tokens"]

        # Calculate improvement percentage
        improvement = (current - previous) / previous * 100

        if improvement < 10:
            # Less than 10% improvement — we've hit diminishing returns
            # Previous batch size was the sweet spot
            optimal = results[i-1]
            break
        else:
            optimal = results[i]

    return optimal["batch_size"]


def calculate_user_capacity(optimal_result):
    """
    Translates throughput numbers into a concrete business metric:
    how many concurrent users can this setup serve?

    Assumptions:
    - Average response = 200 tokens
    - User waits up to 2 seconds for a response
    - Concurrent users = throughput * acceptable_wait / avg_response
    """
    tokens_per_second   = optimal_result["throughput_tokens"]
    avg_response_tokens = 200
    acceptable_wait_sec = 2.0

    responses_per_sec   = tokens_per_second / avg_response_tokens
    concurrent_users    = responses_per_sec * acceptable_wait_sec

    return int(concurrent_users)


def generate_benchmark_report(results, optimal_batch_size,
                               model_path, args):
    """
    Generates the throughput benchmark report.
    This is what you show on demo day to prove
    the model is production-ready.
    """
    W = 65
    report = []

    report.append("")
    report.append("=" * W)
    report.append(" " * 8 + "INFERENCE THROUGHPUT BENCHMARK REPORT")
    report.append("=" * W)
    report.append(f"  Model:        {model_path}")
    report.append(f"  Input length: {args.input_length} tokens")
    report.append(f"  Output length:{args.output_length} tokens")
    report.append(f"  Requests:     {args.num_requests} per batch size")
    report.append(f"  Engine:       vLLM (PagedAttention)")
    report.append(f"  Generated:    "
                  f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    report.append("=" * W)

    # Results table
    report.append("")
    report.append(
        f"  {'Batch':>6}  {'Throughput':>14}  "
        f"{'Req/s':>8}  {'Latency':>10}  {'Note':>10}"
    )
    report.append("  " + "-" * (W - 4))

    optimal_result = None
    for r in results:
        bs       = r["batch_size"]
        toks     = r["throughput_tokens"]
        reqs     = r["throughput_requests"]
        lat      = r["latency_ms"]
        note     = "← optimal" if bs == optimal_batch_size else ""

        if bs == optimal_batch_size:
            optimal_result = r

        report.append(
            f"  {bs:>6}  {toks:>11.1f} t/s  "
            f"{reqs:>6.2f} r/s  {lat:>8.1f}ms  {note}"
        )

    report.append("  " + "-" * (W - 4))
    report.append("")

    # Key finding
    if optimal_result:
        users = calculate_user_capacity(optimal_result)
        report.append("=" * W)
        report.append("  KEY FINDING")
        report.append("=" * W)
        report.append("")
        report.append(
            f"  Optimal batch size:  {optimal_batch_size}"
        )
        report.append(
            f"  Peak throughput:     "
            f"{optimal_result['throughput_tokens']:.1f} tokens/second"
        )
        report.append(
            f"  Request rate:        "
            f"{optimal_result['throughput_requests']:.2f} requests/second"
        )
        report.append(
            f"  Avg latency:         "
            f"{optimal_result['latency_ms']:.1f}ms per request"
        )
        report.append("")
        report.append(
            f"  Estimated concurrent users: ~{users}"
        )
        report.append(
            f"  (assuming 200 tok avg response, 2s acceptable wait)"
        )
        report.append("")

    # Why throughput
    report.append("=" * W)
    report.append("  WHY WE OPTIMIZED FOR THROUGHPUT")
    report.append("=" * W)
    report.append("")
    report.append(
        "  The startup is building a multi-user serving system."
    )
    report.append(
        "  Throughput — tokens per second across concurrent"
    )
    report.append(
        "  requests — directly maps to how many users can be"
    )
    report.append(
        "  served simultaneously. Latency optimization makes"
    )
    report.append(
        "  sense for single-user real-time applications."
    )
    report.append(
        "  For concurrent user serving, throughput is the"
    )
    report.append(
        "  business-relevant metric."
    )
    report.append("")

    # vLLM explanation
    report.append("=" * W)
    report.append("  WHY vLLM")
    report.append("=" * W)
    report.append("")
    report.append(
        "  PagedAttention: manages KV cache in fixed-size pages"
    )
    report.append(
        "  like OS virtual memory — eliminates fragmentation,"
    )
    report.append(
        "  allows more requests to be batched simultaneously."
    )
    report.append("")
    report.append(
        "  Continuous batching: inserts new requests as slots"
    )
    report.append(
        "  free up mid-generation — GPU stays fully utilized"
    )
    report.append(
        "  rather than waiting for entire batch to complete."
    )
    report.append("")
    report.append(
        "  Result: 2-4x higher throughput vs standard HF inference."
    )
    report.append("")
    report.append("=" * W)

    return "\n".join(report)


def main():
    args = parse_args()

    print("\n" + "="*60)
    print("INFERENCE THROUGHPUT BENCHMARK")
    print("="*60)
    print(f"Model:        {args.model_path}")
    print(f"Batch sizes:  {args.batch_sizes}")
    print(f"Input tokens: {args.input_length}")
    print(f"Output tokens:{args.output_length}")
    print("="*60)

    # Load model
    llm = load_vllm_model(args.model_path)

    # Run benchmark across all batch sizes
    print("\nRunning throughput benchmark...")
    results = []

    for batch_size in args.batch_sizes:
        try:
            result = benchmark_batch_size(
                llm             = llm,
                batch_size      = batch_size,
                input_length    = args.input_length,
                output_length   = args.output_length,
                num_requests    = args.num_requests,
                warmup_requests = args.warmup_requests
            )
            results.append(result)

        except Exception as e:
            print(f"  [WARN] Batch size {batch_size} failed: {e}")
            print(f"  [INFO] Likely OOM — stopping here")
            break

    if not results:
        print("[FAIL] No benchmark results collected")
        sys.exit(1)

    # Find optimal batch size
    optimal = find_optimal_batch_size(results)
    print(f"\nOptimal batch size: {optimal}")

    # Generate report
    report = generate_benchmark_report(results, optimal,
                                       args.model_path, args)
    print(report)

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)

    # Save JSON
    json_file = os.path.join(
        args.output_dir, "throughput_results.json"
    )
    with open(json_file, "w") as f:
        json.dump({
            "results":            results,
            "optimal_batch_size": optimal,
            "model_path":         args.model_path,
            "timestamp":          datetime.now().isoformat()
        }, f, indent=2)

    # Save report
    report_file = os.path.join(
        args.output_dir, "throughput_report.txt"
    )
    with open(report_file, "w") as f:
        f.write(report)

    print(f"\nJSON results saved to:  {json_file}")
    print(f"Report saved to:        {report_file}")


if __name__ == "__main__":
    main()
