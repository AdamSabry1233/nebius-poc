#!/usr/bin/env python3
"""
benchmark_throughput.py
Measures inference throughput using HuggingFace pipeline.
Note: vLLM had CUDA/NCCL compatibility issues on this cluster.
In production, vLLM would provide 2-4x higher throughput.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str,
                        default="/mnt/data/outputs/finetuned-mistral")
    parser.add_argument("--output_dir", type=str, default="../results")
    parser.add_argument("--batch_sizes", type=int, nargs="+",
                        default=[1, 2, 4, 8, 16])
    parser.add_argument("--input_length", type=int, default=128)
    parser.add_argument("--output_length", type=int, default=128)
    parser.add_argument("--num_requests", type=int, default=50)
    parser.add_argument("--warmup_requests", type=int, default=5)
    return parser.parse_args()


def load_model(model_path):
    print(f"\nLoading base model + LoRA adapter...")
    base_model = "mistralai/Mistral-7B-Instruct-v0.3"
    cache_dir  = "/mnt/data/hf_cache"

    tokenizer = AutoTokenizer.from_pretrained(
        base_model, cache_dir=cache_dir
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype     = torch.bfloat16,
        cache_dir = cache_dir,
        device_map= "auto"
    )
    model = PeftModel.from_pretrained(model, model_path)
    model.eval()

    print(f"[PASS] Model loaded on {next(model.parameters()).device}")
    return model, tokenizer


def generate_prompts(n):
    templates = [
        "### Question:\nUnder the common law, which of the following is true with respect to an offer?\n\nA) An offer can be accepted after rejection\nB) An offer cannot be revoked once made\nC) An offer is binding once communicated\nD) An offer may be revoked before acceptance\n\n### Answer:",
        "### Question:\nWhich of the following best describes promissory estoppel?\n\nA) A promise supported by consideration\nB) A promise enforceable despite lack of consideration\nC) A promise requiring written documentation\nD) A promise made under duress\n\n### Answer:",
        "### Question:\nIn criminal law, which element distinguishes murder from manslaughter?\n\nA) The age of the victim\nB) The use of a weapon\nC) Malice aforethought\nD) The location of the crime\n\n### Answer:",
        "### Question:\nWhich is required for a valid contract?\n\nA) Written documentation only\nB) Notarization by a public official\nC) Offer, acceptance, and consideration\nD) Witnesses present at signing\n\n### Answer:",
    ]
    return [templates[i % len(templates)] for i in range(n)]


def benchmark_batch(model, tokenizer, batch_size,
                    output_length, num_requests, warmup_requests):
    all_prompts = generate_prompts(num_requests + warmup_requests)

    # Warmup
    warmup_inputs = tokenizer(
        all_prompts[:warmup_requests],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256
    ).to(model.device)

    with torch.no_grad():
        _ = model.generate(
            **warmup_inputs,
            max_new_tokens=output_length,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )
    torch.cuda.synchronize()

    # Timed run — process in batches
    timed_prompts = all_prompts[warmup_requests:]
    total_output_tokens = 0
    start = time.perf_counter()

    for i in range(0, len(timed_prompts), batch_size):
        batch = timed_prompts[i:i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=output_length,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )

        # Count only generated tokens
        input_len = inputs["input_ids"].shape[1]
        generated = outputs[:, input_len:]
        total_output_tokens += generated.numel()

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    throughput_toks = total_output_tokens / elapsed
    throughput_reqs = num_requests / elapsed
    latency_ms      = elapsed / num_requests * 1000

    print(f"  Batch size {batch_size:3d}: "
          f"{throughput_toks:.1f} tok/s  "
          f"{throughput_reqs:.2f} req/s  "
          f"{latency_ms:.1f}ms/req")

    return {
        "batch_size":          batch_size,
        "throughput_tokens":   throughput_toks,
        "throughput_requests": throughput_reqs,
        "latency_ms":          latency_ms,
        "total_output_tokens": total_output_tokens,
        "elapsed_seconds":     elapsed,
    }


def main():
    args = parse_args()

    print("\n" + "=" * 60)
    print("INFERENCE THROUGHPUT BENCHMARK (HuggingFace)")
    print("Note: vLLM incompatible with cluster NCCL/CUDA versions")
    print("Production deployment would use vLLM for 2-4x higher throughput")
    print("=" * 60)
    print(f"Model:        {args.model_path}")
    print(f"Batch sizes:  {args.batch_sizes}")
    print(f"Output tokens:{args.output_length}")
    print("=" * 60)

    model, tokenizer = load_model(args.model_path)

    results = []
    print("\nRunning throughput benchmark...")

    for batch_size in args.batch_sizes:
        try:
            result = benchmark_batch(
                model, tokenizer,
                batch_size      = batch_size,
                output_length   = args.output_length,
                num_requests    = args.num_requests,
                warmup_requests = args.warmup_requests
            )
            results.append(result)
        except Exception as e:
            print(f"  Batch size {batch_size} failed: {e}")
            print(f"  Stopping — likely OOM")
            break

    if not results:
        print("[FAIL] No results collected")
        sys.exit(1)

    # Find optimal batch size
    optimal = max(results, key=lambda r: r["throughput_tokens"])

    # Calculate concurrent user estimate
    users = int(
        optimal["throughput_tokens"] / 200 * 2.0
    )

    # Generate report
    W = 62
    report = []
    report.append("")
    report.append("=" * W)
    report.append("  INFERENCE THROUGHPUT BENCHMARK REPORT")
    report.append("=" * W)
    report.append(f"  Model:     {args.model_path}")
    report.append(f"  Engine:    HuggingFace (vLLM incompatible with cluster)")
    report.append(f"  Note:      vLLM would provide 2-4x higher throughput")
    report.append(f"             in production on compatible infrastructure")
    report.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}")
    report.append("=" * W)
    report.append("")
    report.append(
        f"  {'Batch':>6}  {'Throughput':>14}  "
        f"{'Req/s':>8}  {'Latency':>10}"
    )
    report.append("  " + "-" * (W - 4))

    for r in results:
        note = " ← optimal" if r["batch_size"] == optimal["batch_size"] else ""
        report.append(
            f"  {r['batch_size']:>6}  "
            f"{r['throughput_tokens']:>11.1f} t/s  "
            f"{r['throughput_requests']:>6.2f} r/s  "
            f"{r['latency_ms']:>8.1f}ms{note}"
        )

    report.append("  " + "-" * (W - 4))
    report.append("")
    report.append("=" * W)
    report.append("  KEY FINDING")
    report.append("=" * W)
    report.append(f"  Optimal batch size:  {optimal['batch_size']}")
    report.append(
        f"  Peak throughput:     "
        f"{optimal['throughput_tokens']:.1f} tokens/second"
    )
    report.append(
        f"  Request rate:        "
        f"{optimal['throughput_requests']:.2f} requests/second"
    )
    report.append(
        f"  Avg latency:         "
        f"{optimal['latency_ms']:.1f}ms per request"
    )
    report.append(f"  Est. concurrent users: ~{users}")
    report.append(
        f"  (200 tok avg response, 2s acceptable wait)"
    )
    report.append("")
    report.append("=" * W)

    report_str = "\n".join(report)
    print(report_str)

    # Save
    os.makedirs(args.output_dir, exist_ok=True)

    json_file = os.path.join(args.output_dir, "throughput_results.json")
    with open(json_file, "w") as f:
        json.dump({
            "results":            results,
            "optimal_batch_size": optimal["batch_size"],
            "model_path":         args.model_path,
            "engine":             "huggingface",
            "timestamp":          datetime.now().isoformat()
        }, f, indent=2)

    report_file = os.path.join(args.output_dir, "throughput_report.txt")
    with open(report_file, "w") as f:
        f.write(report_str)

    print(f"\nJSON:   {json_file}")
    print(f"Report: {report_file}")


if __name__ == "__main__":
    main()
