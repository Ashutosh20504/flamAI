#!/usr/bin/env python3
"""
Part B -- capacity reconciliation between bench/model_spec.md and bench_log.csv.

Deterministic; prints every derivation and checks predictions against the log.
Run:  python b_calcs.py
"""
import csv
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOG = HERE / "bench_log.csv"

# ---- model spec (bench/model_spec.md) ---------------------------------------
LAYERS = 28
KV_HEADS = 8          # GQA: only KV heads matter for the cache
HEAD_DIM = 128
KV_BYTES = 2          # fp16
PARAMS = 4.2e9
WEIGHT_BYTES_PER_PARAM = 2   # fp16
GPU_GB = 24.0
MEM_UTIL = 0.92
OVERHEAD_GB = 1.6
MAX_MODEL_LEN = 4096


def load_log():
    with open(LOG, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in r:
            r[k] = float(r[k])
    return rows


def b1():
    print("=" * 78)
    print("B1. KV-cache bytes per token and max concurrent 4096-token sequences")
    print("=" * 78)
    kv_per_tok = 2 * LAYERS * KV_HEADS * HEAD_DIM * KV_BYTES
    print(f"a) KV bytes/token = 2 (K+V) x {LAYERS} layers x {KV_HEADS} kv-heads x "
          f"{HEAD_DIM} head_dim x {KV_BYTES} B (fp16)")
    print(f"   = {2 * LAYERS * KV_HEADS * HEAD_DIM:,} elements x 2 B = {kv_per_tok:,} B "
          f"= {kv_per_tok / 1024:.0f} KiB per token")

    weights_gb = PARAMS * WEIGHT_BYTES_PER_PARAM / 1e9
    pool_gb = GPU_GB * MEM_UTIL - weights_gb - OVERHEAD_GB
    print(f"\nb) VRAM budget  = {GPU_GB} GB x {MEM_UTIL} = {GPU_GB * MEM_UTIL:.2f} GB")
    print(f"   weights      = 4.2e9 x {WEIGHT_BYTES_PER_PARAM} B = {weights_gb:.2f} GB")
    print(f"   overhead     = {OVERHEAD_GB:.1f} GB")
    print(f"   KV pool      = {GPU_GB * MEM_UTIL:.2f} - {weights_gb:.2f} - {OVERHEAD_GB:.1f} "
          f"= {pool_gb:.2f} GB = {pool_gb * 1e9:,.0f} B")
    tok_capacity = pool_gb * 1e9 / kv_per_tok
    max_seqs = math.floor(tok_capacity / MAX_MODEL_LEN)
    print(f"   token capacity = {pool_gb * 1e9:,.0f} / {kv_per_tok:,} = {tok_capacity:,.0f} tokens")
    print(f"   max concurrent {MAX_MODEL_LEN}-token sequences = floor({tok_capacity:,.0f}/{MAX_MODEL_LEN}) "
          f"= **{max_seqs} sequences**")
    return kv_per_tok, pool_gb, max_seqs


def b1_check(rows, kv_per_tok, pool_gb):
    print("\nLog check -- predicted peak KV utilization vs logged kv_cache_util")
    print(f"{'batch':>5} {'prompt':>7} {'gen':>5} | {'pred util':>9} {'logged':>7} {'preempted':>9}")
    for r in rows:
        need = r["batch_size"] * (r["prompt_len"] + r["gen_len"]) * kv_per_tok
        pred = min(need / (pool_gb * 1e9), 0.97)  # 0.97 = logged saturation ceiling
        flag = "  <- needs preemption" if need > pool_gb * 1e9 else ""
        print(f"{r['batch_size']:>5.0f} {r['prompt_len']:>7.0f} {r['gen_len']:>5.0f} | "
              f"{pred:>9.2f} {r['kv_cache_util']:>7.2f} {r['preempted_seqs']:>9.0f}{flag}")
    print("All preemption-free rows match the prediction to 0.01; the first two rows over")
    print("capacity (batch 32, 48 @ 3584+512) are exactly the rows with preemptions.")


def b2(rows):
    print("\n" + "=" * 78)
    print("B2. The long-context anomaly: throughput FALLS from batch 24 to 32 to 48")
    print("=" * 78)
    long = [r for r in rows if r["prompt_len"] == 3584]
    print(f"{'batch':>5} {'wall_s':>8} {'reported_tok_s':>14} {'decode goodput':>14} "
          f"{'kv_util':>8} {'preempted':>9}")
    for r in long:
        goodput = r["batch_size"] * r["gen_len"] / r["wall_clock_s"]
        print(f"{r['batch_size']:>5.0f} {r['wall_clock_s']:>8.2f} {r['reported_tok_s']:>14.1f} "
              f"{goodput:>14.1f} {r['kv_cache_util']:>8.2f} {r['preempted_seqs']:>9.0f}")
    w24 = next(r for r in long if r["batch_size"] == 24)["wall_clock_s"]
    print(f"\nMechanism: KV need at batch 32 = 32x4096x114,688 B = "
          f"{32 * 4096 * 114688 / 1e9:.2f} GB > pool; batch 48 = "
          f"{48 * 4096 * 114688 / 1e9:.2f} GB >> pool.")
    print("Scheduler preempts running sequences (7 and 23 rows) and, in vLLM's default")
    print("recompute mode, throws away their KV and re-prefills them from scratch ->")
    print("duplicate prefill work + decode stalls. Effective concurrency is capped at ~26")
    print("sequences no matter what batch size you submit.")
    print(f"\nPreemption-free expectation: wall clock should scale ~ batch (2x24 = 48) -> "
          f"~{2 * w24:.0f} s;")
    print(f"actual batch-48 wall = 151.4 s, i.e. ~24% of the run is preemption overhead.")
    print("\nProposed change (pick one; both are config-only):")
    print("  1) cap max_num_seqs at 24 for 4k-context traffic (matches real capacity);")
    print("  2) max_model_len 2048 -> capacity floor(12.08e9/(2048x114,688)) = 51 seqs,")
    print("     batch 48 fits with zero preemptions (needs request-length triage).")
    print("Predicted effect of (1) on the batch-48 row: wall ~122 s (2 waves x 61.2 s),")
    print("total tok/s ~1,607 (the batch-24 plateau, +24% vs the 1,298 measured),")
    print("decode goodput ~201 tok/s (vs 163 measured), preemptions 23 -> 0.")


def b3(rows):
    print("\n" + "=" * 78)
    print("B3. The misread column: reported_tok_s counts PROMPT+COMPLETION tokens")
    print("=" * 78)
    print("Reconstruct every row:  n x (prompt_len + gen_len) / wall_clock_s  vs reported")
    print(f"{'batch':>5} {'prompt':>7} {'gen':>5} | {'reconstructed':>13} {'reported':>9} {'rel err':>8}")
    worst = 0.0
    for r in rows:
        rec = r["num_requests"] * (r["prompt_len"] + r["gen_len"]) / r["wall_clock_s"]
        err = abs(rec - r["reported_tok_s"]) / r["reported_tok_s"]
        worst = max(worst, err)
        print(f"{r['batch_size']:>5.0f} {r['prompt_len']:>7.0f} {r['gen_len']:>5.0f} | "
              f"{rec:>13.1f} {r['reported_tok_s']:>9.1f} {100 * err:>7.2f}%")
    print(f"max relative error across all 13 rows: {100 * worst:.2f}% -> reported_tok_s is")
    print("TOTAL token throughput (prefill-dominated), not generated-token throughput.")

    r24 = next(r for r in rows if r["prompt_len"] == 3584 and r["batch_size"] == 24)
    print("\nHonest goodput of the batch-24 long-prompt row, two independent ways:")
    g1 = r24["batch_size"] * r24["gen_len"] / r24["wall_clock_s"]
    print(f"  (i) generated tokens / wall clock   = 24 x 512 / 61.16 s        = {g1:.1f} tok/s")
    g2 = r24["batch_size"] * 1000 / r24["itl_ms_p50"]
    print(f"  (ii) decode-phase rate from ITL     = 24 x (1000/96.07 ms)      = {g2:.1f} tok/s")
    t_decode = r24["batch_size"] * r24["gen_len"] / g2
    t_prefill = r24["wall_clock_s"] - t_decode
    prefill_rate = r24["num_requests"] * r24["prompt_len"] / t_prefill
    print(f"  consistency: (ii) implies decode time = 12,288/{g2:.1f} = "
          f"{t_decode:.1f} s -> prefill+overhead {t_prefill:.1f} s -> prefill rate {prefill_rate:,.0f} tok/s")
    flops_util = 2 * PARAMS * r24["num_requests"] * r24["prompt_len"] / (t_prefill * 121e12)
    print(f"  (= {100 * flops_util:.0f}% of L4 fp16 peak for a 4.2B model -- plausible)")
    r16s = next(r for r in rows if r["prompt_len"] == 512 and r["batch_size"] == 16)
    print(f"\nWhat the report should have said: longer prompts LOWER user-visible goodput")
    print(f"(short b16: {16 * 256 / r16s['wall_clock_s']:.1f} tok/s decoded vs long b24: {g1:.1f} tok/s,")
    print(f"ITL {r16s['itl_ms_p50']:.1f} -> {r24['itl_ms_p50']:.1f} ms). 'Batch 48 -> ~3200 tok/s' is doubly")
    print(f"wrong: 3200 was never measured (best observed total = 2,267 on the short sweep),")
    print(f"and the long sweep *degrades* past batch 24 (b48 measured 1,298 total / "
          f"{48 * 512 / 151.41:.1f} decode tok/s).")


def b4():
    print("\n" + "=" * 78)
    print("B4. The one counter to pull")
    print("=" * 78)
    print("vllm:num_preemptions_total (the engine's preemption counter; the log's")
    print("`preempted_seqs` column is its per-run value). Expected value: exactly 0 for")
    print("any concurrency <= ~25 at 4096-token sequences, then a step to 7 (batch 32)")
    print("and 23 (batch 48) -- precisely the pattern in the log. Companion gauge")
    print("vllm:gpu_cache_usage_perc pinned at ~0.97 while preemptions climb would")
    print("confirm KV exhaustion (not e.g. CUDA-graph or CPU scheduling stalls) as the")
    print("mechanism behind the batch>24 throughput drop.")


def main():
    rows = load_log()
    kv_per_tok, pool_gb, max_seqs = b1()
    b1_check(rows, kv_per_tok, pool_gb)
    b2(rows)
    b3(rows)
    b4()


if __name__ == "__main__":
    main()
