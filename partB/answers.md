# Part B — Capacity reconciliation

All arithmetic below is produced and verified by `b_calcs.py`
(`b_calcs_output.txt` is its captured output); every prediction is checked
against `bench_log.csv` in the script.

## B1 — KV-cache arithmetic and max concurrency

**(a) KV-cache bytes per token, exactly.** The cache stores K and V for every
layer, but only for the **KV heads** (GQA: 8, not the 24 Q heads):

```
2 (K+V) × 28 layers × 8 kv_heads × 128 head_dim × 2 bytes (fp16)
= 57,344 elements × 2 B = 114,688 bytes = 112 KiB per token
```

**(b) Max concurrent 4096-token sequences.**

```
VRAM budget   = 24 GB × 0.92 (gpu_memory_utilization)      = 22.08 GB
weights       = 4.2 B params × 2 B (fp16)                  =  8.40 GB
runtime overhead (spec)                                    =  1.60 GB
KV pool       = 22.08 − 8.40 − 1.60                        = 12.08 GB

token capacity = 12.08e9 / 114,688           ≈ 105,329 tokens
max seqs       = floor(105,329 / 4096)       ≈ 25.7  →  **25 concurrent sequences**
```

**Log check (stronger than a spot check).** If this pool size is right, the
logged `kv_cache_util` of *every* preemption-free row should equal
`batch × (prompt+gen) × 114,688 / 12.08 GB`. It does — 11/11 rows match to
±0.01 (e.g. batch 24 long: predicted 0.933, logged 0.93; batch 64 short:
0.47/0.47). The first rows to *exceed* the pool (batch 32: 15.0 GB needed;
batch 48: 22.6 GB needed vs 12.08 GB available) are exactly the rows with
`preempted_seqs = 7` and `23`. The model spec and the log are mutually
consistent, and real capacity is ~25, not "batch size you submitted".

## B2 — The long-context sweep anomaly

**The anomaly.** Naive "throughput scales with batch" expects monotonic
increase. The long-prompt sweep rises 565 → 903 → 1311 → **1607** tok/s
(batch 4→24), then *falls* to 1384 (batch 32) and 1298 (batch 48) — a peak at
batch 24, with wall clock ballooning 61.2 → 94.7 → 151.4 s.

**Mechanism (from the log's own columns).** At batch ≥ 32 the KV requirement
(32 × 4096 × 114,688 B = 15.0 GB; 48 × → 22.6 GB) exceeds the 12.08 GB pool.
`kv_cache_util` pins at 0.97 and the scheduler **preempts** sequences
(`preempted_seqs`: 7 and 23). vLLM's default preemption mode is *recomputation*:
the victim's KV blocks are discarded and the prompt is re-prefilled from
scratch when the sequence resumes — so a large slice of the run's compute is
duplicate prefill work, and decode stalls while victims wait. Effective
concurrency is capped at ~25 sequences regardless of submitted batch size; the
extra requests mostly add queueing and recompute. Scale check: a
preemption-free batch-48 run should take ≈ 2 × 61.2 = ~122 s; it took 151.4 s —
~24% of the run is preemption overhead.

**One config change with a predicted effect.** Cap `max_num_seqs` at 24 for
4k-context traffic (equivalently: keep batch ≤ KV capacity). Prediction for
the batch-48 load: two full waves → wall ≈ 122 s, total tok/s ≈ **1,607**
(the batch-24 plateau; +24% vs the measured 1,298), decode goodput ≈ **201**
tok/s (vs 163 measured), `preempted_seqs` 23 → 0. (Alternative with the same
mechanism: `max_model_len` 2048 → capacity floor(12.08e9 / 235 MB) = 51
sequences, batch 48 fits without preemption, at the cost of rejecting
longer prompts.)

## B3 — The misread column, honest goodput, and what §2 should have said

**The column: `reported_tok_s` counts prompt **and** completion tokens.**
Proof: reconstruct every row as `n × (prompt_len + gen_len) / wall_clock_s` —
all 13 rows match the logged value to ≤0.02% (table in `b_calcs_output.txt`),
including the two the report quotes (16 short: 883.4 vs 883.2; 16 long:
1311.5 vs 1311.4). The report read this *total* token throughput as if it were
generation throughput. Longer prompts inflate the numerator with prefill
tokens — that's the whole reason "1311 vs 883" appeared.

**Honest goodput of the batch-24 long-prompt row — two independent derivations:**

1. Generated tokens ÷ wall clock: `24 × 512 / 61.16 s` = **200.9 tok/s**.
2. From the decode-phase counter: `24 sequences ÷ itl_ms_p50` = `24 × (1000/96.07)` = **249.8 tok/s** (rate while actually decoding; p50 ITL is per-sequence, so aggregate decode goodput ≈ sequences / ITL).

These two agree to ~20% once you account for prefill time: (2) implies decode
time = 12,288/249.8 ≈ 49.2 s, leaving ≈ 12.0 s of prefill+overhead — i.e. a
prefill rate of ~7,200 tok/s, ≈ 50% of the L4's 121 TFLOPS fp16 peak for a 4.2B
model, which is plausible. Both numbers say the same thing: **the user-visible
generation rate of that row is ~200 tok/s, not 1607.**

**What §2 should have said:** longer prompts make *total token* throughput look
better only because the counter is dominated by prefill compute; user-facing
decode goodput actually *falls* (294.6 tok/s at batch 16/short vs 200.9 at
batch 24/long, with p50 ITL doubling 48.3 → 96.1 ms). And batch 48 cannot
deliver ~3200 tok/s: 3200 was never observed anywhere in the log (best total
throughput is 2,267 on the short sweep), and on long prompts the engine
*degrades* past batch 24 because KV capacity is ~25 sequences — batch 48
measured 1,298 tok/s total and ~163 tok/s of decode goodput.

## B4 — The one counter to pull

`vllm:num_preemptions_total` (the log's `preempted_seqs` is its per-run value).
Expected reading: exactly **0** for any concurrency ≤ ~25 at 4096-token
sequences, then a step to **7** (batch 32) and **23** (batch 48) — precisely
the logged pattern. A companion gauge `vllm:gpu_cache_usage_perc` pinned at
~0.97 while preemptions climb would pin the mechanism to KV exhaustion rather
than CUDA-graph launch stalls or CPU scheduling — which is what distinguishes
this explanation from "the GPU is just slow at batch 48".
