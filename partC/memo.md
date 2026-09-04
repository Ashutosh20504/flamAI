# Part C — Decision memo: casual conversational tone in hi, kn, ta, te, bn, mr

**Recommendation: (c) prompt-engineering on day 1 as the measured baseline, then (a) a LoRA SFT pass on synthetic casualized pairs as the main path — scoped to reviewer-covered languages first. Not (b).**

**Assumptions.** (1) The main assistant is an open-weights instruct model we
serve ourselves (no external API budget ⇒ all data generation happens on our
A100). (2) "Casual" is definable by a 1-page rubric (contractions, direct
address, sentence fragments, lighter honorifics — not slang-max). (3) The
reviewer (hi+kn, 10 h/wk) is the quality bottleneck ⇒ hi/kn are
launch-blocking; ta/te/bn/mr ship behind a flag. (4) Launch review in 3 weeks
is a checkpoint, not a freeze.

**Back-of-envelope.**
- *Data volume:* 6 langs × 2,500 seed prompts (mined from product logs) → main
  model regenerates each casually (~150 tok out, ~400 tok prompt) ≈ 15k × 550
  ≈ 8M tokens. At a conservative ~1,500 tok/s batched decode on A100-80GB,
  ~1.5 h; ×5 for rejection-sampling/filter passes ≈ **1 GPU-day of data
  engine**. Keep ~800 reviewer-approved pairs per covered language + ~2k
  auto-filtered pairs per uncovered language ≈ ~12k training pairs.
- *Reviewer throughput:* 10 h/wk × 2 usable weeks = 20 h; at ~90 s/pair
  (rubric rating + fix) ≈ **800 pairs reviewed** — enough to validate hi+kn
  sets and calibrate the judge, not enough to review all 6 languages (hence
  flag-gating).
- *Training cost:* LoRA (r=16) on the main model, 12k pairs × ~600 tok × 2
  epochs ≈ 15M tokens ≈ 8–10k tok/s on A100 ⇒ **under 1 GPU-hour per run**;
  dozens of iterations fit in the 2-week window. Serving: adapter adds
  ~1–2% VRAM, no added latency.
- *Why not (b):* a ≤1B rewriter needs the *same* casual data (in harder
  parallel form), adds a full extra forward pass per reply (~60–100 ms + one
  more KV slot on the same GPU), and compounds the base model's errors —
  strictly worse economics than fixing the model once.
- *Why not (c) alone:* prompts are ~free and ship day 1, but formality is
  substantially baked into weights; (c) is the floor, not the fix.

**Success metric (numeric).** On a frozen 200-prompt set per language
(100 casual-intent / 100 formal-intent), reviewer-rated: **casualness win-rate
of SFT vs prompt-eng-only ≥ 70% on hi and kn, with formality regressions on
formal-intent prompts ≤ 5%**. Flag-gated languages: LLM-judge (calibrated on
the reviewer's hi/kn labels) ≥ 65% agreement, not launch-blocking.

**Kill criterion.** If the *first* LoRA run (trained by day 5 on the
reviewer-approved hi/kn pairs) scores **< 55% win-rate on hi/kn — i.e. barely
above the prompt-eng baseline — or formality regressions > 10%**, abandon (a)
and ship (c) + style guide; decision by **end of week 1**. Secondary kill:
if the day-1 baseline already shows prompt-engineering achieving ≥ 70%
casualness, SFT is unnecessary risk — ship (c).

**Day-1 experiment.** Spend 4 reviewer hours building the 100-prompt hi/kn
eval set and calibrating the rubric; run the current model against 3
prompt-engineering variants; measure casualness rate. This quantifies the
headroom SFT must clear, prices the reviewer's labeling speed (validating the
800-pair budget), and produces the baseline numbers the kill criterion reads
against — all before any training GPU-hour is spent.
