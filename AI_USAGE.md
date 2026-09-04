# AI_USAGE.md

Honest account of AI's role in this submission.

## Where AI did the work

An AI coding agent (ZCode, GLM model) drove this audit end-to-end in one
working session at my direction: environment setup, locating and downloading
FLORES-200, writing the experiment scripts and the corrected-analysis
pipeline, and drafting the Part B/C memos and this notebook. The compute is
real — every number in this repo comes from an executed command, and each
analysis script is deterministic and re-runnable (`python` entry points are
listed in `README.md`).

## Where AI was wrong or nearly wrong (and how it got caught)

- **Sign error, caught on paper.** The agent's first stated hypothesis was
  that the double-space `split(" ")` bug *inflates* fertility. Writing down
  the predicted direction before running (more phantom words ⇒ lower
  fertility) caught it pre-execution. This is why every A2 entry in
  `partA/audit.md` states direction + magnitude.
- **Unit bug in the B3 reconciliation.** The first goodput consistency check
  printed "prefill rate −2 tok/s" (a double 1000-division). A built-in
  plausibility check (~50% of L4 fp16 peak) caught it; fixed and re-run.
- **Statistical estimand mismatch.** The first A3 bootstrap computed
  mean-of-per-sentence-ratios while the headline was ratio-of-means, so CIs
  didn't bracket their points (7.45x [7.44, 7.62]). Caught by inspection,
  fixed to bootstrap the ratio of means.
- **Mechanical failures**, honestly: wrong cwd when downloading/extracting
  the corpus (two failed extractions), and `cl100k` vs `cl100k_base` naming.
- **A near-miss claim.** The agent initially wanted to flag NFC normalization
  and the unused `random.seed(1337)` as "findings". Measurement showed NFC is
  a strict no-op on the intern's data but load-bearing on real corpora (411
  of 997 raw Bengali lines are NFD), and `random` is dead code with exactly
  zero effect. Both are filed as "audited, fine" — flagging them as bugs
  would have been wrong under the evidence rule.

## What I verified and own

Before submitting I re-ran every script myself and re-derived by hand: the
KV-cache formula (2 × 28 layers × 8 KV heads × 128 head_dim × 2 B = 112
KiB/token), the 12.08 GB pool → ~25-sequence capacity, and the two batch-24
goodput derivations (24×512/61.16 = 200.9 tok/s; 24/(0.09607 s) = 249.8
tok/s). The claims I can defend without notes: why per-word fertility cannot
rank serving costs (its error changes sign across languages — measured), why
"tok/char confirms tok/word" is an identity rather than evidence, why
`reported_tok_s` must be prompt+completion (it reconstructs all 13 rows to
≤0.02%), and why the routing number is tokens-per-parallel-sentence (only
denominator that holds content constant).
