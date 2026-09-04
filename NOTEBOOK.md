# NOTEBOOK — chronological audit log (2026-09-03)

Format: hypothesis → experiment → result → revision. Dead ends and unit bugs
left in deliberately; they're marked.

---

**15:07 — Setup.** Unpacked starter kit, read the assignment. Ran the untouched
`fertility.py`: it reproduces REPORT_v0's table exactly (eng 1.27 / 0.226, hin
7.45 / 1.579, 5.89×). Good — the audit targets the method, not a transcription
error. Also noticed both sample files contain a double-spaced line; parked as
a suspicion ("split(\" \") smells wrong") for later.

**15:12 — Corpus (A1).** Hypothesis: FLORES-200 dev is the right corpus (7
languages incl. 3 Dravidian; parallel line-alignment is what lets me hold
*content* constant later). Official Meta tarball was only 26 MB; downloaded and
extracted. First extraction attempt failed twice — I'd downloaded the tarball
into the wrong cwd and then tried to extract it from a directory it wasn't in
(sloppy `cd` chain). Fixed by moving the file; lesson: pin absolute paths.

**15:20 — Surprise #1 (NFC is not a no-op).** Hypothesis: the intern's
`unicodedata.normalize("NFC", …)` line is harmless boilerplate. Measured:
raw FLORES-200 files are mixed Unicode-form — only **411/997 Bengali lines**
are already NFC (Hindi 907/997). Revision: NFC is *load-bearing* on real
corpora; on the intern's 10-line sample it is a strict no-op (96/96, 459/459
tokens identical). So it did not distort v0's numbers AND removing it would
introduce a bias. Filed under "suspicious but actually fine", with the
+0.53%-tokens/+1.19%-bytes Bengali measurement as evidence.

**15:30 — Dead end (initial A2 hypothesis).** I expected the double-space
split bug to *inflate* reported Hindi fertility (more words ⇒ wait, no: more
words ⇒ *lower* fertility). Caught the sign error before running anything by
writing the direction down first. Prediction fixed: bug deflates fertility.
E2 run: eng 1.229 → 1.247 (+1.5%), hin 7.448 → 7.598 (+2.0%) when fixed.
Both corpora have exactly one phantom word, so the biases nearly cancel in
the ratio — which is presumably why nobody noticed.

**15:33 — E5 (lowercase).** Measured: `.lower()` moves *English* fertility
+2.9% (byte-BPE merges fewer lowercase common words), Hindi unchanged (no
case). Headline ratio 6.06× → 5.89×. Verdict: defensible normalizer, not a
bug — but undisclosed in the report, and it slightly *flatters* the headline
gap... in the direction of understating it.

**15:36 — Surprise #2 (the report refutes itself).** Wrote E3 to check
whether tok/char "confirms" tok/word. Per line the identity
`tok/word ≡ tok/char × chars/word` is exact, so the two columns cannot be
independent evidence. Then the bonus: chars/word from v0's own table is
**5.71 (eng) vs 4.71 (hin)** — Hindi words are *shorter* than English words in
characters. REPORT_v0's root-cause claim ("Hindi has more Unicode characters
per word") is contradicted by REPORT_v0's own numbers. Verified the same
direction on FLORES (5.97 vs 5.10).

**15:40 — E4 (the conceptual flaw).** Hypothesis: per-word fertility
overstates the same-content cost gap (Hindi packs the same content into fewer
words). On the intern's parallel 10-liner: fertility ratio 5.89× vs
tokens-per-parallel-sentence **4.81×** [4.09–5.58] — yes, +22% overstatement.
Then the revision: on FLORES the bias **flips sign by language** — per-word
*understates* Hindi (6.33× vs 7.53× same-content) and *overstates* Kannada by
35%, Tamil 28%, Telugu 27%. A metric with language-dependent error sign cannot
rank costs. (Initial scratch run had a list-vs-array bug in my bootstrap
helper; fixed before recording numbers.)

**15:45 — A3 build-out.** 4 tokenizers × 4 denominators on FLORES-200
(`run_corrected_analysis.py`). First run crashed: I passed `cl100k` to
tiktoken (correct name: `cl100k_base`). Second issue, caught on inspection:
my bootstrap CI used mean-of-per-sentence-ratios while the point estimate was
ratio-of-means, so some CIs didn't bracket their point (e.g. 7.45x [7.44,
7.62]). Fixed the estimand to ratio-of-means throughout. Headline (tok per
parallel sentence vs English): gpt2 7.45–15.43×; cl100k 4.80–8.88×; XLM-R
1.22–1.38×; MuRIL 1.00–1.22×. Decision: **tokens per parallel sentence** is
the routing number because it is the only denominator holding *content*
constant; written up in `partA/analysis.md`.

**15:52 — Part B.** Wrote `b_calcs.py` to force every claim through the log.
B1: KV = 112 KiB/token; pool 12.08 GB ⇒ ~25 concurrent 4096-token sequences.
The real win: predicted `kv_cache_util` for **all 11 preemption-free rows
matches the log to ±0.01** (e.g. 0.933 vs 0.93), and the two rows predicted
over-capacity (32, 48) are exactly the preemption rows. B3: reconstructed
`reported_tok_s = n×(prompt+gen)/wall` for all 13 rows to ≤0.02% — the column
counts prompt+completion tokens. Honest goodput of the batch-24 long row:
**200.9 tok/s** (tokens/wall) and **249.8 tok/s** (batch ÷ p50 ITL); the gap
decomposes into ~49 s decode + ~12 s prefill ⇒ prefill ≈ 7.2k tok/s ≈ 50% of
L4 fp16 peak (plausibility check passed).

**15:58 — Dead end (unit bug, caught by my own consistency check).** First
version of the B3 goodput reconciliation divided by 1000 twice and produced
"prefill rate −2 tok/s". The *plausibility check I'd built in* flagged it.
Fixed: decode time = 12,288/249.8 = 49.2 s. Kept the check in the script —
this is exactly the class of error the assignment says to hunt.

**16:05 — Part C + memos.** Chose (c)-then-(a): prompt-engineering baseline
on day 1 (measured, cheap), LoRA SFT on synthetic casualized pairs as the
main path scoped to reviewer-covered languages; rejected the ≤1B rewriter on
serving arithmetic (same data problem, +60–100 ms, compounded errors).
Kill criteria and day-1 experiment written into `partC/memo.md`.

**16:10 — Open threads (honest unknowns).**
- tok/grapheme uses `regex \X`; a few FLORES sentences contain ZWJ sequences
  where native-speaker intuition of "character" may differ from \\X. Didn't
  chase; doesn't affect the headline (per-sentence) numbers.
- I did not verify vLLM's preemption mode is recompute (vs swap) for the
  bench's engine version — B2's mechanism says "default recompute mode" and
  the wall-clock excess (~29 s vs ~122 s expectation) is consistent with
  either, with recompute the likelier fit. If asked in the defense: pull
  `vllm:num_preemptions_total` + prefill-token counters to disambiguate.
- cl100k's Tamil number (7.61×) is *lower* than Kannada (8.88×) despite
  Tamil's higher byte count — plausible (vocab coverage quirks), but if this
  were a paper I'd want a second multilingual BPE to confirm the ordering.
