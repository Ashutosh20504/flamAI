# A4 — Recommendation memo: tokenizer strategy for Indic serving

**To:** Leadership · **Re:** REPORT_v0 §1 — do not take it to the deck

**Corrected headline numbers.** Same-content cost, measured on 997 parallel
FLORES-200 sentences × 7 languages (English, Hindi, Marathi, Bengali, Kannada,
Tamil, Telugu), 4 tokenizers: for the gpt2-class tokenizer v0 measured, a
same-content Indic request costs **7.5–15.4×** English in tokens (Hindi 7.45×,
Tamil 15.43× — Dravidian languages are *worse* than Hindi, reversing v0's
ordering). With Indic-aware tokenizers the ratio collapses to **1.0–1.4×**
(MuRIL: Hindi 1.16×, Kannada 1.06×; XLM-R: 1.2–1.4×). v0's "5.89×" was
per-word fertility — a denominator that doesn't hold content constant (it
overstates Hindi cost 22% on v0's own data) — and its "6× budget" would
over-provision Indic capacity ~5× for any modern multilingual model.

**Routing recommendation.** Don't build a separate Indic serving path, and
don't accept a 6× premium. Treat same-content token cost as a *model-selection*
criterion: for Indic-heavy traffic, prefer models whose tokenizer achieves
≤1.5× English parity (MuRIL/XLM-R-class vocabularies); a 50k English byte-BPE
model should serve Indic languages only as a fallback, priced at its real
7–15× token cost. If Indic share is material, this criterion will dominate
GPU-hour differences between candidate models.

**Biggest caveat.** This is *input* token cost on formal wiki/news text.
It does not measure (a) generated-token cost, which often dominates and is
model behavior, not tokenizer arithmetic; (b) quality at any fertility level;
(c) casual, code-switched chat ("अच्छा ok fine"), where byte-BPEs fragment
worst — so real-world ratios may exceed these.

**The one metric to monitor in production.** Distribution of
**tokens-per-request by detected language**, tracked as a ratio to English
requests' distribution (weekly p50/p90). It is the same-content comparison
approximated on live traffic, it needs no labels, and it directly tests this
analysis: if Indic/English token ratios drift far above the lab numbers
(>2× for an Indic-aware-tokenizer model), the "parity" conclusion is wrong —
or the traffic is code-switched in ways FLORES never showed us. Alert
threshold: p90 ratio > 2× for two consecutive weeks.
