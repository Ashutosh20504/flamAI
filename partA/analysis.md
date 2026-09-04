# A3 — Corrected analysis: what the numbers actually say

Raw tables (deterministic, reproducible): `results/A3_corrected_analysis.md`,
machine-readable `results/A3_raw_counts.json`. Headline table from §3 of the
results: **tokens per parallel sentence vs English** (same content, paired
sentences, 95% bootstrap CI):

| tokenizer | hin | mar | ben | kan | tam | tel |
|---|---|---|---|---|---|---|
| gpt2 (v0's choice) | 7.45× | 7.79× | 9.66× | 13.59× | 15.43× | 13.04× |
| cl100k_base | 4.80× | 5.03× | 5.92× | 8.88× | 7.61× | 8.35× |
| XLM-R (multilingual) | **1.26×** | 1.22× | 1.38× | 1.37× | 1.35× | 1.33× |
| MuRIL (Indic-aware) | **1.16×** | 1.05× | 1.00× | 1.06× | 1.05× | 1.22× |

Three facts survive every denominator check:

1. **v0's direction was right, its magnitude was badly wrong, and its
   attribution was wrong.** GPT-2-class tokenization of Indic text is genuinely
   expensive (7–15× per same-content request, worse for Dravidian than Hindi —
   the opposite ordering of v0's single-number story). But with an
   Indic-aware tokenizer the same-content cost ratio collapses to **1.0–1.4×**.
   The 5.89× was a property of the tokenizer choice, not "a property of the
   script" — and "any tokenizer will struggle" is measured-false.
2. **Per-word fertility is not a cost number.** Under gpt2 it understates the
   same-content ratio for Hindi by 16% and overstates Kannada by 35% (E4);
   under MuRIL, tok/word says Kannada is 1.44× English while same-content
   tokens say 1.06×. The error sign flips across languages because
   words-per-content differs by language (Hindi 24.7 words/sentence vs English
   21.0 for the same 997 sentences; Kannada 15.5).
3. **Even tok/byte inverts under the right tokenizer**: XLM-R encodes Hindi at
   0.49× English tokens *per byte* — Indic text costs *fewer* tokens per
   transmitted byte. Any per-*size* denominator measures bytes, not requests.

## Which single number should drive the routing-and-cost decision

**Tokens per parallel sentence** — total tokens to encode sentence *i* divided
by total tokens for its English translation, averaged over parallel pairs
(equivalently: the ratio of corpus totals, which is what I report; the two
estimands agree to <2% here).

Reasoning from the denominator up. A routing/cost decision compares "the same
user request, served in language A vs language B", because serving cost is
linear in tokens (prefill FLOPs, KV-cache bytes, decode bandwidth, output
length all scale with token count). So the denominator must hold **content**
constant and vary only the language. Audit the candidates:

- **whitespace word** — holds *nothing* constant: chars/word ranges 5.1 (hin)
  to 9.1 (tam) across our languages, and words-per-content itself differs
  (24.7 vs 21.0 words for the same sentences). This is v0's choice.
- **grapheme cluster** — holds the linguistic character unit constant, but the
  same content is 82.4k graphemes in Hindi and 125.2k in English, so the
  denominator still moves with script, not content.
- **UTF-8 byte** — holds *transmitted size* constant. This is the right
  denominator for bandwidth/network-cost questions, but the same content is
  ~2.6× the bytes in Devanagari, so as a cost proxy it silently multiplies in
  script verbosity. (It's the best *available* proxy when no parallel data
  exists — say so when using it.)
- **parallel sentence** — the only unit where content is held fixed by
  construction. The ratio is exactly "expected tokens for the same request",
  which is exactly "expected serving-cost ratio". Paired bootstrap CIs are
  tight (±1–2%) because every sentence is its own control.

**The decision it drives:** model/tokenizer choice for Indic traffic should be
made on same-content token cost — prefer models with Indic-aware tokenizers
(cost parity 1.0–1.4×) over gpt2-class byte-BPEs (7–15×); *don't* budget a 6×
premium, and *don't* build a separate Indic serving path on the v0 evidence.
Note a tokenizer is not swappable on a fixed model — this number is a
*model-selection* weight, which is what "routing" here actually is.

**What this analysis still cannot tell you** (see CORPUS.md for the full list):
fertility ≠ output quality; prompts ≠ generations (response token length per
language is a separate measurement); FLORES is formal wiki/news register, so
casual code-switched traffic will differ. All three caveats shape the
production monitoring in the A4 memo.
