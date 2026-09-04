# A2 — Audit of `fertility.py` and REPORT_v0 §1

Every claim below ships with an isolated experiment (`experiments/`), the
before/after numbers, and the direction + magnitude of the distortion. Claims
I could **not** support with measurement are in §4 (audited, found fine) —
flagging those as bugs would be wrong, and I'm explicitly *not* making them.

Baseline check first: running the untouched script
(`python fertility.py --corpus eng=... --corpus hin=... --tokenizer gpt2`)
reproduces the report exactly: eng **1.27 / 0.226**, hin **7.45 / 1.579**,
ratio **5.89×**. So the audit is aimed at the code that produced these numbers,
not at a transcription error.

## 1. Confirmed code bug — `line.split(" ")` counts phantom words

**Claim.** `fertility.py:62` splits on a single literal space. Any line with a
repeated space produces empty strings in the word list, overcounting words and
**deflating fertility**. It also silently treats tabs as word-internal.

**Evidence (E2, `experiments/e2_split_bug.py`).** Both sample files happen to
contain a double-spaced line ("Please keep the books  in the cupboard." /
"किताबें  अलमारी में रखी हैं।"). Changing *only* the splitter:

| corpus | fertility, v0 splitter | fertility, `split()` | delta |
|---|---|---|---|
| eng (10 lines) | 1.229 | 1.247 | **+1.5%** |
| hin (10 lines) | 7.448 | 7.598 | **+2.0%** |

**Direction/magnitude:** fertility biased **low** by ~1.5–2% on this corpus;
the effect scales with the frequency of doubled whitespace, so on messy
production text it can be larger. On the ratio hin/eng the two biases nearly
cancel here (both corpora have exactly one phantom word) — which is exactly why
nobody noticed. One line to fix: `words = line.split()`.

## 2. Confirmed conceptual flaw — the denominator doesn't hold content constant

**Claim.** The script computes exactly what it says — tokens per *whitespace
word* — and that is the wrong thing to compute for a routing/cost decision.
Whitespace words are not a constant unit of content across languages: the same
utterance has a different number of words in Hindi vs English (and the word
*lengths* differ by script — see A3's structure table). So "Hindi fertility is
X× English" is not "Hindi costs X× per request".

**Evidence (E4, `experiments/e4_denominator.py`).** Hold content fixed by using
parallel line *i* in each language and compare token counts directly.

On the intern's **own 10-line parallel corpus** (gpt2, v0 settings):

| metric | value |
|---|---|
| per-word fertility ratio (v0's headline) | **5.89×** |
| tokens per **parallel sentence** (same content) | **4.81×** [95% CI 4.09–5.58] |

→ v0 **overstates the per-request cost gap by ~22% on its own data**. Mechanism:
Hindi expresses the same content in *fewer* words (6.1 words/sentence vs 7.8
English), so dividing by words inflates Hindi.

The distortion is not even one-directional: on FLORES-200 (997 parallel
sentences, gpt2) the per-word ratio **understates** the same-content token ratio
for Hindi by 16% (6.33× vs 7.53×) and **overstates** it for Kannada by 35%
(18.55× vs 13.76×), Tamil 28%, Telugu 27%. A metric whose error changes sign
across languages cannot rank serving costs.

**Why the report's "confirmation" is not evidence (E3).** REPORT_v0 says
tok/char "agrees" with tok/word, "so the result is robust". Per line,
`tokens/word ≡ tokens/char × chars/word` — the two columns are one measurement
plus a length statistic. They *cannot* disagree except through line-weighting
(the residual is ≤1.8% from macro-averaging). Agreement of two algebraically
linked numbers is an identity, not replication.

**And the report's root-cause claim is contradicted by its own table (E3).**
"Root cause: Hindi simply has more Unicode characters per word" — the report's
own numbers give chars/word eng = 5.71 vs hin = 4.71 (ratio **0.83**, i.e.
Hindi words are *shorter* in characters). On FLORES: 5.10 vs 5.97. The claimed
mechanism is false in both datasets.

## 3. Confirmed report error (follows from #2 + A3) — "6× cost", "any tokenizer"

"Any tokenizer will struggle — this is a property of the script, not the
tokenizer" is falsified by measurement: on FLORES-200 parallel sentences,
XLM-R's Hindi/English ratio is **1.26×** and MuRIL's **1.16×** (see
`results/A3_corrected_analysis.md`). The 5.89× is a property of *gpt2's 50k
English-heavy byte-BPE vocabulary*, not of Hindi. The "budget 6× serving cost"
recommendation is off by ~5× for any Indic-aware tokenizer and points the
routing decision in an unnecessarily expensive direction.

## 4. Audited and found FINE (do not "fix"; flagging these would be wrong)

- **NFC normalization (`fertility.py:49`) — correct and necessary.** Suspicious
  because it rewrites every line. Evidence (E6, `experiments/e6_nfc.py`): on the
  intern's sample it is a strict no-op (0 lines changed, token counts identical:
  96/96 eng, 459/459 hin) — so it did **not** distort v0's numbers. On real
  multilingual text it is load-bearing: raw FLORES-200 contains NFD sequences
  (only 411/997 Bengali lines are already NFC); without NFC, Bengali token/byte
  counts shift +0.5%/+1.2% — a language-asymmetric confound. NFC *removes* a
  bias; removing this line would introduce one.
- **`random.seed(1337)` + `import random` — dead code, zero effect.**
  `grep -n random fertility.py` → lines 21, 25 only; nothing in the program is
  random (no sampling, no shuffling). Measured effect on outputs: exactly zero.
  Smell, not bug.
- **Lowercasing (`fertility.py:60`) — defensible normalizer, but it is not
  free and the report doesn't mention it.** Evidence (E5): lowercasing moves
  *English* fertility +2.9% (1.229 → 1.265; byte-BPE merges fewer lowercase
  common words), Hindi unchanged (no case), so the headline ratio moves
  6.06× → 5.89× (−2.8%). It's applied symmetrically and documented in the
  comment; fine — but any corrected analysis must either lowercase both sides
  or neither, and say so. (A3 does: no lowercasing, since Indic scripts have
  no case and lowercasing only perturbs the baseline.)
- **Macro (per-line) averaging of fertility instead of corpus totals** — a
  choice, not a bug: it weights a 5-word line equal to a 40-word line. Measured
  effect on v0's numbers vs micro-average: eng 1.27 vs 1.22, hin 7.45 vs 7.40.
  Small, disclosed here, immaterial to any conclusion.
- **tiktoken byte-fallback "exploding" Devanagari into 2–3 byte tokens** — that
  is what byte-level BPE *is*; the mechanism is real, the conclusion drawn from
  it (§3 above) is what's wrong.

## Verdict on REPORT_v0 §1

The 5.89× number is reproducible but answers the wrong question, its
"confirmation" is an identity, its root-cause claim is refuted by its own table,
and its 6× budget recommendation would misallocate capacity by ~5× for any
Indic-aware tokenizer. Corrected numbers and the number that *should* drive the
decision: see `analysis.md` (A3).
