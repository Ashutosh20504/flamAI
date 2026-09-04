# A3 — Corrected cross-language tokenizer analysis (FLORES-200 dev, 997 parallel sentences × 7 languages)

Reproduce: `python run_corrected_analysis.py` (deterministic; seed 1337).

## 1. Corpus structure (the confound v0 ignored)

| lang | sentences | words | graphemes | unicode chars | utf-8 bytes | chars/word | graph/word |
|---|---|---|---|---|---|---|---|
| eng | 997 | 20,954 | 125,194 | 125,194 | 125,290 | 5.97 | 5.97 |
| hin | 997 | 24,607 | 82,404 | 125,495 | 322,640 | 5.10 | 3.35 |
| mar | 997 | 18,065 | 76,101 | 125,698 | 335,677 | 6.96 | 4.21 |
| ben | 997 | 18,756 | 78,123 | 124,583 | 333,496 | 6.64 | 4.17 |
| kan | 997 | 15,430 | 86,177 | 131,749 | 357,408 | 8.54 | 5.59 |
| tam | 997 | 16,134 | 94,467 | 146,126 | 398,795 | 9.06 | 5.86 |
| tel | 997 | 16,388 | 73,568 | 127,172 | 338,804 | 7.76 | 4.49 |

Reading: whitespace words are *longer* in the Dravidian languages (8.5–9.1 chars/word vs 5.97 English) and *shorter* in Hindi (5.10). A denominator that scales with word length cannot compare tokenization quality across languages.

## 2. Tokens per unit, all languages (corpus totals)


### tokenizer: gpt2 (byte-BPE 50k) (`gpt2`)

| lang | tok/word | tok/grapheme | tok/utf8-byte | tok/sentence |
|---|---|---|---|---|
| eng | 1.228 | 0.206 | 0.205 | 25.8 |
| hin | 7.796 | 2.328 | 0.595 | 192.4 |
| mar | 11.104 | 2.636 | 0.598 | 201.2 |
| ben | 13.253 | 3.182 | 0.745 | 249.3 |
| kan | 22.668 | 4.059 | 0.979 | 350.8 |
| tam | 24.617 | 4.204 | 0.996 | 398.4 |
| tel | 20.481 | 4.562 | 0.991 | 336.7 |

### tokenizer: cl100k_base (byte-BPE 100k) (`cl100k`)

| lang | tok/word | tok/grapheme | tok/utf8-byte | tok/sentence |
|---|---|---|---|---|
| eng | 1.231 | 0.206 | 0.206 | 25.9 |
| hin | 5.035 | 1.503 | 0.384 | 124.3 |
| mar | 7.191 | 1.707 | 0.387 | 130.3 |
| ben | 8.142 | 1.955 | 0.458 | 153.2 |
| kan | 14.848 | 2.658 | 0.641 | 229.8 |
| tam | 12.174 | 2.079 | 0.493 | 197.0 |
| tel | 13.146 | 2.928 | 0.636 | 216.1 |

### tokenizer: XLM-R (SP 250k, 100 lang) (`hf:xlm-roberta-base`)

| lang | tok/word | tok/grapheme | tok/utf8-byte | tok/sentence |
|---|---|---|---|---|
| eng | 1.384 | 0.232 | 0.231 | 29.1 |
| hin | 1.489 | 0.445 | 0.114 | 36.7 |
| mar | 1.956 | 0.464 | 0.105 | 35.4 |
| ben | 2.132 | 0.512 | 0.120 | 40.1 |
| kan | 2.567 | 0.460 | 0.111 | 39.7 |
| tam | 2.423 | 0.414 | 0.098 | 39.2 |
| tel | 2.362 | 0.526 | 0.114 | 38.8 |

### tokenizer: MuRIL (WordPiece, 17 Indic lang) (`hf:google/muril-base-cased`)

| lang | tok/word | tok/grapheme | tok/utf8-byte | tok/sentence |
|---|---|---|---|---|
| eng | 1.258 | 0.211 | 0.210 | 26.4 |
| hin | 1.246 | 0.372 | 0.095 | 30.7 |
| mar | 1.532 | 0.364 | 0.082 | 27.8 |
| ben | 1.404 | 0.337 | 0.079 | 26.4 |
| kan | 1.811 | 0.324 | 0.078 | 28.0 |
| tam | 1.723 | 0.294 | 0.070 | 27.9 |
| tel | 1.955 | 0.435 | 0.095 | 32.1 |

## 3. Cost ratio vs English under each denominator

Each cell: (tokens per unit in lang) ÷ (same denominator, English). v0 headlines only the tok/word column of gpt2.

| lang | gpt2 tok/word | cl100k_base tok/word | XLM-R tok/word | MuRIL tok/word | gpt2 tok/byte | cl100k_base tok/byte | XLM-R tok/byte | MuRIL tok/byte | gpt2 tok/sentence (95% CI) | cl100k_base tok/sentence (95% CI) | XLM-R tok/sentence (95% CI) | MuRIL tok/sentence (95% CI) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hin | 6.35x | 4.09x | 1.08x | 0.99x | 2.89x | 1.86x | 0.49x | 0.45x | 7.45x [7.37, 7.54] | 4.80x [4.75, 4.86] | 1.26x [1.25, 1.28] | 1.16x [1.15, 1.17] |
| mar | 9.04x | 5.84x | 1.41x | 1.22x | 2.91x | 1.88x | 0.45x | 0.39x | 7.79x [7.71, 7.88] | 5.03x [4.98, 5.09] | 1.22x [1.21, 1.23] | 1.05x [1.04, 1.06] |
| ben | 10.79x | 6.61x | 1.54x | 1.12x | 3.63x | 2.22x | 0.52x | 0.38x | 9.66x [9.54, 9.77] | 5.92x [5.85, 5.99] | 1.38x [1.36, 1.39] | 1.00x [0.99, 1.01] |
| kan | 18.45x | 12.06x | 1.85x | 1.44x | 4.76x | 3.11x | 0.48x | 0.37x | 13.59x [13.43, 13.75] | 8.88x [8.78, 8.99] | 1.37x [1.35, 1.38] | 1.06x [1.05, 1.07] |
| tam | 20.04x | 9.89x | 1.75x | 1.37x | 4.85x | 2.39x | 0.42x | 0.33x | 15.43x [15.25, 15.60] | 7.61x [7.53, 7.70] | 1.35x [1.33, 1.36] | 1.05x [1.04, 1.07] |
| tel | 16.67x | 10.68x | 1.71x | 1.55x | 4.82x | 3.09x | 0.49x | 0.45x | 13.04x [12.88, 13.20] | 8.35x [8.24, 8.46] | 1.33x [1.32, 1.35] | 1.22x [1.20, 1.23] |

## 4. The number that should drive a routing-and-cost decision

**Tokens per parallel sentence** (same content, both languages): it is the only denominator that holds
*content* constant; words/graphemes/chars/bytes all scale with script-internal structure that has nothing
to do with what the user asked for. Full reasoning in `analysis.md`.
