#!/usr/bin/env python3
"""
A3 -- Corrected cross-language tokenizer analysis on the FLORES-200 eval corpus.

Tokenizers (4 > required 2; two are multilingual/Indic-aware):
  gpt2     tiktoken, byte-BPE 50k vocab      (the v0 report's tokenizer)
  cl100k   tiktoken, byte-BPE 100k vocab     (GPT-3.5/4-era; better multilingual BPE)
  xlm-r    hf:xlm-roberta-base, 250k SentencePiece, 100-language
  muril    hf:google/muril-base-cased, WordPiece trained on 17 Indian languages

Denominators (4 > required 2), per language, corpus-level:
  tok/word       whitespace words            (v0's denominator)
  tok/grapheme   user-perceived characters   (regex \\X)
  tok/utf8-byte  encoded size
  tok/sentence   PARALLEL sentences          (same content in every language --
                                             the only denominator that holds
                                             content constant)

Also computes: paired per-sentence token ratios vs English with 95% bootstrap
CIs (10k resamples, seed 1337), and the words/graphemes/chars/bytes/sentences
structure table that explains every ratio.

Output: results/A3_corrected_analysis.md (this script is deterministic).
"""
import json
import sys
from pathlib import Path

import numpy as np
import regex as re

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import EVAL_CORPUS, LANGS, get_encode, load_corpus

OUT = Path(__file__).resolve().parent / "results" / "A3_corrected_analysis.md"
TOKENIZERS = ["gpt2", "cl100k", "hf:xlm-roberta-base", "hf:google/muril-base-cased"]
PRETTY = {"gpt2": "gpt2 (byte-BPE 50k)", "cl100k": "cl100k_base (byte-BPE 100k)",
          "hf:xlm-roberta-base": "XLM-R (SP 250k, 100 lang)",
          "hf:google/muril-base-cased": "MuRIL (WordPiece, 17 Indic lang)"}


def bootstrap_ci_ratio(a, b, n=10000, seed=1337):
    """95% CI for mean(a_i)/mean(b_i) over paired sentences (ratio of means,
    i.e. total-tokens cost ratio), bootstrap-resampling sentence pairs."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n, len(a)))
    stats = a[idx].mean(axis=1) / b[idx].mean(axis=1)
    return np.percentile(stats, [2.5, 97.5])


def main():
    corpus = load_corpus(LANGS)
    encode = {t: get_encode(t) for t in TOKENIZERS}

    # ---- structure table ------------------------------------------------
    struct = {}
    for lang in LANGS:
        sents = corpus[lang]
        struct[lang] = {
            "sentences": len(sents),
            "words": sum(len(s.split()) for s in sents),
            "graphemes": sum(len(re.findall(r"\X", s)) for s in sents),
            "chars": sum(len(s) for s in sents),
            "utf8_bytes": sum(len(s.encode("utf-8")) for s in sents),
        }

    # ---- tokens per denominator -----------------------------------------
    tokens = {t: {l: sum(len(enc(s)) for s in corpus[l]) for l in LANGS} for t, enc in encode.items()}

    # ---- paired per-sentence ratios vs eng -------------------------------
    eng_tok = {t: np.array([len(enc(s)) for s in corpus["eng"]], dtype=float)
               for t, enc in encode.items()}
    paired = {}
    for t in TOKENIZERS:
        paired[t] = {}
        for lang in LANGS[1:]:
            v = np.array([len(encode[t](s)) for s in corpus[lang]], dtype=float)
            lo, hi = bootstrap_ci_ratio(v, eng_tok[t])
            paired[t][lang] = {"ratio": v.mean() / eng_tok[t].mean(), "lo": lo, "hi": hi}

    # ---- emit markdown ----------------------------------------------------
    lines = []
    w = lines.append
    w("# A3 — Corrected cross-language tokenizer analysis (FLORES-200 dev, 997 parallel sentences × 7 languages)\n")
    w("Reproduce: `python run_corrected_analysis.py` (deterministic; seed 1337).\n")

    w("## 1. Corpus structure (the confound v0 ignored)\n")
    w("| lang | sentences | words | graphemes | unicode chars | utf-8 bytes | chars/word | graph/word |")
    w("|---|---|---|---|---|---|---|---|")
    for l in LANGS:
        s = struct[l]
        w(f"| {l} | {s['sentences']} | {s['words']:,} | {s['graphemes']:,} | {s['chars']:,} | "
          f"{s['utf8_bytes']:,} | {s['chars'] / s['words']:.2f} | {s['graphemes'] / s['words']:.2f} |")
    w("\nReading: whitespace words are *longer* in the Dravidian languages (8.5–9.1 chars/word vs 5.97 English) and *shorter* in Hindi (5.10). A denominator that scales with word length cannot compare tokenization quality across languages.\n")

    w("## 2. Tokens per unit, all languages (corpus totals)\n")
    for t in TOKENIZERS:
        w(f"\n### tokenizer: {PRETTY[t]} (`{t}`)\n")
        w("| lang | tok/word | tok/grapheme | tok/utf8-byte | tok/sentence |")
        w("|---|---|---|---|---|")
        for l in LANGS:
            n = tokens[t][l]
            w(f"| {l} | {n / struct[l]['words']:.3f} | {n / struct[l]['graphemes']:.3f} | "
              f"{n / struct[l]['utf8_bytes']:.3f} | {n / struct[l]['sentences']:.1f} |")

    w("\n## 3. Cost ratio vs English under each denominator\n")
    w("Each cell: (tokens per unit in lang) ÷ (same denominator, English). v0 headlines only the tok/word column of gpt2.\n")
    header = "| lang | " + " | ".join(f"{PRETTY[t].split(' ')[0]} tok/word" for t in TOKENIZERS) + \
             " | " + " | ".join(f"{PRETTY[t].split(' ')[0]} tok/byte" for t in TOKENIZERS) + \
             " | " + " | ".join(f"{PRETTY[t].split(' ')[0]} tok/sentence (95% CI)" for t in TOKENIZERS) + " |"
    w(header)
    w("|" + "---|" * (1 + 3 * len(TOKENIZERS)))
    for l in LANGS[1:]:
        cells = []
        for t in TOKENIZERS:
            cells.append(f"{(tokens[t][l] / struct[l]['words']) / (tokens[t]['eng'] / struct['eng']['words']):.2f}x")
        for t in TOKENIZERS:
            cells.append(f"{(tokens[t][l] / struct[l]['utf8_bytes']) / (tokens[t]['eng'] / struct['eng']['utf8_bytes']):.2f}x")
        for t in TOKENIZERS:
            p = paired[t][l]
            cells.append(f"{p['ratio']:.2f}x [{p['lo']:.2f}, {p['hi']:.2f}]")
        w(f"| {l} | " + " | ".join(cells) + " |")

    w("\n## 4. The number that should drive a routing-and-cost decision\n")
    w("**Tokens per parallel sentence** (same content, both languages): it is the only denominator that holds")
    w("*content* constant; words/graphemes/chars/bytes all scale with script-internal structure that has nothing")
    w("to do with what the user asked for. Full reasoning in `analysis.md`.\n")

    OUT.write_text("\n".join(lines), encoding="utf-8")
    (OUT.parent / "A3_raw_counts.json").write_text(json.dumps(
        {"structure": struct, "tokens": tokens, "paired_ratio_vs_eng": paired}, indent=2))
    print(f"wrote {OUT}")
    print("\nHeadline — tokens per parallel sentence vs English (95% bootstrap CI):")
    for t in TOKENIZERS:
        row = ", ".join(f"{l} {paired[t][l]['ratio']:.2f}x" for l in LANGS[1:])
        print(f"  {PRETTY[t]:<34} {row}")


if __name__ == "__main__":
    main()
