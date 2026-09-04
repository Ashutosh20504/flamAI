#!/usr/bin/env python3
"""
E4 -- The conceptual flaw: fertility (tokens per whitespace word) computes
exactly what it says, but a cost decision needs tokens per unit of *content*,
and whitespace words do not hold content constant across languages (Hindi
sentences use different numbers of words than their English translations).

Isolation: hold the CONTENT fixed -- use parallel line i in each language --
and compare tokens(sent i, hin) / tokens(sent i, eng) against the per-word
fertility ratio on the same data.

Run 1: the intern's own 10-line parallel smoke-test corpus (gpt2, v0 settings).
Run 2: FLORES-200 (997 parallel sentences), all languages.

Expect: the fertility ratio and the per-sentence (same-content) token ratio
disagree in BOTH directions depending on words-per-sentence structure.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import EVAL_CORPUS, SAMPLE_DIR, get_encode, load_corpus


def sentence_token_lists(lines, encode):
    return [len(encode(l)) for l in lines]


def main():
    encode = get_encode("gpt2")

    print("=== Run 1: intern's 10-line parallel sample (gpt2, v0 settings: lower+NFC) ===")
    corpus = load_corpus(["eng", "hin"], sample_dir=SAMPLE_DIR, lowercase=True)
    eng_t = sentence_token_lists(corpus["eng"], encode)
    hin_t = sentence_token_lists(corpus["hin"], encode)
    ratios = np.array(hin_t) / np.array(eng_t)
    lo, hi = bootstrap_ci(np.array(hin_t, dtype=float), np.array(eng_t, dtype=float))
    f_eng = np.mean([t / len(l.split(" ")) for t, l in zip(eng_t, corpus["eng"])])
    f_hin = np.mean([t / len(l.split(" ")) for t, l in zip(hin_t, corpus["hin"])])
    print(f"per-word fertility ratio (v0's number):        {f_hin / f_eng:.2f}x")
    print(f"tokens per parallel sentence (same content):   {ratios.mean():.2f}x "
          f"[95% bootstrap CI {lo:.2f}-{hi:.2f}]")
    print(f"-> v0 overstates the per-request cost gap by "
          f"{100 * ((f_hin / f_eng) / ratios.mean() - 1):.0f}% on its own data")
    print(f"   (words/sentence: eng {np.mean([len(l.split()) for l in corpus['eng']]):.1f} "
          f"vs hin {np.mean([len(l.split()) for l in corpus['hin']]):.1f} for the same content)")

    print("\n=== Run 2: FLORES-200, 997 parallel sentences, gpt2 ===")
    langs = ["eng", "hin", "mar", "ben", "kan", "tam", "tel"]
    corpus = load_corpus(langs)
    base = np.array(sentence_token_lists(corpus["eng"], encode), dtype=float)
    print(f"{'lang':<5}{'fertility ratio':>16}{'parallel-sentence ratio':>25}  {'direction of per-word bias':>28}")
    for lang in langs[1:]:
        t = np.array(sentence_token_lists(corpus[lang], encode), dtype=float)
        w_eng = np.array([len(l.split()) for l in corpus["eng"]], dtype=float)
        w_l = np.array([len(l.split()) for l in corpus[lang]], dtype=float)
        fert_ratio = (t / w_l).mean() / (base / w_eng).mean()
        ps = (t / base)
        lo, hi = bootstrap_ci(t, base)
        bias = fert_ratio / ps.mean() - 1
        print(f"{lang:<5}{fert_ratio:>15.2f}x{ps.mean():>21.2f}x [{lo:.2f},{hi:.2f}]"
          f"{'understates cost by' if bias < 0 else 'overstates cost by':>22} {abs(100 * bias):>4.0f}%")


def bootstrap_ci(a, b, n=10000, seed=1337):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n, len(a)))
    stats = (a[idx] / b[idx]).mean(axis=1)
    return np.percentile(stats, [2.5, 97.5])


if __name__ == "__main__":
    main()
