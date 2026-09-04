#!/usr/bin/env python3
"""
E5 -- Measured side effect of v0's `line.lower()`.

This is a design choice, not a bug -- but it is not free: byte-BPE vocabularies
are case-sensitive, and lowercasing English *raises* its measured fertility
(merged-case tokens disappear), which quietly deflates the hin/eng fertility
ratio that the report headlines. Hindi (no case) is untouched.

Isolation: same corpus, same splitter, same averaging; toggle lowercase only.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import SAMPLE_DIR, get_encode, load_corpus, v0_words


def fert(lines, encode):
    return sum(len(encode(l)) / len(v0_words(l)) for l in lines) / len(lines)


def main():
    encode = get_encode("gpt2")
    corpus_raw = load_corpus(["eng", "hin"], sample_dir=SAMPLE_DIR, lowercase=False)
    corpus_low = load_corpus(["eng", "hin"], sample_dir=SAMPLE_DIR, lowercase=True)

    r = fert(corpus_raw["hin"], encode) / fert(corpus_raw["eng"], encode)
    l = fert(corpus_low["hin"], encode) / fert(corpus_low["eng"], encode)
    for name, c in [("raw", corpus_raw), ("lower", corpus_low)]:
        fe, fh = fert(c["eng"], encode), fert(c["hin"], encode)
        print(f"{name:>6}: eng fertility {fe:.3f}   hin fertility {fh:.3f}   ratio {fh / fe:.2f}x")
    print(f"\nlowercasing moves eng fertility {100 * (fert(corpus_low['eng'], encode) / fert(corpus_raw['eng'], encode) - 1):+.1f}%"
          f" -> headline hin/eng ratio moves {r:.2f}x -> {l:.2f}x ({100 * (l / r - 1):+.1f}%).")
    print("Hindi is unchanged (no case). Verdict: defensible normalizer, but it silently")
    print("changes the baseline; the report does not mention it. Not a bug -- a disclosed choice.")


if __name__ == "__main__":
    main()
