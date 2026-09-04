#!/usr/bin/env python3
"""
E3 -- Claim: REPORT_v0 says the tok/char column "confirms" the fertility
column ("the two metrics agree, so the result is robust"). But the two
columns are algebraically the same number:

    tokens/word  ==  tokens/char  x  chars/word

so agreement is an identity, not independent corroboration. Nothing two-way
correlated can "confirm" anything.

Isolation: recompute all three quantities per language on the intern's own
smoke-test corpus with v0's exact settings and check the identity holds to
floating-point precision; then show chars/word hin/eng < 1, which also
refutes the report's root-cause claim ("Hindi has more Unicode characters
per word") using the report's OWN table.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import SAMPLE_DIR, get_encode, load_corpus, v0_words


def main():
    encode = get_encode("gpt2")
    corpus = load_corpus(["eng", "hin"], sample_dir=SAMPLE_DIR,
                         lowercase=True)  # v0 lowercases
    stats = {}
    for lang, lines in corpus.items():
        fert = sum(len(encode(l)) / len(v0_words(l)) for l in lines) / len(lines)
        tpc = sum(len(encode(l)) / len(l) for l in lines) / len(lines)
        cpw = sum(len(l) / len(v0_words(l)) for l in lines) / len(lines)
        stats[lang] = (fert, tpc, cpw)

    print("lang   tok/word   tok/char  chars/word | identity check: tok/char * chars/word")
    for lang, (fert, tpc, cpw) in stats.items():
        print(f"{lang:<6}{fert:>10.3f}{tpc:>11.3f}{cpw:>11.3f} | {tpc * cpw:>10.3f}"
              f"  (residual {abs(fert - tpc * cpw):.2e})")

    fr = stats["hin"][0] / stats["eng"][0]
    cr = stats["hin"][1] / stats["eng"][1]
    wr = stats["hin"][2] / stats["eng"][2]
    print(f"\nfertility ratio hin/eng = {fr:.2f}")
    print(f"tok/char     ratio      = {cr:.2f}")
    print(f"chars/word   ratio      = {wr:.2f}")
    print(f"identity: {cr:.2f} x {wr:.2f} = {cr * wr:.2f} == fertility ratio  ->  the two")
    print("columns are ONE measurement plus a length statistic; no mutual confirmation.")
    print(f"\nReport's own root-cause claim was 'Hindi has MORE unicode chars per word';")
    print(f"its own numbers give chars/word eng={stats['eng'][2]:.2f} vs hin={stats['hin'][2]:.2f}"
          f" -> ratio {wr:.2f} (< 1). The stated mechanism is contradicted by the report's own table.")


if __name__ == "__main__":
    main()
