#!/usr/bin/env python3
"""
E6 -- "Suspicious but actually fine" #1: NFC normalization.

read_lines() applies unicodedata.normalize("NFC", ...) to every line. This
LOOKS like the kind of silent preprocessing that could distort a multilingual
comparison. Audited: it is correct and necessary.

Evidence:
  1. On the intern's smoke-test corpus it is a strict no-op (0 lines change,
     0 tokens change) -- so it did NOT distort the v0 numbers.
  2. On real multilingual text it is NOT a no-op: raw FLORES-200 files contain
     NFD sequences (Bengali: only 411/997 lines already NFC). Evaluating
     tokenizers on un-normalized text measures the corpus's normalization
     state, not the tokenizer.
  3. Byte-level effect is ~+0.1..+1.2% bytes for affected languages -- small,
     but systematic and language-asymmetric, exactly the kind of confound you
     must not bake into a cross-language comparison.
"""
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import SAMPLE_DIR, get_encode, load_corpus


def main():
    encode = get_encode("gpt2")

    print("=== 1. intern's smoke-test corpus ===")
    on = load_corpus(["eng", "hin"], sample_dir=SAMPLE_DIR, nfc=True)
    off = load_corpus(["eng", "hin"], sample_dir=SAMPLE_DIR, nfc=False)
    for lang in on:
        t_on = sum(len(encode(l)) for l in on[lang])
        t_off = sum(len(encode(l)) for l in off[lang])
        changed = sum(a != b for a, b in zip(on[lang], off[lang]))
        print(f"{lang}: lines changed by NFC = {changed}, tokens {t_off} -> {t_on} (delta {t_on - t_off})")

    print("\n=== 2. FLORES-200 raw (NFC off vs on), gpt2 ===")
    dev = Path(__file__).resolve().parent.parent / "corpus_prep" / "flores_raw"
    flores = {
        "hin": dev / "hin_Deva.dev",
        "ben": dev / "ben_Beng.dev",
        "kan": dev / "kan_Knda.dev",
    }
    import unicodedata as ud

    for lang, path in flores.items():
        raw = [l.strip() for l in open(path, encoding="utf-8") if l.strip()]
        nfc = [ud.normalize("NFC", l) for l in raw]
        t_raw = sum(len(encode(l)) for l in raw)
        t_nfc = sum(len(encode(l)) for l in nfc)
        b_raw = sum(len(l.encode("utf-8")) for l in raw)
        b_nfc = sum(len(l.encode("utf-8")) for l in nfc)
        already = sum(ud.normalize("NFC", l) == l for l in raw)
        print(f"{lang}: already-NFC {already}/997 | tokens {t_raw} -> {t_nfc} ({100 * (t_nfc - t_raw) / t_raw:+.2f}%)"
              f" | utf8 bytes {b_raw} -> {b_nfc} ({100 * (b_nfc - b_raw) / b_raw:+.2f}%)")


if __name__ == "__main__":
    main()
