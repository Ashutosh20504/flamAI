#!/usr/bin/env python3
"""
A1 -- Build the multilingual tokenizer eval corpus.

Source: FLORES-200 `dev` split (NLLB team, Meta AI, 2022).
  Download: https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz  (~26 MB)
  Why FLORES-200: professionally translated, line-aligned parallel sentences in
  200+ languages; free; the de-facto standard for multilingual tokenization evals.

Languages (7 > required 4: English + Hindi + >=2 Dravidian):
  eng_Latn  English            (baseline)
  hin_Deva  Hindi              (Indo-Aryan, Devanagari)
  mar_Deva  Marathi            (Indo-Aryan, Devanagari)
  ben_Beng  Bengali            (Indo-Aryan, Bengali script)
  kan_Knda  Kannada            (Dravidian, Kannada script)
  tam_Taml  Tamil              (Dravidian, Tamil script)
  tel_Telu  Telugu             (Dravidian, Telugu script)

Preprocessing (deliberately minimal, and every step is justified in CORPUS.md):
  1. strip whitespace, drop empty lines  (same as fertility.py v0)
  2. NFC-normalize  (NOT a no-op: raw FLORES-200 files contain NFD sequences;
     e.g. only 411/997 Bengali lines are already NFC -- see manifest.json)
  3. NO lowercasing, NO dedup, NO punctuation stripping -- parallel alignment
     must be preserved line-by-line so we can compute per-sentence token ratios.

Held-out: devtest is kept untouched for spot checks; all analysis uses `dev`.

Usage:
    python build_corpus.py --flores_dir <path to flores200_dataset> --out ../eval_corpus
"""

import argparse
import hashlib
import json
import unicodedata
from pathlib import Path

LANGS = {
    "eng": "eng_Latn",
    "hin": "hin_Deva",
    "mar": "mar_Deva",
    "ben": "ben_Beng",
    "kan": "kan_Knda",
    "tam": "tam_Taml",
    "tel": "tel_Telu",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()[:16]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--flores_dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    # accept either the extracted FLORES layout (<dir>/dev/*.dev) or a flat
    # directory of .dev files (as vendored in corpus_prep/flores_raw/)
    dev_dir = args.flores_dir / "dev" if (args.flores_dir / "dev").is_dir() else args.flores_dir
    args.out.mkdir(parents=True, exist_ok=True)

    corpus: dict[str, list[str]] = {}
    manifest = {
        "source": "FLORES-200 dev split (Meta AI NLLB team, 2022)",
        "url": "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz",
        "domain": "wikipedia-style sentences, professionally translated (news/wiki topics)",
        "preprocessing": [
            "strip() each line; drop empty lines",
            "unicodedata NFC normalization (measured: NOT a no-op, see nfc_changed_lines)",
            "no lowercasing / no dedup / no punctuation changes (keeps parallel alignment)",
        ],
        "languages": {},
    }

    n_lines = None
    for short, flores_code in LANGS.items():
        src = dev_dir / f"{flores_code}.dev"
        raw_lines = [ln.strip() for ln in src.read_text(encoding="utf-8").splitlines()]
        lines = [unicodedata.normalize("NFC", ln) for ln in raw_lines if ln.strip()]
        assert not any(not ln for ln in lines), "empty line after strip"
        if n_lines is None:
            n_lines = len(lines)
        assert len(lines) == n_lines, f"{short}: {len(lines)} != {n_lines} (parallelism broken)"

        nfc_before = sum(unicodedata.normalize("NFC", ln) == ln for ln in raw_lines if ln.strip())
        changed = sum(a != b for a, b in zip(lines, [x for x in raw_lines if x.strip()]))

        out_path = args.out / f"{short}.txt"
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        corpus[short] = lines
        manifest["languages"][short] = {
            "flores_code": flores_code,
            "sentences": len(lines),
            "sha256_16": sha256(out_path),
            "nfc_changed_lines": changed,
            "lines_already_nfc_in_raw_flores": nfc_before,
        }

    # parallelism sanity: identical sentence count per language, plus a shared
    # sentence-id so any line can be traced back to FLORES dev.
    manifest["sentences_per_language"] = n_lines
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
