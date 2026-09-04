"""Shared helpers for the Part A audit scripts."""

import sys
import unicodedata
from pathlib import Path

PART_A = Path(__file__).resolve().parent
EVAL_CORPUS = PART_A / "eval_corpus"
# the intern's smoke-test corpora (copy of starter_kit/corpus_sample)
SAMPLE_DIR = PART_A / "corpus_sample"

LANGS = ["eng", "hin", "mar", "ben", "kan", "tam", "tel"]

_tokenizer_cache = {}


def read_lines(path, nfc=True, lowercase=False):
    """Load lines the way fertility.py v0 does (strip, drop empties), with
    switches so experiments can isolate the NFC / lowercase steps."""
    out = []
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if nfc:
                line = unicodedata.normalize("NFC", line)
            if lowercase:
                line = line.lower()
            out.append(line)
    return out


def load_corpus(langs=LANGS, sample_dir=None, nfc=True, lowercase=False):
    """Corpus of line lists. sample_dir=<starter_kit corpus_sample> loads the
    intern's 10-line smoke-test corpora instead of FLORES."""
    corpus = {}
    for lang in langs:
        if sample_dir:
            name = {"eng": "eng_sample.txt", "hin": "hin_sample.txt"}[lang]
            corpus[lang] = read_lines(Path(sample_dir) / name, nfc=nfc, lowercase=lowercase)
        else:
            corpus[lang] = read_lines(EVAL_CORPUS / f"{lang}.txt", nfc=nfc, lowercase=lowercase)
    return corpus


def get_encode(spec):
    """Return encode(str)->list[int] for 'gpt2', 'cl100k' (tiktoken) or
    'hf:<repo_id>' (HuggingFace, no torch needed for tokenizers)."""
    if spec in _tokenizer_cache:
        return _tokenizer_cache[spec]
    if spec.startswith("hf:"):
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(spec[3:])
        encode = lambda s: tok.encode(s, add_special_tokens=False)
    else:
        import tiktoken

        name = "cl100k_base" if spec == "cl100k" else spec
        encode = tiktoken.get_encoding(name).encode
    _tokenizer_cache[spec] = encode
    return encode


# the two functions in fertility.py's analyze(), kept byte-for-byte faithful so
# experiments can toggle one factor at a time
def v0_words(line):
    return line.split(" ")          # v0: counts empty strings on double spaces


def fixed_words(line):
    return line.split()
