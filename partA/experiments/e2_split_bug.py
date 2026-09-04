#!/usr/bin/env python3
"""
E2 -- Claim: `words = line.split(" ")` in fertility.py counts empty strings
when a line contains repeated spaces, overcounting words and understating
fertility.

Isolation: same lines, same tokenizer, same per-line macro averaging as v0 --
only the word splitter changes. Run on the intern's own smoke-test corpus,
where both sample files happen to contain a double-spaced line.

Expect: v0 fertility rises when the splitter is fixed (bias was downward).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import SAMPLE_DIR, get_encode, load_corpus, v0_words, fixed_words


def v0_analyze(lines, encode, words_fn):
    per_line_fertility = []
    for line in lines:
        tokens = encode(line)
        words = words_fn(line)
        per_line_fertility.append(len(tokens) / len(words))
    return sum(per_line_fertility) / len(per_line_fertility)


def main():
    encode = get_encode("gpt2")
    corpus = load_corpus(["eng", "hin"], sample_dir=SAMPLE_DIR)
    print(f"{'lang':<5}{'split(\" \") [v0]':>16}{'split() [fixed]':>17}{'delta':>10}")
    for lang, lines in corpus.items():
        empties = sum(1 for l in lines for w in v0_words(l) if w == "")
        f_v0 = v0_analyze(lines, encode, v0_words)
        f_fix = v0_analyze(lines, encode, fixed_words)
        print(f"{lang:<5}{f_v0:>16.3f}{f_fix:>17.3f}{100 * (f_fix - f_v0) / f_v0:>+9.1f}%"
              f"   (empty-string 'words' counted by v0: {empties}; "
              f"line: {next(l for l in lines if '  ' in l)!r})")


if __name__ == "__main__":
    main()
