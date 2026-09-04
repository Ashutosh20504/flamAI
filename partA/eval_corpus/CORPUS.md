# A1 — Eval corpus: construction notes and caveats

## What I built

**FLORES-200 `dev` split**, 997 professionally translated parallel sentences in
**7 languages** (assignment requires ≥4 incl. English, Hindi, 2 Dravidian):

| code | language | script | branch |
|---|---|---|---|
| eng | English | Latin | baseline |
| hin | Hindi | Devanagari | Indo-Aryan |
| mar | Marathi | Devanagari | Indo-Aryan |
| ben | Bengali | Bengali | Indo-Aryan |
| kan | Kannada | Kannada | **Dravidian** |
| tam | Tamil | Tamil | **Dravidian** |
| tel | Telugu | Telugu | **Dravidian** |

- **Source:** official Meta release tarball (`https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz`, ~26 MB). `manifest.json` records per-file sha256 prefixes, sentence counts and preprocessing provenance. Rebuild with `python corpus_prep/build_corpus.py --flores_dir <dir> --out eval_corpus`.
- **Domain:** news/Wikipedia-register sentences (FLORES is drawn from web/wiki sources and professionally translated). Informal/colloquial register is *absent* — see caveats.
- **Size:** 997 sentences/language ≈ 1.4k–3.3k bytes, ~20k–30k characters per Indic language; ~15–25k tokens per language per tokenizer. Big enough for stable per-language means (95% CI on fertility ≈ ±0.02), far too small to compare *subword vocabularies* per word type.

## Preprocessing (each step deliberate)

1. `strip()` + drop empty lines — same as v0.
2. **NFC normalization** — I verified this is *not* a no-op: raw FLORES-200 files are mixed-form. Already-NFC lines per language: eng 997/997, hin 907/997, **ben 411/997**, kan 987/997, tam 995/997, tel 994/997, mar 997/997. NFC merges NFD sequences (typically nukta forms like ज़ = ज + ◌़) into single code points, which changes both byte counts and token counts. Normalizing before comparing languages is what makes `tok/char` comparable at all. (The v0 script did this too — audited in A2 and found **correct**, not a bug.)
3. **No lowercasing, no dedup, no punctuation changes.** The corpus must stay line-aligned so the same content unit (sentence *i*) exists in all 7 languages — this is what enables the per-sentence ratios that drive A3.

## What this corpus **cannot** tell you (read before citing any number from it)

- **Translationese + register.** FLORES sentences are translated, formal, wiki/news text. Real product traffic is casual chat: shorter, messier, code-switched ("अच्छा ok fine"). Byte-level tokenizers fragment code-switching heavily; FLORES will *understate* real-world token inflation for Indic traffic in mixed-language conversations.
- **Sample size at the word-type level.** 997 sentences ≈ 15–25k words/language. Fine for corpus-level means; nowhere near enough to say anything about how a tokenizer handles *rare words*, named entities, or morphological richness per type. Per-sentence means are the only statistic I trust from this data, and I report bootstrap CIs.
- **Input only.** Fertility measures *encoding of prompts*. Serving cost is dominated by *generated* tokens too, and generation length for the same content is a different measurement (the model decides when to stop). Nothing here measures output fertility.
- **Single domain, no numbers/dates/URLs/emoji stress** beyond what wiki sentences happen to contain. Production prompts contain code blocks, JSON, etc., where GPT-2-style byte BPEs behave differently.
- **No claim about *quality*.** Two tokenizers can have identical fertility and very different downstream quality. This analysis is about cost only.
- **English baseline is FLORES English**, i.e. slightly non-native ("translated") English; its fertility is within a hair of the intern's conversational sample, so baseline drift is minimal (measured in `results/e1_reproduce_v0.md`).
