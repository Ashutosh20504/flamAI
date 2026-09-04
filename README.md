# Submission — audit of REPORT_v0

Entry points (Python 3.11+; deps: `tiktoken`, `transformers`, `sentencepiece`,
`protobuf`, `regex`, `numpy` — no torch needed, only tokenizer weights are
downloaded from HF Hub on first run).

```
partA
  corpus_prep/build_corpus.py     # A1: rebuild eval_corpus/ from FLORES-200 raw
                                  #   python build_corpus.py --flores_dir corpus_prep/flores_raw --out ../eval_corpus
  eval_corpus/                    # built corpus (7 langs x 997 parallel lines) + manifest.json + CORPUS.md
  experiments/e2_split_bug.py     # A2 claims, one isolated experiment each:
  experiments/e3_metric_identity.py
  experiments/e4_denominator.py
  experiments/e5_lowercase.py
  experiments/e6_nfc.py
  run_corrected_analysis.py       # A3: 4 tokenizers x 4 denominators + bootstrap CIs
                                  #   -> results/A3_corrected_analysis.md (+ raw JSON)
  audit.md                        # A2 write-up (claims, evidence, direction/magnitude, what's FINE)
  analysis.md                     # A3 write-up + denominator reasoning
  memo.md                         # A4 recommendation memo
partB
  b_calcs.py                      # B1-B4: all arithmetic + log reconciliation
                                  #   -> prints; captured in b_calcs_output.txt
  answers.md                      # written answers
  model_spec.md, bench_log.csv    # copies of starter_kit inputs
partC
  memo.md                         # decision memo
NOTEBOOK.md                       # chronological lab notebook
AI_USAGE.md                       # AI usage disclosure
```

The starter kit's `fertility.py` reproduces REPORT_v0 §1 exactly (eng 1.27 /
0.226, hin 7.45 / 1.579, 5.89×); every number quoted in the write-ups comes
from the scripts above or, for Part B, from `bench_log.csv` via `b_calcs.py`.
