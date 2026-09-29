# Performant Text Classification with Naive Bayes for Kaqchikel Maya (MIT 18.337)

This repository is the completed wrap-up of an MIT 18.337 class project on NLP for Kaqchikel Maya.  
The original sentiment-labeling path stalled, so the final system reframes the task as **register classification**:

- `classical`: Kaqchikel Chronicle (`Kiwujil`)
- `modern`: Tang/Bennett written corpus

This keeps the project aligned with the original class goals:

- Julia implementation
- TF-IDF feature extraction
- Naive Bayes classification
- parallel/serial benchmarking

## What Is Implemented

- `src/data_prep.jl`
  - Regenerates `src/datasets/Kiwujil.csv` from source text (clean `Sentence` column only).
  - Loads Tang/Bennett corpus from local path.
  - Builds a balanced manifest (`src/datasets/register_manifest.csv`) with train/test split.
- `src/train_classifier.jl`
  - Trains a hand-rolled multinomial Naive Bayes model on:
    - word unigram TF-IDF
    - character trigram TF-IDF
  - Evaluates accuracy, confusion matrix, precision/recall, top discriminative terms.
  - Writes metrics to `src/results/train_metrics.json`.
- `src/benchmark.jl`
  - Benchmarks serial vs threaded feature extraction and NB training with
    BenchmarkTools, using an explicit warmup so JIT compilation is excluded.
  - Writes metrics to `src/results/benchmark_results.json`.
  - Saves scaling plot to `paper/images/thread_scaling.{svg,png}`.
  - Falls back to a size-matched surrogate corpus built from the committed
    Chronicle text when Tang/Bennett is absent, so scaling numbers are
    reproducible in CI. `corpus_mode` in the JSON records which was used.
- `src/confound_check.jl`
  - Diagnostic: tests whether the char trigram advantage is linguistic or
    orthographic. Profiles apostrophe/punctuation conventions per corpus, fits
    two no-model baselines (an exclusive-character rule and a length threshold),
    then retrains under four normalization regimes and compares each to the word
    unigram model with 95% Wilson intervals and exact McNemar tests.
  - Requires **both** corpora, so it cannot run in CI.
  - Writes `src/results/confound_check.json`.

## Current Results (Seed 42)

- Data prep:
  - `4013` classical Chronicle sentences
  - `43535` modern Tang/Bennett lines after filtering
  - balanced dataset: `8026` samples (`6420` train / `1606` test)
- Test accuracy:
  - word unigram TF-IDF NB: `0.9433`
  - char trigram TF-IDF NB: `0.9894`
  - TextAnalysis.jl `NaiveBayesClassifier` cross-check: `0.9900`
- Orthographic ablation (char trigrams) — **the 0.9894 figure is confounded**:
  - raw: `0.9894`
  - apostrophe variants folded to `U+0027`: `0.9440` (vs word unigram: McNemar p = 1.0)
  - folded + fixed punctuation list stripped: `0.9259` (p = 0.007)
  - folded + restricted to characters shared by both corpora: `0.9265` (p = 0.009)
  - The Chronicle writes the glottal stop as `U+2019` (15,982 times) and
    Tang/Bennett as `U+0027` (269,074 times, with zero other punctuation).
    Since labels come from provenance, encoding leaks the label. Normalizing it
    puts char trigrams level with the `0.9433` word unigram baseline, and
    removing all orthographic cues puts them significantly below it.
  - No-model baseline: predicting `classical` whenever a sentence contains a
    character never seen in modern training text scores `0.9819`.
  - The labels identify the *source*, not register alone; the residual signal
    is largely topic (e.g. `jehová`, `jesús` vs Spanish colonial names). See the
    paper's Limitations section.
- Benchmark (8-thread run on this machine, medians of 30 samples, warmup excluded):
  - char trigram feature extraction: `0.16166s` serial vs `0.09693s` at 8 workers (`1.67x`)
  - NB training: `0.004849s` serial vs `0.005123s` at 8 workers (`0.95x`,
    within the serial run's interquartile range, i.e. no measurable speedup)
  - Only the per-class accumulation in `train_multinomial_nb` is parallelized;
    `sparse(transpose(X))` and the log-likelihood loop remain serial.

## Reproducibility

Install Julia, then run from repo root:

```bash
julia --project=src -e 'using Pkg; Pkg.instantiate()'
julia --project=src src/data_prep.jl
julia --project=src src/train_classifier.jl
julia --project=src src/benchmark.jl
```

For threaded runs:

```bash
JULIA_NUM_THREADS=8 julia --project=src src/train_classifier.jl
JULIA_NUM_THREADS=8 julia --project=src src/benchmark.jl
```

The orthographic diagnostic needs both corpora present locally:

```bash
julia --project=src src/confound_check.jl
```

`src/benchmark.jl` accepts `--quick` to reduce sample counts and `--output-dir DIR`
to write the figure and JSON somewhere other than `paper/images/` and
`src/results/`. CI uses both, writing to `ci_benchmark/`, so the paper always
shows the committed 8-thread run.

## Data and Licensing Notes

- The Tang/Bennett corpus is **not redistributed** here.
- Its authors' `readme.txt` states redistribution restrictions; this repo stores only:
  - source path configuration
  - index manifest
  - aggregate metrics
- The corpus location is resolved in this order, so no machine-specific path
  needs to be committed:
  1. the `--tang-path` command-line flag
  2. the `KAQ_TANG_PATH` environment variable
  3. `src/local_config.toml` (gitignored)
  4. a legacy hardcoded development path, as a last resort

  To set up locally, create `src/local_config.toml`:

  ```toml
  tang_path = "C:/path/to/tang_bennett_2018_corpus_v01_14042023.txt"
  ```

## Julia Ecosystem Contribution

Kaqchikel language support landed in `Languages.jl` through the sequence:

- [PR #43](https://github.com/JuliaText/Languages.jl/pull/43) (this author):
  Kaqchikel language type, initial word lists, and test example
- [PR #46](https://github.com/JuliaText/Languages.jl/pull/46) (Avik Sengupta, maintainer):
  added the Central Kaqchikel trigram profile from
  [wooorm/trigrams](https://github.com/wooorm/trigrams) (UDHR-derived), enabling
  language detection and completing #43

That contribution remains a meaningful output of the project.