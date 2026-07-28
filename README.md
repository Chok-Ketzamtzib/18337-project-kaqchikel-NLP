# Wrap-Up: Kaqchikel Naive Bayes (MIT 18.337)

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
  - Benchmarks serial vs threaded feature extraction and NB training.
  - Writes metrics to `src/results/benchmark_results.json`.
  - Saves scaling plot to `paper/images/thread_scaling.png`.

## Current Results (Seed 42)

- Data prep:
  - `4013` classical Chronicle sentences
  - `43535` modern Tang/Bennett lines after filtering
  - balanced dataset: `8026` samples (`6420` train / `1606` test)
- Test accuracy:
  - word unigram TF-IDF NB: `0.9433`
  - char trigram TF-IDF NB: `0.9894`
- Benchmark (8-thread run on this machine):
  - char trigram feature extraction: `0.2806s` serial vs `0.1946s` at 8 workers
  - NB training: `0.0251s` serial vs `0.0078s` at 8 workers (best observed)

## Reproducibility

Install Julia, then run from repo root:

```bash
julia --project=src src/data_prep.jl
julia --project=src src/train_classifier.jl
julia --project=src src/benchmark.jl
```

For threaded runs:

```bash
JULIA_NUM_THREADS=8 julia --project=src src/train_classifier.jl
JULIA_NUM_THREADS=8 julia --project=src src/benchmark.jl
```

## Data and Licensing Notes

- The Tang/Bennett corpus is **not redistributed** here.
- Its authors' `readme.txt` states redistribution restrictions; this repo stores only:
  - source path configuration
  - index manifest
  - aggregate metrics
- The default local corpus path is encoded in `src/kaq_pipeline.jl` and can be overridden via `--tang-path`.

## Julia Ecosystem Contribution

Kaqchikel language support landed in `Languages.jl` through the sequence:

- PR #43: initial Kaqchikel data contribution
- PR #46: trigram fix for language detection

That contribution remains a meaningful output of the project.