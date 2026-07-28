---
title: "Performant Text Classification with Naive Bayes for Kaqchikel Maya: Project Wrap-Up"
date: "July 2026"
author: "William J. Wakefield"
github: https://github.com/Chok-Ketzamtzib/18337-project-kaqchikel-NLP
---

# Abstract

This project began as a 2023 proposal to build a performant Julia NLP pipeline for Kaqchikel Maya using TF-IDF and Naive Bayes. The original task framing was sentiment analysis, but manual sentiment labeling became the primary blocker. The wrap-up in 2026 reframes the task to avoid that bottleneck while preserving the original computational goals: classify **classical** Kaqchikel Chronicle text versus **modern** written Kaqchikel from the Tang/Bennett corpus. The final pipeline is implemented in Julia, includes reproducible data preparation, hand-rolled multinomial Naive Bayes, character trigram and word unigram TF-IDF features, and serial-vs-threaded timing experiments. On a balanced 8,026-sample dataset (6,420 train / 1,606 test), unigram TF-IDF reaches 94.3% test accuracy and character trigrams reach 98.9%. This closes the class project with working code, measurable outputs, and a clearer path to future work in modern low-resource NLP.

# NLP Pipeline

![NLP Pipeline](https://raw.githubusercontent.com/Chok-Ketzamtzib/18337-project-kaqchikel-NLP/main/paper/images/pipeline.png)

# Context and Related Work

Kaqchikel NLP is no longer empty space. Recent work includes MayaVoice (Spanish<->14 Mayan MT) [@regalado2025mayavoice], FLORES+ Mayas benchmark/dataset construction [@floresmayas2025], and broad low-resource shared tasks through AmericasNLP [@degibert2025americasnlp]. Monolingual baselines now exist as well, including Goldfish models for `cak_latn` [@chang2026goldfish].  

However, the specific corpus used in this project (Kaqchikel Chronicles) remains useful as a historical register resource and as a compact testbed for reproducible Julia-based methods.

# Corpora and Data Constraints

Two corpora are used:

1. **Kaqchikel Chronicle / Kiwujil text** (classical register) [@maxwell2006chronicles]
2. **Tang/Bennett written corpus** (modern register) [@tang2018predictability; @bennett2018stop]

The Tang/Bennett readme explicitly disallows redistribution, so this repository stores only:

- local path configuration,
- sentence index manifests,
- aggregate metrics.

No Tang/Bennett text is committed.

# Why the Task Was Reframed

The original sentiment-labeling objective was not completed in 2023. The previous CSV had a placeholder bug (`Sentiment = String` on almost all rows), making supervised sentiment training impossible without major annotation effort.  

For project completion, the label is generated from data provenance:

- `classical` if sentence source is Chronicle,
- `modern` if sentence source is Tang/Bennett.

This preserves the main computational objective (TF-IDF + Naive Bayes + parallelization) while removing manual labeling dependency.

# Procedure

1. Regenerate Chronicle CSV from source text (`Sentence` only).
2. Load Tang/Bennett corpus locally and filter lines by token count.
3. Build a balanced sample between classes.
4. Create a stratified train/test split (seed = 42).
5. Build feature matrices:
   - word unigram TF-IDF (TextAnalysis.jl),
   - character trigram TF-IDF (custom sparse construction).
6. Train hand-rolled multinomial Naive Bayes (Laplace smoothing).
7. Evaluate on held-out test set.
8. Benchmark serial vs threaded feature extraction and training.

# Parallel Naive Bayes Classifier

$$P(c|x) = P(x|c) * P(c) / P(x)$$ 

The implementation follows the standard multinomial Naive Bayes formulation with log priors and log likelihoods, and applies worker-level parallelism during aggregation of per-class feature statistics, inspired by prior parallel NB literature [@amazal2018parallelnb].

# Results

## Data Preparation Summary

- Chronicle rows after regeneration: **4013**
- Modern rows after filtering: **43535**
- Balanced dataset: **8026**
- Split: **6420 train / 1606 test**

## Classification Performance

- **Word unigram TF-IDF + multinomial NB**: 0.9433 test accuracy
- **Character trigram TF-IDF + multinomial NB**: 0.9894 test accuracy

The class boundary is strong, which is expected because historical chronicle style differs heavily from modern written register.

## Benchmark Summary (8-thread run)

- Character trigram feature extraction:
  - serial: 0.2806s
  - 8 workers: 0.1946s
- Naive Bayes training:
  - serial: 0.0251s
  - best observed: 0.0078s (8 workers)

Measured speedups are not monotonic at every worker count; this is common at small workloads where scheduling overhead can dominate.

![Thread scaling](images/thread_scaling.png)

# Contributions to Julia Text Ecosystem

The project also contributed Kaqchikel support to `Languages.jl` through PR #43 (initial addition) and PR #46 (trigram fix), making language detection support practical for downstream Julia NLP workflows.

# Discussion

This project demonstrates an important lesson for low-resource NLP: **labeling is not always the right first bottleneck to solve**.  

When labels are expensive, useful alternatives include:

- weak labels from source metadata (used here),
- self-supervised objectives,
- multilingual transfer from related languages.

For future work, Python/Hugging Face tooling is likely better for modern LLM workflows, while Julia remains an effective environment for transparent, reproducible algorithmic baselines and performance experiments.

# References

Citations are maintained in `paper/paper.bib`.
