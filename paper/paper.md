---
title: "Performant Text Classification with Naive Bayes for Kaqchikel Maya"
date: "July 2026"
author: "William J. Wakefield"
github: https://github.com/Chok-Ketzamtzib/18337-project-kaqchikel-NLP
---

# Abstract

This project began as a 2023 proposal to build a performant Julia NLP pipeline for Kaqchikel Maya using TF-IDF and Naive Bayes. The original task framing was sentiment analysis, but manual sentiment labeling became the primary blocker. The 2026 revision reframes the task to avoid that bottleneck while preserving the original computational goals: distinguish sentences from the Kaqchikel Chronicle (labeled **classical**) from sentences in the Tang/Bennett written corpus (labeled **modern**). Because the labels come from provenance, the task is strictly source identification; register is one of several differences between the two sources. The pipeline is implemented in Julia and includes reproducible data preparation, hand-rolled multinomial Naive Bayes, character trigram and word unigram TF-IDF features, and serial-vs-threaded timing experiments. On a balanced 8,026-sample dataset (6,420 train / 1,606 test), word unigrams reach 94.3% test accuracy and character trigrams reach 98.9%. An ablation shows that the character trigram advantage is **orthographic rather than linguistic**. The two corpora encode the glottal stop differently and differ in punctuation, digits, and quotation marks, so a one-line rule that flags any character never seen in modern training text already reaches 98.2%. Folding apostrophes to one encoding brings trigrams level with word unigrams (94.4%; exact McNemar p = 1.0), and restricting both corpora to their shared character set drops trigrams to 92.7%, significantly below word unigrams (p = 0.009). The project closes with working code, measurable outputs, a documented label-leakage failure mode, and a clearer path to future work on NLP for low-resource languages.

# NLP Pipeline

![NLP pipeline](images/pipeline.pdf)

# Context and Related Work

Kaqchikel NLP is no longer empty space. Recent work includes MayaVoice (Spanish<->14 Mayan MT) [@regalado2025mayavoice], FLORES+ Mayas benchmark/dataset construction [@floresmayas2025], and broad low-resource shared tasks through AmericasNLP [@degibert2025americasnlp]. Monolingual baselines now exist as well, including Goldfish models for `cak_latn` [@chang2026goldfish].  

However, the specific corpus used in this project (Kaqchikel Chronicles) remains useful as a historical register resource and as a compact testbed for reproducible Julia-based methods.

# Corpora and Data Constraints

Two corpora are used:

1. **Kaqchikel Chronicle / Kiwujil text** (labeled `classical`) [@maxwell2006chronicles]
2. **Tang/Bennett written corpus** (labeled `modern`) [@tang2018predictability; @bennett2018stop]

The Tang/Bennett readme explicitly disallows redistribution, so this repository stores only:

- local path configuration,
- sentence index manifests,
- aggregate metrics.

No Tang/Bennett text is committed.

The Chronicle text used here is a 2022 edited transcription written in modern
orthographic conventions, including diaeresis-marked vowels (ä, ë, ï, ö, ü) that
also appear throughout Tang/Bennett. The `classical` label therefore refers to the
text's age and content, not to colonial-era spelling.

# Why the Task Was Reframed

The original sentiment-labeling objective was not completed in 2023. The previous CSV had a placeholder bug (`Sentiment = String` on almost all rows), making supervised sentiment training impossible without major annotation effort.  

For project completion, the label is generated from data provenance:

- `classical` if sentence source is Chronicle,
- `modern` if sentence source is Tang/Bennett.

This preserves the main computational objective (TF-IDF + Naive Bayes + parallelization) while removing manual labeling dependency.

Provenance labels make this a source-identification task. The two sources differ
in register, but also in topic, genre, editorial conventions, and how sentences
were segmented, and a classifier may exploit any of these. The results below
measure how far orthography alone explains the separation; the Limitations
section covers the rest.

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
8. Benchmark serial vs threaded feature extraction and training (`src/benchmark.jl`).
9. Run an orthographic ablation, with no-model baselines and paired significance tests, to test whether the feature advantage is linguistic (`src/confound_check.jl`).

# Parallel Naive Bayes Classifier

Each sentence is assigned the class with the highest log posterior:

$$\hat{c} = \arg\max_{c} \Big[ \log P(c) + \sum_{f} x_f \log \hat{\theta}_{c,f} \Big], \qquad \hat{\theta}_{c,f} = \frac{N_{c,f} + \alpha}{\sum_{f'} N_{c,f'} + \alpha\,|V|}$$

where $x_f$ is the sentence's TF-IDF weight for feature $f$, $N_{c,f}$ is the sum of those weights over training sentences of class $c$, $|V|$ is the vocabulary size, and $\alpha = 1$ (Laplace smoothing). The implementation applies thread-level parallelism (`Threads.@spawn`) to the accumulation of $N_{c,f}$, inspired by prior parallel NB literature [@amazal2018parallelnb].

# Results

## Data Preparation Summary

- Chronicle rows after regeneration: **4013**
- Modern rows after filtering: **43535**
- Balanced dataset: **8026**
- Split: **6420 train / 1606 test**

## Classification Performance

- **Word unigram TF-IDF + multinomial NB**: 0.9433 test accuracy
- **Character trigram TF-IDF + multinomial NB**: 0.9894 test accuracy
- **TextAnalysis.jl `NaiveBayesClassifier` cross-check**: 0.9900 test accuracy

The cross-check is not independent evidence that the 0.9894 figure reflects the
language. `NaiveBayesClassifier` tokenizes with WordTokenizers, which splits
`q’ij, k’a` into `q`, `’`, `ij`, `,`, `k`, `’`, `a`: the curly apostrophe and the
comma become standalone tokens that occur only in `classical` text. It agrees
with the character trigram model because it sees the same leak, described in the
next section. What the agreement does show is that the leak is not specific to
one feature construction or one implementation.

## The Character Trigram Advantage Is Orthographic

The 4.6-point gap between character trigrams and word unigrams invites a
linguistic reading: trigrams capture Kaqchikel morphology that whitespace
tokenization misses. That reading is wrong, and the diagnostic in
`src/confound_check.jl` shows why.

All 20 top discriminative trigrams in `train_metrics.json` contain either U+2019
(right single quotation mark) or a comma, and all 20 favour `classical`. The two
corpora turn out to use nearly disjoint orthographic conventions:

| Corpus | U+2019 | U+0027 | Commas | All punct. |
|---|---|---|---|---|
| Chronicle | 15,982 | 171 | 3,355 | 3,610 |
| Tang/Bennett | 0 | 269,074 | 0 | 0 |

Both corpora mark the glottal stop at almost the same rate (73.1 vs 67.6
apostrophes per 1,000 characters), so the segment itself is not distinctive. Only
its *encoding* is. The Chronicle writes it curly, Tang/Bennett writes it
straight, and Tang/Bennett has been stripped of every other punctuation mark.
Curly-share divergence between the corpora is 0.989 out of a possible 1.0.

Because the labels are derived from provenance, and provenance perfectly predicts
encoding, any character trigram containing U+2019 or a comma is a flawless
`classical` detector. This is label leakage through the digitization pipeline.

The leak extends beyond the apostrophe. In the training split, 22 characters
occur in Chronicle sentences and never in Tang/Bennett sentences: the curly
apostrophe, curly double quotes, commas and other punctuation, all ten digits, the
underscore, and the en dash. No character occurs only in Tang/Bennett. A rule
with no model at all — predict `classical` if and only if a sentence contains one
of those 22 characters — flags 96.4% of Chronicle test sentences and reaches
**0.9819** test accuracy, within a point of the trained trigram model. For
comparison, the best single sentence-length threshold (13 tokens or more
predicts `modern`) reaches only 0.6220, so length is a weak cue.

Retraining the character trigram model under four normalization regimes isolates
the effect. Regime D keeps only characters that appear in the training text of
*both* classes, which removes every Chronicle-only character, including those
that Regime C's fixed punctuation list misses. Intervals are 95% Wilson intervals
on the 1,606 test sentences; *p* is an exact McNemar test of paired predictions
against the word unigram model.

| Regime | Accuracy (95% CI) | vs. word unigram | Top-50 with non-letters |
|---|---|---|---|
| A. Raw (as originally reported) | 0.9894 (0.983–0.993) | +4.6 pts, $p < 10^{-15}$ | 49/50 |
| B. Apostrophes folded to U+0027 | 0.9440 (0.932–0.954) | +0.1 pts, *p* = 1.0 | 42/50 |
| C. Folded, fixed punctuation list stripped | 0.9259 (0.912–0.938) | -1.7 pts, *p* = 0.007 | 10/50 |
| D. Folded, shared character set only | 0.9265 (0.913–0.938) | -1.7 pts, *p* = 0.009 | 0/50 |
| *Word unigram baseline* | *0.9433 (0.931–0.954)* | — | — |
| *Exclusive-character rule (no model)* | *0.9819 (0.974–0.987)* | — | — |

The last column counts top-50 trigrams containing any character other than a
letter, a space, or U+0027. Straight U+0027 is treated as a letter here because it
is how both corpora, once folded, write the glottal stop.

Regime B preserves the glottal stop as a linguistic segment and changes only its
encoding. That alone erases the entire advantage: 0.9440 against 0.9433, with the
two models disagreeing on roughly 100 sentences and splitting them 50 to 49. The
commas, digits and quotation marks that still appear in 42 of Regime B's top 50
trigrams add nothing once the apostrophe signal is gone. Regimes C and D remove
the remaining non-letter cues, and character trigrams then fall *below* word
unigrams by 1.7 points, a paired difference unlikely to be chance (*p* < 0.01).
Once orthographic cues are removed, character trigrams are worse than word
unigrams on this task.

What remains is not obviously register either. With no orthographic cues left,
the strongest Regime D trigrams favouring `modern` spell Biblical names and
Spanish function words (`jehová`, `jesús`, ` y `, ` o `, ` más`); the strongest
favouring `classical` come from Spanish personal names and colonial titles
(`lópez`, `díaz`, other `-ez` surnames, *gobernador*, *padre*). The word unigram model's top features
tell the same story (`jehová`, `jesús`, `más` versus `kastilan`, `don`). Much of
the residual separation is therefore topic and genre: a Christian-text-heavy
modern corpus against a colonial-era historical narrative.

The defensible claim is therefore narrower than the headline number. Sentences
from these two sources can be told apart at roughly 93–94% once orthographic
encoding is controlled, and the apparent 98.9% is largely a measurement of which
file a sentence was digitized into. Whether any of the remaining signal is
register, rather than topic, genre, or editorial convention, this experimental
design cannot say (see Limitations).

## Benchmark Summary (8-thread run)

Timings are medians of 30 BenchmarkTools samples on the 6,420-row training split
(feature matrix 6,420 × 7,442, 387,213 nonzeros), with an explicit warmup call
per configuration so that compilation is never inside the sample set.

| Workers | Features (s) | Speedup | NB train (s) | Speedup |
|---|---|---|---|---|
| 1 | 0.16166 | 1.00x | 0.004849 | 1.00x |
| 2 | 0.13305 | 1.22x | 0.004861 | 1.00x |
| 4 | 0.11139 | 1.45x | 0.004875 | 0.99x |
| 8 | 0.09693 | 1.67x | 0.005123 | 0.95x |

![Thread scaling](images/thread_scaling.png)

An earlier version of this benchmark used `@elapsed` with no warmup and reported
Naive Bayes training falling from 0.0251s serial to 0.0078s at 8 workers, a
~3.2x speedup. That result does not survive proper measurement. `@elapsed` timed
the *first* call through each code path, so the serial number absorbed
compilation of the serial path and the threaded numbers absorbed compilation of
the `@spawn` path. The same flaw produced an apparent 0.26x "slowdown" at 2
workers, which was JIT latency rather than scheduling overhead.

With warmup excluded, **Naive Bayes training shows no measurable speedup from
thread-level parallelism.** The 8-thread median (5.12 ms) lies inside the
interquartile range of the serial run (4.54–5.28 ms), so the data support neither
a speedup nor a slowdown. The code structure is consistent with an Amdahl's-law
limit, although the serial fraction was not profiled directly. Only one region of
`train_multinomial_nb` is parallelized — the per-class accumulation of
`feature_sums` over the 387,213 nonzeros — while two serial regions remain: the
`sparse(transpose(X))` materialization and the dense `n_classes × n_features`
log-likelihood loop (2 × 7,442 logarithms). At about 5 ms per call, task-spawn
and reduction overhead are also a non-trivial share of the total. Allocations
rise from 37 (6.5 MB) at one thread to 114 (7.5 MB) at eight.

Feature extraction does scale, because chunked trigram counting is independent
per document, but sublinearly: 1.67x on 8 threads, with non-overlapping
interquartile ranges against the serial run. Two plausible limits were not
separated by profiling. The per-chunk `Dict{String,Int}` merge is serial and grows
with thread count, and each call makes about 1.6 million allocations (about 150 MB),
so garbage collection may also cap the speedup. Hashing trigrams to integer IDs
and accumulating into preallocated arrays would reduce both, and is the obvious
next optimization.

# Contributions to Julia Text Ecosystem

The project added Kaqchikel (ISO 639-3 `cak`) to `Languages.jl`. In
[PR #43](https://github.com/JuliaText/Languages.jl/pull/43) (merged May 2023), the
author registered Kaqchikel as a language type and contributed its initial word
lists (stopwords, pronouns, prepositions, articles) along with a Kaqchikel example
sentence for the test suite. Merging it revealed that language detection had no
trigram profile for Kaqchikel, so the new example was misclassified as Ilocano.
The maintainer, Avik Sengupta, resolved this in
[PR #46](https://github.com/JuliaText/Languages.jl/pull/46) by adding the Central
Kaqchikel trigram profile from the `wooorm/trigrams` dataset, which derives its
profiles from translations of the Universal Declaration of Human Rights. With both
changes merged, Kaqchikel is supported by the package's word-list and
language-detection functions.

# Discussion

This project demonstrates an important lesson for low-resource NLP: **labeling is not always the right first bottleneck to solve**.  

When labels are expensive, useful alternatives include:

- weak labels from source metadata (used here),
- self-supervised objectives,
- multilingual transfer from related languages.

The ablation adds a second lesson, and a sharper one: **weak labels derived from
provenance leak provenance.** Because every `classical` example came from one
file and every `modern` example from another, any incidental difference between
those files — encoding, punctuation policy, transcription era, editorial
convention — becomes a free feature. Character n-grams are especially exposed,
since they see exactly the surface detail that tokenization would discard. The
word unigram path scored lower here because TextAnalysis `strip_punctuation`
happened to delete most of the leaking characters: apostrophes, commas, quotation
marks, and underscores. The "weaker" feature set was therefore the less exposed
one, though not a leak-free one. Digits and en dashes survive
`strip_punctuation`, and the same step deletes the glottal stop, which is a
phoneme in Kaqchikel, merging words that differ only by a glottal stop.

The practical safeguard is cheap. Before interpreting a margin between feature
sets, check what a no-model baseline achieves — here, a single set lookup
reached 98.2% — then normalize the orthography and re-measure. A gap that
vanishes under encoding normalization was never linguistic. For this corpus pair,
that check costs one function and a few seconds of compute, and it changes the
conclusion.

A third lesson concerns performance measurement. The parallel speedup originally
reported here was an artifact of timing first calls in a JIT-compiled language.
Warmup is not a refinement in Julia benchmarking; without it, the measurement can
invert the sign of the result.

# Limitations

The results above should be read with the following constraints in mind. None
changes the orthographic finding, but several bound how far the remaining ~93%
figure can be interpreted.

- **Source is not register.** Every `classical` sentence comes from one colonial-era
  narrative and every `modern` sentence from one multi-source corpus. The classes
  also differ in topic, genre, and editorial conventions, and the top residual
  features suggest topic carries much of the remaining signal.
- **Segmentation differs by source.** Chronicle sentences are split on `.`, `!`,
  `?` and line breaks; Tang/Bennett sentences are its lines as distributed. Mean
  length is 9.2 tokens against 16.0, although length alone predicts only 62% of
  test labels.
- **Evaluation design.** The TF-IDF vocabulary and IDF weights are computed over
  all 8,026 sentences, including the test split. This uses no labels but is
  transductive. 40 of the 1,606 test sentences (2.5%) also appear verbatim in the
  training split. The split is random at sentence level, so neighbouring sentences
  of the same narrative, which share names and events, fall on both sides of it.
  All figures come from one seed and one split.
- **Model choice.** TF-IDF weights are fed to multinomial Naive Bayes as
  fractional counts, a common heuristic rather than the model's generative
  assumption.
- **Reproducibility.** The Tang/Bennett corpus cannot be redistributed, so the
  accuracy and ablation figures can be reproduced only by readers who obtain it
  from its authors.

# Future Work

- Replace provenance labels with a design that separates register from source:
  several documents per period, matched for genre where possible.
- Deduplicate before splitting, fit features on the training split only, hold out
  contiguous blocks of each narrative, and report results across several seeds.
- Evaluate `Languages.jl` language detection on both corpora, since its trigram
  profile comes from a single modern document.
- Port the pipeline to Python. Python/Hugging Face tooling is better suited to
  modern language-model workflows, including the Goldfish `cak_latn` models, while
  Julia remains an effective environment for transparent, reproducible algorithmic
  baselines and performance experiments.

# References
