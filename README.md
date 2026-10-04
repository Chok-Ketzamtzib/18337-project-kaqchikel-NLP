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

## Animations

`animations/` holds [Manim Community](https://www.manim.community/) scenes for
short explainer videos. Every scene `Foo` also has a `FooVertical` version at
3:4 portrait (1080x1440 at `-qh`, for Instagram):

- `scenes/sparse_transpose.py`: why `train_multinomial_nb` builds
  `sparse(transpose(X))`. CSC storage scatters each sentence's entries; the
  transposed copy makes each sentence one contiguous slice.
- `scenes/trigram_cube.py`: every char trigram as a point in a cube whose axes
  are its three characters. Shows the `’`/`'` confound, then the accuracy after
  folding apostrophes and after dropping Chronicle-only characters.
- `scenes/decision_boundary.py`: Naive Bayes is linear. Each test sentence's
  score is split into apostrophe, punctuation, and letter trigrams, so the
  decision boundary is a plane in 3D; seen edge-on it is the cut on a histogram
  of scores, replayed for the raw, apostrophe-folded, and shared-character
  regimes.
- `scenes/thread_scaling.py`: why 8 threads are not 8x faster. Timelines with
  one lane per thread show that only trigram counting (features) and the
  per-class sums (training) split across threads; the one-thread steps,
  mostly the matrix build and `sparse(transpose(X))`, set a floor (Amdahl's
  law). Ends on the measured speedups from `src/benchmark.jl`.
- `scenes/trigram_window.py`: a three-character window slides along a
  Chronicle phrase; each trigram gets its Naive Bayes weight, and the 50
  strongest clues for each corpus turn out to be mostly `’` or `'` trigrams.
- `scenes/confusion_morph.py`: the confusion matrix as one dot per test
  sentence. Dots move between cells from the word model to raw trigrams and
  then through the apostrophe-folded and shared-character regimes.
- `scenes/confound_reveal.py`: the two apostrophe code points and their counts
  per corpus, Tang/Bennett's missing punctuation, and the no-training
  exclusive-character rule that scores 98.2%.
- `scenes/tug_of_war.py`: the word model's score for two Chronicle test
  sentences, one word at a time on a number line, including a close call the
  word model gets wrong and the trigram model does not.
- `scenes/starfield_zoom.py`: all of `X_train` as points of light, with rows
  grouped by corpus and columns by apostrophe type; zooms show that
  Tang/Bennett rows never touch the `’` columns.

[uv](https://docs.astral.sh/uv/) manages the Python environment (and installs
Python 3.12 itself); LaTeX is not needed. From `animations/`:

```bash
uv sync
uv run manim -pql scenes/sparse_transpose.py SparseTranspose           # quick 480p preview
uv run manim -pqh scenes/trigram_cube.py TrigramCube                   # 1080p60 for posting
uv run manim -pqh scenes/trigram_cube.py TrigramCubeVertical           # 1080x1440, 60 fps
uv run manim -qh -a scenes/decision_boundary.py                        # both aspect ratios
```

Videos are written to `animations/media/videos/` (gitignored). The scenes read
`animations/data/*.json`, which two scripts regenerate from both corpora:

```bash
julia --project=src animations/export_data.jl          # features, trigram counts, NB scores
julia -t 8 --project=src animations/export_phases.jl   # per-phase timings, 1 to 8 threads
```

Every scene ends on a card crediting the data sources (`end_credits` in
`scenes/style.py`), and every caption carries the same credit (`data_credit`
in `social_posts.toml`). The Tang/Bennett authors require it; keep both when
adding scenes or posts.

`export_social.py` turns the renders into a posting kit in
`animations/social/` (gitignored), in the order and with the captions set in
`social_posts.toml`: for each post, an Instagram 3:4 feed video, a 9:16 Reel
(also used for LinkedIn's vertical video), the LinkedIn 16:9 video, covers,
caption text, and phone mockups; plus `schedule.md`, `schedule.csv`, and
overview mockups of the series. Render every scene at `-qh` first, then:

```bash
uv run python export_social.py
```

## Data Notes

- The Tang/Bennett corpus (version 0.1, released 14 April 2023; about 0.7
  million word tokens from religious texts, spoken transcripts, government
  documents, medical handbooks, and educational books) is **not redistributed**
  here. Its `readme.txt` reads: "You must not share/copy/reproduce any part of
  this corpus with anyone, unless you receive written permissions from the
  authors." The authors gave written permission (K. Tang, email, 2 October
  2026) to share the derived material listed below, including in social media
  posts, provided the data source is clearly credited. This repo stores only
  material derived from it:
  - source path configuration
  - index manifest (`src/datasets/register_manifest.csv`: line numbers and
    labels, no text)
  - the most discriminative words and character trigrams per class, with model
    weights, in `src/results/train_metrics.json` and
    `src/results/confound_check.json`
  - aggregate metrics, including the per-class trigram counts in
    `animations/data/trigram_cube.json` (the 250 most frequent trigrams per class)
    and the per-test-sentence model scores in `animations/data/nb_scores.json`
    (numbers and labels only, no text)
  - `animations/data/starfield.pgm`, the nonzero pattern of the training
    matrix summed over 4x4 blocks of cells, too coarse to recover any sentence
  - `animations/data/tug_of_war.json` and `trigram_window.json` quote only
    Chronicle sentences
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

## License

- **Code** (Julia, Python, and the GitHub Actions workflow): MIT; see
  [LICENSE](LICENSE). The repository layout and original paper workflow come
  from Jackson Burns's
  [18.337 project template](https://github.com/JacksonBurns/18337-project-template);
  all Julia and Python code is by William J. Wakefield.
- **Paper, figures, slides, and rendered animations** (`paper/`, the `.pptx`
  presentation, and videos produced from `animations/scenes/`):
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Reuse is welcome
  with attribution to William J. Wakefield. For the animations, Kevin Tang and
  Ryan Bennett are also designated to receive attribution as the source of the
  data, so reuse should keep the on-screen data credit.
- **Data is not covered by either license.** The Kaqchikel Chronicle text and
  documents in `src/datasets/` are used with permission and remain under their
  original terms. Files derived from the Tang/Bennett corpus
  (`src/datasets/register_manifest.csv`, the term lists in
  `src/results/train_metrics.json` and `src/results/confound_check.json`, and
  `animations/data/`) remain subject to that corpus's terms above.

To cite this work, use GitHub's "Cite this repository" button, which reads
[CITATION.cff](CITATION.cff).