#!/usr/bin/env julia
# confound_check.jl — is the char-trigram result linguistic, or orthographic?
#
# MOTIVATION
# All 20 top discriminative char trigrams in train_metrics.json contain either
# U+2019 (right single quotation mark) or a comma, and all 20 favour `classical`.
# The committed Chronicle text uses U+2019 for the glottal stop 15,982 times and
# straight U+0027 only 171 times. If Tang/Bennett uses the opposite convention,
# a char-trigram model can separate the classes on encoding alone, without
# learning anything about register.
#
# Supporting evidence: word_tfidf_matrix applies TextAnalysis `strip_punctuation`,
# which deletes both characters — and the word path scores 4.6 points LOWER
# (0.9433) than the char path (0.9894). That gap is the quantity under test.
#
# WHAT THIS DOES
#   1. Reports orthographic statistics for each corpus separately.
#   2. Fits two trivial baselines on the train split: a one-line rule that
#      flags any character never seen in modern training text, and a
#      sentence-length threshold.
#   3. Retrains char-trigram NB under four normalisation regimes and measures
#      how much of the discriminative signal rides on non-letter characters.
#   4. Compares every regime to the word-unigram baseline with 95% Wilson
#      intervals and an exact McNemar test on the shared test set.
#
# It deliberately does NOT decide anything. Run it, read the JSON, then choose
# whether this is a limitation to note or a finding to write up.
#
# Usage:
#   julia --project=src src/confound_check.jl
#   julia --project=src src/confound_check.jl --top-k 50

using JSON3
using Printf
using Statistics

include(joinpath(@__DIR__, "kaq_pipeline.jl"))
using .KaqPipeline

const APOSTROPHES = ['\u2019', '\u0027', '\u02BC', '\u2018', '`']
const PUNCT = [',', '.', ';', ':', '!', '?', '"', '(', ')', '[', ']', '-', '\u2014']

arg_or_default(flag, default) = begin
    idx = findfirst(==(flag), ARGS)
    (idx === nothing || idx == length(ARGS)) ? default : ARGS[idx + 1]
end

# ------------------------------------------------------- orthography profile

function orthography_profile(texts::Vector{String}, name::String)
    joined = join(texts, " ")
    nchars = length(joined)
    counts = Dict{String, Any}("corpus" => name, "n_sentences" => length(texts),
                               "n_chars" => nchars)

    for ch in APOSTROPHES
        c = count(==(ch), joined)
        counts["U+" * uppercase(string(UInt32(ch), base = 16, pad = 4))] = c
    end
    counts["comma"] = count(==(','), joined)
    counts["total_punct"] = sum(count(==(p), joined) for p in PUNCT)
    counts["digits"] = count(isdigit, joined)

    counts["apostrophes_per_1k_chars"] =
        1000 * sum(count(==(ch), joined) for ch in APOSTROPHES) / max(nchars, 1)
    counts["commas_per_1k_chars"] = 1000 * counts["comma"] / max(nchars, 1)

    # Which apostrophe convention dominates? This is the crux.
    curly = count(==('\u2019'), joined)
    straight = count(==('\u0027'), joined)
    counts["curly_share_of_apostrophes"] =
        (curly + straight) == 0 ? 0.0 : curly / (curly + straight)
    return counts
end

# ------------------------------------------------------ normalisation regimes

"""Regime A — the current pipeline. No change."""
normalize_raw(t::String) = t

"""Regime B — fold all apostrophe variants to U+0027. Removes the ENCODING
difference while preserving the glottal stop as a linguistic segment."""
function normalize_apostrophes(t::String)
    out = t
    for ch in ('\u2019', '\u02BC', '\u2018', '`')
        out = replace(out, ch => '\'')
    end
    return out
end

"""Regime C — fold apostrophes AND strip other punctuation. Approximates what
the word-unigram path sees, isolating punctuation density as a signal."""
function normalize_strip_punct(t::String)
    out = normalize_apostrophes(t)
    for p in PUNCT
        out = replace(out, p => ' ')
    end
    return replace(out, r"\s+" => " ") |> strip |> String
end

"""
Regime D — fold apostrophes, then keep only characters that occur in the
training text of EVERY class. Regime C's fixed list misses digits, curly double
quotes, underscores and en dashes, all of which occur only in the Chronicle.
The shared set is learned from the train split alone, so no test text informs it.
"""
function shared_charset_normalizer(texts, labels, train_idx)
    seen = Dict(c => Set{Char}() for c in unique(labels))
    for i in train_idx
        union!(seen[labels[i]], lowercase(normalize_apostrophes(texts[i])))
    end
    shared = intersect(values(seen)...)
    normalize(t::String) = begin
        kept = map(ch -> ch in shared ? ch : ' ', lowercase(normalize_apostrophes(t)))
        replace(kept, r"\s+" => " ") |> strip |> String
    end
    return normalize, shared
end

# ------------------------------------------------------------- statistics

"""95% Wilson score interval for a binomial proportion."""
function wilson_ci(p::Float64, n::Int; z::Float64 = 1.959964)
    denom = 1 + z^2 / n
    centre = (p + z^2 / (2n)) / denom
    half = z * sqrt(p * (1 - p) / n + z^2 / (4n^2)) / denom
    return [centre - half, centre + half]
end

"""
Exact two-sided McNemar test on paired predictions. Only the discordant pairs
carry information: `a_only` counts test items model A gets right and B gets
wrong, `b_only` the reverse.
"""
function mcnemar_exact(y, pred_a, pred_b)
    a_only = count(i -> pred_a[i] == y[i] && pred_b[i] != y[i], eachindex(y))
    b_only = count(i -> pred_a[i] != y[i] && pred_b[i] == y[i], eachindex(y))
    n = a_only + b_only
    p = n == 0 ? 1.0 :
        Float64(min(1, 2 * sum(binomial(big(n), k) for k in 0:min(a_only, b_only)) / big(2)^n))
    return Dict("a_only_correct" => a_only, "b_only_correct" => b_only, "p_value" => p)
end

accuracy(y, pred) = count(y .== pred) / length(y)

# -------------------------------------------------------- trivial baselines

"""
Predict `classical` iff a sentence contains a character that never appears in
modern TRAINING text. No model, no features — a set lookup.
"""
function exclusive_char_baseline(texts, labels, train_idx, test_idx)
    seen = Dict(c => Set{Char}() for c in unique(labels))
    for i in train_idx
        union!(seen[labels[i]], lowercase(texts[i]))
    end
    classical_only = setdiff(seen["classical"], seen["modern"])
    preds = [any(in(classical_only), lowercase(texts[i])) ? "classical" : "modern"
             for i in test_idx]
    return preds, classical_only
end

"""Best single token-count threshold on the train split, applied to test."""
function length_stump_baseline(texts, labels, train_idx, test_idx)
    ntok(i) = length(split(texts[i]))
    best_acc, best_th, long_class = 0.0, 0, ""
    for th in 1:100, long in ("classical", "modern")
        short = long == "classical" ? "modern" : "classical"
        acc = count(i -> (ntok(i) >= th ? long : short) == labels[i], train_idx) / length(train_idx)
        if acc > best_acc
            best_acc, best_th, long_class = acc, th, long
        end
    end
    short = long_class == "classical" ? "modern" : "classical"
    preds = [ntok(i) >= best_th ? long_class : short for i in test_idx]
    return preds, best_th, long_class
end

# ------------------------------------------------------------------ analysis

"""
True for characters that are part of the Kaqchikel alphabet as the model sees
it: letters, space, the U+0027 glottal stop, and the `^`/`\$` padding markers
added by the trigram extractor. Everything else is encoding or punctuation.
"""
is_orthographic_letter(c::Char) = isletter(c) || c in (' ', '\'', '^', '$')

function run_regime(name::String, transform::Function,
                    texts, labels, split; top_k::Int = 50)
    xf = String[transform(t) for t in texts]

    train_idx = findall(==("train"), split)
    test_idx  = findall(==("test"),  split)

    X, vocab = KaqPipeline.char_trigram_tfidf_matrix(xf; threaded = true)
    model = KaqPipeline.train_multinomial_nb(X[train_idx, :], labels[train_idx])
    preds = KaqPipeline.predict_multinomial_nb(model, X[test_idx, :])
    acc, conf, per_class =
        KaqPipeline.classification_metrics(labels[test_idx], preds, model.classes)

    top = KaqPipeline.top_discriminative_terms(model, vocab; top_n = top_k)

    # How much of the discriminative signal rides on non-letter characters?
    terms = collect(keys(top))
    n_nonletter = count(t -> any(!is_orthographic_letter, t), terms)

    favoring = [top[t]["favor_class"] for t in terms]
    class_skew = Dict(c => count(==(c), favoring) for c in unique(favoring))

    @printf("  %-22s acc=%.4f   non-letter top-%d: %d/%d (%.0f%%)\n",
        name, acc, top_k, n_nonletter, length(terms), 100 * n_nonletter / max(length(terms), 1))

    result = Dict(
        "regime" => name,
        "test_accuracy" => acc,
        "test_accuracy_ci95" => wilson_ci(acc, length(test_idx)),
        "vocabulary_size" => length(vocab),
        "confusion_matrix" => vec(conf),
        "classes" => model.classes,
        "per_class" => per_class,
        "top_k" => top_k,
        "top_terms_nonletter" => n_nonletter,
        "top_terms_total" => length(terms),
        "top_terms_nonletter_fraction" => n_nonletter / max(length(terms), 1),
        "top_terms_class_skew" => class_skew,
        "top_terms" => top,
    )
    return result, preds
end

function main()
    top_k = parse(Int, arg_or_default("--top-k", "50"))
    tang_path = arg_or_default("--tang-path", KaqPipeline.default_tang_path())
    manifest_path = arg_or_default("--manifest", KaqPipeline.DEFAULT_MANIFEST_CSV)

    if !isfile(tang_path)
        error("""
        Tang/Bennett corpus not found at:
          $tang_path
        This diagnostic needs BOTH corpora — the whole question is whether they
        differ orthographically. Set KAQ_TANG_PATH or src/local_config.toml.
        """)
    end

    println("\n=== Orthography profiles ===")
    classical = KaqPipeline.load_kiwujil_sentences()
    modern    = KaqPipeline.load_tang_sentences(tang_path)

    prof_c = orthography_profile(classical, "classical_chronicle")
    prof_m = orthography_profile(modern,    "modern_tang_bennett")

    for p in (prof_c, prof_m)
        @printf("  %-22s curly=%-7d straight=%-7d curly_share=%.3f  commas/1k=%.2f\n",
            p["corpus"], p["U+2019"], p["U+0027"],
            p["curly_share_of_apostrophes"], p["commas_per_1k_chars"])
    end

    delta = abs(prof_c["curly_share_of_apostrophes"] - prof_m["curly_share_of_apostrophes"])
    @printf("\n  curly-share divergence between corpora: %.3f\n", delta)
    if delta > 0.5
        println("  >> LARGE. The corpora use different apostrophe encodings.")
        println("  >> Char trigrams can separate classes on encoding alone.")
    elseif delta > 0.15
        println("  >> MODERATE. Encoding contributes some separable signal.")
    else
        println("  >> SMALL. Encoding is unlikely to explain the char-trigram margin.")
    end

    texts, labels, split = KaqPipeline.load_manifest_texts(
        tang_path = tang_path, manifest_path = manifest_path)
    train_idx = findall(==("train"), split)
    test_idx  = findall(==("test"),  split)
    y_test = labels[test_idx]
    n_test = length(test_idx)

    println("\n=== Reference: word unigram NB (same pipeline as train_classifier.jl) ===")
    word_X, _ = KaqPipeline.word_tfidf_matrix(texts)
    word_model = KaqPipeline.train_multinomial_nb(word_X[train_idx, :], labels[train_idx])
    word_preds = KaqPipeline.predict_multinomial_nb(word_model, word_X[test_idx, :])
    word_acc = accuracy(y_test, word_preds)
    @printf("  word_unigram           acc=%.4f\n", word_acc)

    println("\n=== Trivial baselines (fitted on train split) ===")
    rule_preds, classical_only = exclusive_char_baseline(texts, labels, train_idx, test_idx)
    rule_acc = accuracy(y_test, rule_preds)
    test_classical = findall(==("classical"), y_test)
    rule_coverage = count(i -> rule_preds[i] == "classical", test_classical) / length(test_classical)
    @printf("  exclusive-char rule    acc=%.4f   (%d chars never seen in modern train; flags %.1f%% of classical test sentences)\n",
        rule_acc, length(classical_only), 100 * rule_coverage)

    stump_preds, stump_th, stump_long = length_stump_baseline(texts, labels, train_idx, test_idx)
    stump_acc = accuracy(y_test, stump_preds)
    @printf("  length stump           acc=%.4f   (>= %d tokens => %s)\n", stump_acc, stump_th, stump_long)

    println("\n=== Ablation: char trigram NB under four regimes ===")
    shared_normalizer, shared_chars = shared_charset_normalizer(texts, labels, train_idx)
    regimes = [
        ("A_raw",             normalize_raw),
        ("B_apostrophe_fold", normalize_apostrophes),
        ("C_fold_and_strip",  normalize_strip_punct),
        ("D_shared_charset",  shared_normalizer),
    ]
    results = Dict{String, Any}[]
    for (name, f) in regimes
        result, preds = run_regime(name, f, texts, labels, split; top_k = top_k)
        result["mcnemar_vs_word_unigram"] = mcnemar_exact(y_test, preds, word_preds)
        push!(results, result)
    end

    println("\n=== Paired comparison against word unigram (exact McNemar) ===")
    for r in results
        m = r["mcnemar_vs_word_unigram"]
        @printf("  %-22s delta=%+.4f   regime-only correct=%-3d word-only correct=%-3d p=%.3g\n",
            r["regime"], r["test_accuracy"] - word_acc,
            m["a_only_correct"], m["b_only_correct"], m["p_value"])
    end

    acc_a, acc_c = results[1]["test_accuracy"], results[3]["test_accuracy"]
    @printf("\n  accuracy attributable to punctuation/orthography: %.4f (%.1f points)\n",
        acc_a - acc_c, 100 * (acc_a - acc_c))

    out = Dict(
        "orthography" => [prof_c, prof_m],
        "curly_share_divergence" => delta,
        "test_count" => n_test,
        "word_unigram_baseline" => Dict(
            "test_accuracy" => word_acc,
            "test_accuracy_ci95" => wilson_ci(word_acc, n_test),
        ),
        "trivial_baselines" => Dict(
            "exclusive_char_rule" => Dict(
                "test_accuracy" => rule_acc,
                "test_accuracy_ci95" => wilson_ci(rule_acc, n_test),
                "classical_recall" => rule_coverage,
                "classical_only_chars" => join(sort!(collect(classical_only))),
                "mcnemar_vs_word_unigram" => mcnemar_exact(y_test, rule_preds, word_preds),
            ),
            "length_stump" => Dict(
                "test_accuracy" => stump_acc,
                "test_accuracy_ci95" => wilson_ci(stump_acc, n_test),
                "threshold_tokens" => stump_th,
                "long_class" => stump_long,
            ),
        ),
        "shared_charset" => join(sort!(collect(shared_chars))),
        "regimes" => results,
        "accuracy_delta_raw_minus_stripped" => acc_a - acc_c,
        "interpretation_note" =>
            "Regime D keeps only characters present in both classes' training text. " *
            "If D is statistically indistinguishable from the word unigram baseline, " *
            "the char-trigram advantage is orthographic, not linguistic.",
    )

    out_path = joinpath(@__DIR__, "results", "confound_check.json")
    mkpath(dirname(out_path))
    open(out_path, "w") do io
        JSON3.pretty(io, out)
    end
    println("\n  wrote: ", out_path)
end

main()
