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
#   2. Measures what share of top-K discriminative mass is punctuation-bearing.
#   3. Retrains char-trigram NB under three normalisation regimes and reports
#      the accuracy delta.
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

# ------------------------------------------------------------------ analysis

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

    # How much of the discriminative signal rides on punctuation?
    terms = collect(keys(top))
    bears_punct(t) = any(c -> c in APOSTROPHES || c in PUNCT, collect(t))
    n_punct = count(bears_punct, terms)

    favoring = [top[t]["favor_class"] for t in terms]
    class_skew = Dict(c => count(==(c), favoring) for c in unique(favoring))

    @printf("  %-22s acc=%.4f   punct-bearing top-%d: %d/%d (%.0f%%)\n",
        name, acc, top_k, n_punct, length(terms), 100 * n_punct / max(length(terms), 1))

    return Dict(
        "regime" => name,
        "test_accuracy" => acc,
        "vocabulary_size" => length(vocab),
        "confusion_matrix" => vec(conf),
        "classes" => model.classes,
        "per_class" => per_class,
        "top_k" => top_k,
        "top_terms_punct_bearing" => n_punct,
        "top_terms_total" => length(terms),
        "top_terms_punct_fraction" => n_punct / max(length(terms), 1),
        "top_terms_class_skew" => class_skew,
        "top_terms" => top,
    )
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

    println("\n=== Ablation: char trigram NB under three regimes ===")
    texts, labels, split = KaqPipeline.load_manifest_texts(
        tang_path = tang_path, manifest_path = manifest_path)

    regimes = [
        ("A_raw",             normalize_raw),
        ("B_apostrophe_fold", normalize_apostrophes),
        ("C_fold_and_strip",  normalize_strip_punct),
    ]
    results = [run_regime(n, f, texts, labels, split; top_k = top_k) for (n, f) in regimes]

    acc_a, acc_c = results[1]["test_accuracy"], results[3]["test_accuracy"]
    @printf("\n  accuracy attributable to punctuation/orthography: %.4f (%.1f points)\n",
        acc_a - acc_c, 100 * (acc_a - acc_c))
    @printf("  word-unigram baseline for reference: 0.9433\n")

    out = Dict(
        "orthography" => [prof_c, prof_m],
        "curly_share_divergence" => delta,
        "regimes" => results,
        "accuracy_delta_raw_minus_stripped" => acc_a - acc_c,
        "interpretation_note" =>
            "Regime C approximates what the word-unigram path sees. If C ≈ 0.9433, " *
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
