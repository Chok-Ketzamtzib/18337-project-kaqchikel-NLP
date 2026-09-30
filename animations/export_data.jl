#!/usr/bin/env julia
# export_data.jl — writes the small JSON inputs read by the Manim scenes.
#
# Uses the same manifest, train split, and trigram extractor as the model, so
# the animations show exactly what the classifier saw. Only aggregate counts
# are written: the Tang/Bennett licence forbids redistributing its text.
#
# Outputs:
#   animations/data/feature_matrix.json  shape and sparsity of X_train
#   animations/data/trigram_cube.json    most frequent trigrams per class, with
#                                        counts before and after folding
#                                        apostrophes (regime B of
#                                        confound_check.jl)
#   animations/data/nb_scores.json       per test sentence: label and Naive
#                                        Bayes log-odds under regimes A, B, D
#                                        and the word model, with each trigram
#                                        score split by feature type
#   animations/data/tug_of_war.json      Chronicle test sentences with the push
#                                        of each word under the word model,
#                                        and their trigram-model score
#   animations/data/trigram_window.json  the trigrams of one Chronicle phrase
#                                        with their regime-A weights
#   animations/data/starfield.pgm        X_train nonzeros summed over 4x4
#   animations/data/starfield.json       blocks, rows grouped by class and
#                                        columns by apostrophe type; too coarse
#                                        to recover any sentence
#
# Usage (from the repo root; needs both corpora, like confound_check.jl):
#   julia --project=src animations/export_data.jl
#   julia --project=src animations/export_data.jl --top-per-class 300

using JSON3
using LinearAlgebra
using SparseArrays
using TextAnalysis

include(joinpath(@__DIR__, "..", "src", "kaq_pipeline.jl"))
using .KaqPipeline

arg_or_default(flag, default) = begin
    idx = findfirst(==(flag), ARGS)
    (idx === nothing || idx == length(ARGS)) ? default : ARGS[idx + 1]
end

"""Regime B of confound_check.jl: every apostrophe variant becomes U+0027."""
fold_apostrophes(t::AbstractString) =
    replace(t, '\u2019' => '\'', '\u02BC' => '\'', '\u2018' => '\'', '`' => '\'')

"""Regime D of confound_check.jl: fold apostrophes, then blank every character
not seen in the training text of both classes."""
function shared_charset_normalizer(texts, labels, train_idx)
    seen = Dict(c => Set{Char}() for c in unique(labels))
    for i in train_idx
        union!(seen[labels[i]], lowercase(fold_apostrophes(texts[i])))
    end
    shared = intersect(values(seen)...)
    normalize(t) = begin
        kept = map(ch -> ch in shared ? ch : ' ', lowercase(fold_apostrophes(t)))
        replace(kept, r"\s+" => " ") |> strip |> String
    end
    return normalize, shared
end

const APOSTROPHES = ('\u2019', '\'', '\u02BC', '\u2018', '`')
const PART_NAMES = ("apostrophe", "other_symbols", "letters")

"""Same test as confound_check.jl; `^` and `\$` are the line padding."""
is_orthographic_letter(c::Char) = isletter(c) || c in (' ', '\'', '^', '\$')

function part_index(term::AbstractString)
    any(in(APOSTROPHES), term) && return 1
    any(!is_orthographic_letter, term) && return 2
    return 3
end

"""
Train NB on `train_idx` and return the log-odds (classical minus modern) of
every test row. `predict_multinomial_nb` picks classical exactly when this is
>= 0. With `vocab`, the score is also split into one sum per `PART_NAMES`
feature type; the parts plus `offset` add up to the score.
"""
function nb_scores(X, labels, train_idx, test_idx; vocab = nothing)
    model = KaqPipeline.train_multinomial_nb(X[train_idx, :], labels[train_idx])
    @assert model.classes == ["classical", "modern"]
    w = model.log_likelihood[1, :] .- model.log_likelihood[2, :]
    offset = model.log_priors[1] - model.log_priors[2]
    X_test = X[test_idx, :]
    score = X_test * w .+ offset

    preds = KaqPipeline.predict_multinomial_nb(model, X_test)
    disagree = count((score .>= 0) .!= (preds .== "classical"))
    disagree == 0 || @warn "$disagree test rows where sign(score) differs from predict_multinomial_nb"

    out = Dict{String, Any}(
        "accuracy" => count(preds .== labels[test_idx]) / length(test_idx),
        "offset" => offset,
        "score" => round.(score; digits = 3),
    )
    if vocab !== nothing
        idx = part_index.(vocab)
        out["parts"] = Dict(name => round.(X_test * (w .* (idx .== k)); digits = 3)
                            for (k, name) in enumerate(PART_NAMES))
    end
    return out
end

function trigram_counts(texts, labels, idx)
    counts = Dict(c => Dict{String, Int}() for c in unique(labels))
    for i in idx
        bucket = counts[labels[i]]
        for tri in KaqPipeline._line_char_trigrams(texts[i])
            bucket[tri] = get(bucket, tri, 0) + 1
        end
    end
    return counts
end

function fold_counts(bucket::Dict{String, Int})
    folded = Dict{String, Int}()
    for (tri, n) in bucket
        key = fold_apostrophes(tri)
        folded[key] = get(folded, key, 0) + n
    end
    return folded
end

function top_trigrams(bucket::Dict{String, Int}, k::Int)
    ranked = sort!(collect(bucket); by = p -> (-p.second, p.first))
    return first.(ranked[1:min(k, length(ranked))])
end

nb_weights(model) = model.log_likelihood[1, :] .- model.log_likelihood[2, :]

"""Word tokens exactly as `word_tfidf_matrix` produces them, grouped by the
whitespace word they came from (strip_punctuation also deletes ’ and ')."""
function word_tokens(word::AbstractString)
    doc = StringDocument(lowercase(word))
    prepare!(doc, strip_punctuation)
    return String.(tokens(doc))
end

"""
Per-word pushes of Chronicle test sentences under the word model. A term that
occurs k times contributes X[row, term] * w[term] / k per occurrence, so the
pushes plus the offset add up to the word-model score.
"""
function tug_candidates(texts, labels, train_idx, test_idx, trigram_score; n_keep = 8)
    word_X, terms = KaqPipeline.word_tfidf_matrix(texts)
    model = KaqPipeline.train_multinomial_nb(word_X[train_idx, :], labels[train_idx])
    w = nb_weights(model)
    offset = model.log_priors[1] - model.log_priors[2]
    col = Dict(t => j for (j, t) in enumerate(terms))

    rows = Dict{String, Any}[]
    for (k, i) in enumerate(test_idx)
        labels[i] == "classical" || continue
        words = split(texts[i])
        7 <= length(words) <= 12 || continue
        groups = word_tokens.(words)
        all_tokens = reduce(vcat, groups)
        occurrences = Dict{String, Int}()
        for t in all_tokens
            occurrences[t] = get(occurrences, t, 0) + 1
        end
        push_of(t) = word_X[i, col[t]] * w[col[t]] / occurrences[t]
        pushes = [sum(push_of, g; init = 0.0) for g in groups]
        score = offset + sum(pushes)
        @assert isapprox(score, offset + dot(word_X[i, :], w); atol = 1e-8)
        push!(rows, Dict(
            "words" => String.(words),
            "tokens" => groups,
            "push" => round.(pushes; digits = 3),
            "word_score" => round(score; digits = 3),
            "trigram_score" => round(trigram_score[k]; digits = 3),
        ))
    end

    tugged(r) = sum(p for p in r["push"] if p < 0; init = 0.0)
    correct = filter(r -> 1.0 <= r["word_score"] <= 6.0, rows)
    sort!(correct; by = tugged)
    close_calls = filter(r -> -2.5 <= r["word_score"] < 0 && r["trigram_score"] > 0, rows)
    sort!(close_calls; by = r -> tugged(r))
    return Dict(
        "source" => "word-unigram multinomial NB; Chronicle test sentences only",
        "offset" => offset,
        "correct" => correct[1:min(n_keep, end)],
        "close_calls" => close_calls[1:min(n_keep, end)],
        "misclassified_classical" => count(r -> r["word_score"] < 0, rows),
        "classical_candidates" => length(rows),
    )
end

const WINDOW_PHRASE = "K\u2019a ri\u2019 xek\u2019ule\u2019"

function window_export(X, vocab, labels, train_idx)
    model = KaqPipeline.train_multinomial_nb(X[train_idx, :], labels[train_idx])
    w = nb_weights(model)
    col = Dict(t => j for (j, t) in enumerate(vocab))
    trigrams = KaqPipeline._line_char_trigrams(WINDOW_PHRASE)
    order = sortperm(w; rev = true)
    return Dict(
        "phrase" => WINDOW_PHRASE,
        "padded" => "^" * lowercase(WINDOW_PHRASE) * "\$",
        "trigrams" => [Dict("trigram" => t, "weight" => round(w[col[t]]; digits = 3),
                            "part" => PART_NAMES[part_index(t)]) for t in trigrams],
        "top_classical" => [Dict("trigram" => vocab[j], "weight" => round(w[j]; digits = 3),
                                 "part" => PART_NAMES[part_index(vocab[j])]) for j in order[1:50]],
        "top_modern" => [Dict("trigram" => vocab[j], "weight" => round(w[j]; digits = 3),
                              "part" => PART_NAMES[part_index(vocab[j])]) for j in reverse(order)[1:50]],
    )
end

band_of(term) = '\u2019' in term ? 1 : ('\'' in term ? 3 : 2)

function starfield_export(X_train, vocab, train_labels, out_dir; block = 4)
    row_perm = vcat(findall(==("classical"), train_labels), findall(==("modern"), train_labels))
    bands = band_of.(vocab)
    col_perm = sortperm(bands; alg = MergeSort)
    Xp = X_train[row_perm, col_perm]
    h, wd = cld.(size(Xp), block)
    img = zeros(Int, h, wd)
    rows, cols, _ = findnz(Xp)
    for (r, c) in zip(rows, cols)
        img[cld(r, block), cld(c, block)] += 1
    end
    open(joinpath(out_dir, "starfield.pgm"), "w") do io
        write(io, "P5\n$wd $h\n255\n")
        write(io, UInt8.(min.(img, 255))')
    end
    println("  wrote: ", KaqPipeline.repo_relpath(joinpath(out_dir, "starfield.pgm")))

    n_classical = count(==("classical"), train_labels)
    sorted_bands = bands[col_perm]
    nnz_by = Dict{String, Dict{String, Int}}()
    for (cls, rr) in (("classical", 1:n_classical), ("modern", (n_classical + 1):size(Xp, 1)))
        sub = Xp[rr, :]
        nnz_by[cls] = Dict(name => nnz(sub[:, sorted_bands .== b])
                           for (b, name) in enumerate(("curly", "shared", "straight")))
    end
    write_json(joinpath(out_dir, "starfield.json"), Dict(
        "source" => "char trigram TF-IDF nonzeros of X_train, summed over $(block)x$(block) blocks",
        "block" => block,
        "width" => wd,
        "height" => h,
        "rows" => size(Xp, 1),
        "cols" => size(Xp, 2),
        "classical_rows" => n_classical,
        "band_cols" => Dict(name => count(==(b), bands)
                            for (b, name) in enumerate(("curly", "shared", "straight"))),
        "nnz" => nnz(Xp),
        "nnz_by_class_band" => nnz_by,
        "max_block" => maximum(img),
    ))
end

function write_json(path, data; pretty = true)
    mkpath(dirname(path))
    open(path, "w") do io
        pretty ? JSON3.pretty(io, data) : JSON3.write(io, data)
    end
    println("  wrote: ", KaqPipeline.repo_relpath(path))
end

function main()
    top_per_class = parse(Int, arg_or_default("--top-per-class", "250"))
    tang_path = arg_or_default("--tang-path", KaqPipeline.default_tang_path())
    manifest_path = arg_or_default("--manifest", KaqPipeline.DEFAULT_MANIFEST_CSV)
    isfile(tang_path) || error("""
        Tang/Bennett corpus not found at:
          $tang_path
        Set KAQ_TANG_PATH or src/local_config.toml.
        """)

    texts, labels, split = KaqPipeline.load_manifest_texts(
        tang_path = tang_path, manifest_path = manifest_path)
    train_idx = findall(==("train"), split)
    out_dir = joinpath(@__DIR__, "data")

    X, vocab = KaqPipeline.char_trigram_tfidf_matrix(texts)
    X_train = X[train_idx, :]
    write_json(joinpath(out_dir, "feature_matrix.json"), Dict(
        "source" => "char trigram TF-IDF, train split of $(KaqPipeline.repo_relpath(manifest_path))",
        "n_docs" => size(X_train, 1),
        "n_features" => size(X_train, 2),
        "nnz" => nnz(X_train),
        "density" => nnz(X_train) / length(X_train),
    ))

    raw = trigram_counts(texts, labels, train_idx)
    folded = Dict(c => fold_counts(bucket) for (c, bucket) in raw)
    selected = sort!(collect(union((top_trigrams(b, top_per_class) for b in values(raw))...)))

    shared_normalize, shared = shared_charset_normalizer(texts, labels, train_idx)

    rows = map(selected) do tri
        key = fold_apostrophes(tri)
        Dict(
            "trigram" => tri,
            "classical" => get(raw["classical"], tri, 0),
            "modern" => get(raw["modern"], tri, 0),
            "folded" => key,
            "folded_classical" => get(folded["classical"], key, 0),
            "folded_modern" => get(folded["modern"], key, 0),
        )
    end

    write_json(joinpath(out_dir, "trigram_cube.json"), Dict(
        "source" => "char trigram counts, train split of $(KaqPipeline.repo_relpath(manifest_path))",
        "top_per_class" => top_per_class,
        "trigram_totals" => Dict(c => sum(values(b)) for (c, b) in raw),
        "distinct_trigrams" => Dict(c => length(b) for (c, b) in raw),
        "shared_chars_after_fold" => join(sort!(collect(shared))),
        "trigrams" => rows,
    ))

    test_idx = findall(==("test"), split)
    regimes = [
        ("A_raw", identity),
        ("B_apostrophe_fold", fold_apostrophes),
        ("D_shared_charset", shared_normalize),
    ]
    scores = Dict{String, Any}()
    for (name, transform) in regimes
        X_r, vocab_r = KaqPipeline.char_trigram_tfidf_matrix(String[transform(t) for t in texts])
        scores[name] = nb_scores(X_r, labels, train_idx, test_idx; vocab = vocab_r)
        println("  ", rpad(name, 20), " acc=", round(scores[name]["accuracy"]; digits = 4))
    end
    word_X, _ = KaqPipeline.word_tfidf_matrix(texts)
    scores["word_unigram"] = nb_scores(word_X, labels, train_idx, test_idx)
    println("  ", rpad("word_unigram", 20), " acc=", round(scores["word_unigram"]["accuracy"]; digits = 4))

    write_json(joinpath(out_dir, "nb_scores.json"), Dict(
        "source" => "multinomial NB log-odds (classical minus modern), test split of $(KaqPipeline.repo_relpath(manifest_path))",
        "labels" => labels[test_idx],
        "parts" => collect(PART_NAMES),
        "regimes" => scores,
    ); pretty = false)

    tug = tug_candidates(texts, labels, train_idx, test_idx, scores["A_raw"]["score"])
    println("  tug-of-war: ", length(tug["correct"]), " correct, ", length(tug["close_calls"]), " close calls")
    write_json(joinpath(out_dir, "tug_of_war.json"), tug)
    write_json(joinpath(out_dir, "trigram_window.json"), window_export(X, vocab, labels, train_idx))
    starfield_export(X_train, vocab, labels[train_idx], out_dir)
end

main()
