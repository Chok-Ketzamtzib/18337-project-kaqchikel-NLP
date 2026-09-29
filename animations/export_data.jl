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
#
# Usage (from the repo root; needs both corpora, like confound_check.jl):
#   julia --project=src animations/export_data.jl
#   julia --project=src animations/export_data.jl --top-per-class 300

using JSON3
using SparseArrays

include(joinpath(@__DIR__, "..", "src", "kaq_pipeline.jl"))
using .KaqPipeline

arg_or_default(flag, default) = begin
    idx = findfirst(==(flag), ARGS)
    (idx === nothing || idx == length(ARGS)) ? default : ARGS[idx + 1]
end

"""Regime B of confound_check.jl: every apostrophe variant becomes U+0027."""
fold_apostrophes(t::AbstractString) =
    replace(t, '\u2019' => '\'', '\u02BC' => '\'', '\u2018' => '\'', '`' => '\'')

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

function write_json(path, data)
    mkpath(dirname(path))
    open(path, "w") do io
        JSON3.pretty(io, data)
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

    X, _ = KaqPipeline.char_trigram_tfidf_matrix(texts)
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

    # Characters each class uses once apostrophes are folded; mirrors
    # shared_charset_normalizer in confound_check.jl (regime D).
    seen = Dict(c => Set{Char}() for c in keys(raw))
    for i in train_idx
        union!(seen[labels[i]], lowercase(fold_apostrophes(texts[i])))
    end
    shared = intersect(values(seen)...)

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
end

main()
