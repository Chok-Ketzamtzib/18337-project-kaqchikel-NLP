#!/usr/bin/env julia
# export_phases.jl — times each phase of the two threaded stages that
# src/benchmark.jl measures, for the thread-scaling animation.
#
# benchmark.jl reports one wall time per stage. The animation splits each
# stage into the part every worker runs and the parts one thread runs, so this
# script replays char_trigram_tfidf_matrix and train_multinomial_nb step by
# step, with the pipeline's own chunking and counting helpers, and times each
# step. One difference: the pipeline merges each task's result as soon as it
# is fetched, overlapping the merge with tasks still running; here every task
# is awaited before the merge starts, so the two phases are timed apart.
# The real pipeline functions are timed too, as a check on the sum.
#
# Output: animations/data/thread_phases.json
#
# Usage (from the repo root; needs both corpora, like benchmark.jl):
#   julia -t 8 --project=src animations/export_phases.jl [--samples 30]

using JSON3
using LinearAlgebra
using SparseArrays
using Statistics

include(joinpath(@__DIR__, "..", "src", "kaq_pipeline.jl"))
using .KaqPipeline
using Base.Threads: @spawn, nthreads

arg_or_default(flag, default) = begin
    idx = findfirst(==(flag), ARGS)
    (idx === nothing || idx == length(ARGS)) ? default : ARGS[idx + 1]
end

seconds_since(t0) = (time_ns() - t0) / 1e9

"""char_trigram_tfidf_matrix, split into count (parallel), merge, and build."""
function feature_phases(texts, workers)
    n_docs = length(texts)
    doc_counts = Vector{Dict{String, Int}}(undef, n_docs)
    df_counts = Dict{String, Int}()
    t = Dict{String, Float64}()

    t0 = time_ns()
    if workers == 1
        # The serial path adds to df_counts as it goes, so it has no merge.
        for i in 1:n_docs
            counts = Dict{String, Int}()
            for tri in KaqPipeline._line_char_trigrams(texts[i])
                counts[tri] = get(counts, tri, 0) + 1
            end
            for tri in keys(counts)
                df_counts[tri] = get(df_counts, tri, 0) + 1
            end
            doc_counts[i] = counts
        end
        t["count"] = seconds_since(t0)
        t["merge"] = 0.0
    else
        ranges = KaqPipeline._chunk_ranges(n_docs, min(workers, n_docs))
        tasks = map(chunk -> @spawn(KaqPipeline._count_trigrams_chunk(texts, chunk)), ranges)
        results = fetch.(tasks)
        t["count"] = seconds_since(t0)

        t0 = time_ns()
        for (chunk, (local_docs, local_df)) in zip(ranges, results)
            for (offset, i) in enumerate(chunk)
                doc_counts[i] = local_docs[offset]
            end
            for (tri, cnt) in local_df
                df_counts[tri] = get(df_counts, tri, 0) + cnt
            end
        end
        t["merge"] = seconds_since(t0)
    end

    t0 = time_ns()
    vocab = sort!(collect(keys(df_counts)))
    vocab_index = Dict(term => idx for (idx, term) in enumerate(vocab))
    row_idx, col_idx, vals = Int[], Int[], Float64[]
    for i in 1:n_docs
        for (tri, count) in doc_counts[i]
            push!(row_idx, i)
            push!(col_idx, vocab_index[tri])
            push!(vals, float(count))
        end
    end
    tf = sparse(row_idx, col_idx, vals, n_docs, length(vocab))
    idf = [log((n_docs + 1) / (df_counts[term] + 1)) + 1.0 for term in vocab]
    tfidf = tf * Diagonal(idf)
    t["build"] = seconds_since(t0)
    return t, tfidf
end

"""train_multinomial_nb, split into setup, transpose, accumulate (parallel), merge, and finish."""
function nb_phases(X, y, workers; alpha = 1.0)
    t = Dict{String, Float64}()

    t0 = time_ns()
    classes = sort(unique(y))
    class_to_idx = Dict(c => i for (i, c) in enumerate(classes))
    y_idx = [class_to_idx[label] for label in y]
    n_docs, n_features = size(X)
    n_classes = length(classes)
    class_counts = zeros(Float64, n_classes)
    for c in y_idx
        class_counts[c] += 1
    end
    log_priors = log.(class_counts ./ n_docs)
    feature_sums = zeros(Float64, n_classes, n_features)
    t["setup"] = seconds_since(t0)

    t0 = time_ns()
    Xt = sparse(transpose(X))
    t["transpose"] = seconds_since(t0)

    t0 = time_ns()
    if workers == 1
        for doc_idx in 1:n_docs
            c = y_idx[doc_idx]
            for ptr in nzrange(Xt, doc_idx)
                feature_sums[c, Xt.rowval[ptr]] += Xt.nzval[ptr]
            end
        end
        t["accumulate"] = seconds_since(t0)
        t["merge"] = 0.0
    else
        ranges = KaqPipeline._chunk_ranges(n_docs, min(workers, n_docs))
        tasks = map(ranges) do chunk
            @spawn begin
                local_sum = zeros(Float64, n_classes, n_features)
                for doc_idx in chunk
                    c = y_idx[doc_idx]
                    for ptr in nzrange(Xt, doc_idx)
                        local_sum[c, Xt.rowval[ptr]] += Xt.nzval[ptr]
                    end
                end
                local_sum
            end
        end
        sums = fetch.(tasks)
        t["accumulate"] = seconds_since(t0)

        t0 = time_ns()
        for s in sums
            feature_sums .+= s
        end
        t["merge"] = seconds_since(t0)
    end

    t0 = time_ns()
    log_likelihood = zeros(Float64, n_classes, n_features)
    for c in 1:n_classes
        denom = sum(view(feature_sums, c, :)) + alpha * n_features
        for f in 1:n_features
            log_likelihood[c, f] = log((feature_sums[c, f] + alpha) / denom)
        end
    end
    t["finish"] = seconds_since(t0)
    return t, (log_priors, log_likelihood)
end

"""
Median seconds of each phase, and of the pipeline function itself, over
`samples` runs. The heap is collected before every run, so a collection
triggered by earlier garbage cannot land in (and inflate) an arbitrary phase.
"""
function profile(phased, pipeline, phases, workers, samples)
    phased(workers)
    pipeline(workers)
    runs = map(1:samples) do _
        GC.gc()
        first(phased(workers))
    end
    totals = Float64[]
    for _ in 1:samples
        GC.gc()
        t0 = time_ns()
        pipeline(workers)
        push!(totals, seconds_since(t0))
    end
    out = Dict{String, Float64}(p => median(r[p] for r in runs) for p in phases)
    out["pipeline_total"] = median(totals)
    return out
end

function main()
    samples = parse(Int, arg_or_default("--samples", "30"))
    tang_path = arg_or_default("--tang-path", KaqPipeline.default_tang_path())
    manifest_path = arg_or_default("--manifest", KaqPipeline.DEFAULT_MANIFEST_CSV)
    isfile(tang_path) || error("Tang/Bennett corpus not found at $tang_path")
    nthreads() >= 8 || @warn "Started with $(nthreads()) threads; run with julia -t 8 for the full sweep."

    texts, labels, split = KaqPipeline.load_manifest_texts(
        tang_path = tang_path, manifest_path = manifest_path)
    train_idx = findall(==("train"), split)
    texts, labels = texts[train_idx], labels[train_idx]
    worker_counts = [w for w in (1, 2, 4, 8) if w <= nthreads()]

    reference, _ = KaqPipeline.char_trigram_tfidf_matrix(texts)
    reference_model = KaqPipeline.train_multinomial_nb(reference, labels)
    for w in worker_counts
        _, X = feature_phases(texts, w)
        X == reference || error("phased feature extraction differs from the pipeline at $w workers")
        _, (priors, ll) = nb_phases(reference, labels, w)
        (priors ≈ reference_model.log_priors && ll ≈ reference_model.log_likelihood) ||
            error("phased NB training differs from the pipeline at $w workers")
    end

    pipeline_features(w) = w == 1 ?
        KaqPipeline.char_trigram_tfidf_matrix(texts) :
        KaqPipeline.char_trigram_tfidf_matrix(texts; threaded = true, worker_count = w)
    pipeline_nb(w) = w == 1 ?
        KaqPipeline.train_multinomial_nb(reference, labels) :
        KaqPipeline.train_multinomial_nb(reference, labels; threaded = true, worker_count = w)

    stages = [
        ("feature_extraction", ["count", "merge", "build"], "count",
         w -> feature_phases(texts, w), pipeline_features),
        ("naive_bayes_train", ["setup", "transpose", "accumulate", "merge", "finish"], "accumulate",
         w -> nb_phases(reference, labels, w), pipeline_nb),
    ]
    results = Dict{String, Any}(
        "source" => "train split of $(KaqPipeline.repo_relpath(manifest_path))",
        "rows" => length(texts),
        "threads" => nthreads(),
        "samples" => samples,
        "method" => "median of time_ns() per phase over $samples runs, after one warmup, GC.gc() before each run",
    )
    for (name, phases, parallel, phased, pipeline) in stages
        per_worker = Dict(string(w) => profile(phased, pipeline, phases, w, samples) for w in worker_counts)
        results[name] = Dict("phases" => phases, "parallel_phase" => parallel, "workers" => per_worker)
        println(name)
        for w in worker_counts
            r = per_worker[string(w)]
            parts = join(("$p=$(round(1e3 * r[p]; digits = 2))" for p in phases), " ")
            println("  $w workers: ", parts, " ms | sum=", round(1e3 * sum(r[p] for p in phases); digits = 2),
                    " pipeline=", round(1e3 * r["pipeline_total"]; digits = 2), " ms")
        end
    end

    out = joinpath(@__DIR__, "data", "thread_phases.json")
    mkpath(dirname(out))
    open(io -> JSON3.pretty(io, results), out, "w")
    println("  wrote: ", KaqPipeline.repo_relpath(out))
end

main()
