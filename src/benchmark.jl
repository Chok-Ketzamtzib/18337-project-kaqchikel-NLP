#!/usr/bin/env julia
# benchmark.jl — serial vs threaded scaling for TF-IDF feature extraction and
# multinomial Naive Bayes training.
#
# Changes vs the previous version:
#   * Uses BenchmarkTools (@benchmark) instead of @elapsed. The old version
#     measured the FIRST threaded call, so worker_count=2 absorbed JIT
#     compilation of the @spawn path and reported a 0.26x "slowdown" that was
#     compilation, not scheduling.
#   * Reports median + IQR, not a mean of 5 noisy samples.
#   * Explicit warmup call per configuration.
#   * Plots both stages with an ideal-speedup reference line (CairoMakie, vector).
#   * Falls back to a reproducible surrogate corpus when the Tang/Bennett corpus
#     is unavailable (e.g. in CI), so scaling figures are reproducible without
#     redistributing restricted text.

using BenchmarkTools
using CairoMakie
using JSON3
using SparseArrays
using Statistics
using Printf

include(joinpath(@__DIR__, "kaq_pipeline.jl"))
using .KaqPipeline

CairoMakie.activate!(type = "svg")

# ---------------------------------------------------------------- arg parsing

function arg_or_default(flag::AbstractString, default)
    idx = findfirst(==(flag), ARGS)
    (idx === nothing || idx == length(ARGS)) && return default
    return ARGS[idx + 1]
end

has_flag(flag::AbstractString) = any(==(flag), ARGS)

# ------------------------------------------------------------ corpus loading

"""
Build the benchmark corpus.

Prefers the real manifest (Chronicle + Tang/Bennett). If the Tang/Bennett
corpus is not present on this machine, falls back to a surrogate corpus built
only from the committed Chronicle text, padded by deterministic resampling to
`target_n` rows. Scaling behaviour depends on corpus SIZE and token
distribution, not on the class labels, so the surrogate is a valid substrate
for timing experiments — but it is not valid for accuracy claims, and the
output JSON records which mode was used.
"""
function benchmark_corpus(; tang_path, manifest_path, target_n::Int = 6420)
    if isfile(tang_path)
        texts, labels, split = KaqPipeline.load_manifest_texts(
            tang_path = tang_path, manifest_path = manifest_path)
        train_idx = findall(==("train"), split)
        return texts[train_idx], labels[train_idx], "manifest"
    end

    @warn "Tang/Bennett corpus not found; using surrogate corpus from Chronicle text only." tang_path
    classical = KaqPipeline.load_kiwujil_sentences()
    isempty(classical) && error("No Chronicle sentences available for surrogate corpus.")

    texts  = Vector{String}(undef, target_n)
    labels = Vector{String}(undef, target_n)
    for i in 1:target_n
        texts[i]  = classical[mod1(i, length(classical))]
        labels[i] = iseven(i) ? "modern" : "classical"   # balanced, timing-only
    end
    return texts, labels, "surrogate"
end

# --------------------------------------------------------------- measurement

"""
Run `f` under BenchmarkTools and return a summary NamedTuple in seconds.
A warmup call is issued first so compilation is never inside the sample set.
"""
function measure(f::Function; samples::Int = 30, seconds::Float64 = 20.0)
    f()  # warmup: force compilation of this specialization
    b = @benchmarkable $f()
    trial = run(b; samples = samples, seconds = seconds, evals = 1)
    t = trial.times ./ 1e9      # ns -> s
    return (
        median = median(t),
        minimum = minimum(t),
        q25 = quantile(t, 0.25),
        q75 = quantile(t, 0.75),
        samples = length(t),
        allocs = trial.allocs,
        memory_bytes = trial.memory,
    )
end

summary_dict(s) = Dict(
    "median_seconds"  => s.median,
    "minimum_seconds" => s.minimum,
    "q25_seconds"     => s.q25,
    "q75_seconds"     => s.q75,
    "samples"         => s.samples,
    "allocations"     => s.allocs,
    "memory_bytes"    => s.memory_bytes,
)

# ------------------------------------------------------------------ plotting

function scaling_figure(worker_counts, feature_stats, train_stats, out_path)
    fig = Figure(size = (900, 380))

    for (col, (title, stats)) in enumerate((
            ("Char trigram TF-IDF extraction", feature_stats),
            ("Multinomial NB training",        train_stats)))

        ax = Axis(fig[1, col];
            title = title,
            xlabel = "Worker count",
            ylabel = col == 1 ? "Speedup vs serial" : "",
            xticks = (worker_counts, string.(worker_counts)))

        serial = stats[1].median
        med = [serial / stats[w].median for w in worker_counts]
        # Speedup bounds derive from the inverse of the timing quartiles.
        lo  = [serial / stats[w].q75 for w in worker_counts]
        hi  = [serial / stats[w].q25 for w in worker_counts]

        lines!(ax, worker_counts, float.(worker_counts);
            linestyle = :dash, color = (:gray, 0.7), label = "Ideal (linear)")
        band!(ax, worker_counts, lo, hi; color = (:steelblue, 0.25))
        lines!(ax, worker_counts, med; color = :steelblue, linewidth = 2.5)
        scatter!(ax, worker_counts, med; color = :steelblue, markersize = 11,
            label = "Measured (median, IQR)")

        hlines!(ax, [1.0]; color = (:black, 0.35), linewidth = 1)
        ylims!(ax, 0, max(maximum(worker_counts), maximum(hi)) * 1.1)
        col == 2 && axislegend(ax; position = :lt, framevisible = false)
    end

    Label(fig[0, :], "Thread scaling: median of BenchmarkTools samples, warmup excluded";
        fontsize = 13, padding = (0, 0, 4, 0))

    mkpath(dirname(out_path))
    save(out_path, fig)
    # Also emit PNG for Markdown/GitHub preview.
    png_path = replace(out_path, r"\.svg$" => ".png")
    save(png_path, fig; px_per_unit = 2)
    return out_path, png_path
end

# ---------------------------------------------------------------------- main

function main()
    tang_path     = arg_or_default("--tang-path", KaqPipeline.default_tang_path())
    manifest_path = arg_or_default("--manifest",  KaqPipeline.DEFAULT_MANIFEST_CSV)
    quick         = has_flag("--quick")
    samples       = quick ? 5 : 30
    seconds       = quick ? 5.0 : 20.0
    # Defaults write the committed paper figure and results JSON. CI passes a
    # separate directory so its surrogate --quick run never replaces them.
    output_dir    = arg_or_default("--output-dir", "")
    svg_path = isempty(output_dir) ?
        joinpath(@__DIR__, "..", "paper", "images", "thread_scaling.svg") :
        joinpath(abspath(output_dir), "thread_scaling.svg")
    out_json = isempty(output_dir) ?
        joinpath(@__DIR__, "results", "benchmark_results.json") :
        joinpath(abspath(output_dir), "benchmark_results.json")

    texts, labels, mode = benchmark_corpus(
        tang_path = tang_path, manifest_path = manifest_path)

    max_threads = Threads.nthreads()
    worker_counts = [w for w in (1, 2, 4, 8, 16) if w <= max_threads]
    isempty(worker_counts) && (worker_counts = [1])

    @info "Benchmark configuration" rows=length(texts) corpus_mode=mode threads=max_threads workers=worker_counts

    if max_threads == 1
        @warn "Julia started with 1 thread; scaling plot will be degenerate. Set JULIA_NUM_THREADS."
    end

    # --- stage 1: feature extraction
    feature_stats = Dict{Int, Any}()
    for w in worker_counts
        @info "Feature extraction" workers=w
        feature_stats[w] = w == 1 ?
            measure(() -> KaqPipeline.char_trigram_tfidf_matrix(texts; threaded = false);
                    samples = samples, seconds = seconds) :
            measure(() -> KaqPipeline.char_trigram_tfidf_matrix(texts; threaded = true, worker_count = w);
                    samples = samples, seconds = seconds)
    end

    # --- stage 2: NB training (features built once, outside the timed region)
    X, vocab = KaqPipeline.char_trigram_tfidf_matrix(texts; threaded = true, worker_count = max_threads)
    train_stats = Dict{Int, Any}()
    for w in worker_counts
        @info "NB training" workers=w
        train_stats[w] = w == 1 ?
            measure(() -> KaqPipeline.train_multinomial_nb(X, labels; threaded = false);
                    samples = samples, seconds = seconds) :
            measure(() -> KaqPipeline.train_multinomial_nb(X, labels; threaded = true, worker_count = w);
                    samples = samples, seconds = seconds)
    end

    # --- figure
    svg_out, png_out = scaling_figure(worker_counts, feature_stats, train_stats, svg_path)

    # --- results
    results = Dict(
        "corpus_mode"          => mode,
        "corpus_rows"          => length(texts),
        "threads_available"    => max_threads,
        "worker_counts"        => worker_counts,
        "feature_matrix_shape" => collect(size(X)),
        "feature_matrix_nnz"   => nnz(X),
        "vocabulary_size"      => length(vocab),
        "timing_method"        => "BenchmarkTools @benchmark, warmup excluded, median reported",
        "feature_extraction"   => Dict(string(w) => summary_dict(feature_stats[w]) for w in worker_counts),
        "naive_bayes_train"    => Dict(string(w) => summary_dict(train_stats[w]) for w in worker_counts),
        "feature_speedup_vs_serial" => Dict(string(w) =>
            feature_stats[1].median / feature_stats[w].median for w in worker_counts),
        "naive_bayes_speedup_vs_serial" => Dict(string(w) =>
            train_stats[1].median / train_stats[w].median for w in worker_counts),
        "scaling_plot_svg" => KaqPipeline.repo_relpath(svg_out),
        "scaling_plot_png" => KaqPipeline.repo_relpath(png_out),
    )

    mkpath(dirname(out_json))
    open(out_json, "w") do io
        JSON3.pretty(io, results)
    end

    println("\n", "="^62)
    @printf("%-10s %14s %14s %10s\n", "workers", "features (s)", "NB train (s)", "NB speedup")
    for w in worker_counts
        @printf("%-10d %14.5f %14.5f %10.2fx\n", w,
            feature_stats[w].median, train_stats[w].median,
            train_stats[1].median / train_stats[w].median)
    end
    println("="^62)
    println("corpus mode : ", mode)
    println("JSON        : ", out_json)
    println("figure      : ", svg_out)
end

main()
