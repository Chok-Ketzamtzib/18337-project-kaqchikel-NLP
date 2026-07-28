using JSON3
using Plots
using Statistics

include(joinpath(@__DIR__, "kaq_pipeline.jl"))
using .KaqPipeline

function arg_or_default(flag::String, default::String)
    idx = findfirst(==(flag), ARGS)
    if idx === nothing || idx == length(ARGS)
        return default
    end
    return ARGS[idx + 1]
end

function benchmark_seconds(f::Function; repeats::Int=5)
    durations = Float64[]
    for _ in 1:repeats
        elapsed = @elapsed f()
        push!(durations, elapsed)
    end
    return mean(durations)
end

function main()
    tang_path = arg_or_default("--tang-path", KaqPipeline.DEFAULT_TANG_PATH)
    manifest_path = arg_or_default("--manifest", KaqPipeline.DEFAULT_MANIFEST_CSV)

    texts, labels, split = KaqPipeline.load_manifest_texts(tang_path=tang_path, manifest_path=manifest_path)
    train_idx = findall(==("train"), split)
    train_texts = texts[train_idx]
    train_labels = labels[train_idx]

    println("Running benchmarks on $(length(train_texts)) training rows.")
    max_threads = Threads.nthreads()
    worker_counts = [1, 2, 4, 8]
    worker_counts = [w for w in worker_counts if w <= max_threads]
    if isempty(worker_counts)
        worker_counts = [1]
    end

    println("Available threads: $max_threads")
    println("Worker counts benchmarked: $(worker_counts)")

    serial_feature_time = benchmark_seconds(() -> KaqPipeline.char_trigram_tfidf_matrix(train_texts; threaded=false))

    feature_times = Dict{String, Float64}()
    feature_times["1"] = serial_feature_time
    for w in worker_counts
        if w == 1
            continue
        end
        t = benchmark_seconds(() -> KaqPipeline.char_trigram_tfidf_matrix(train_texts; threaded=true, worker_count=w))
        feature_times[string(w)] = t
    end

    X_train, _ = KaqPipeline.char_trigram_tfidf_matrix(train_texts; threaded=true, worker_count=max_threads)
    serial_train_time = benchmark_seconds(() -> KaqPipeline.train_multinomial_nb(X_train, train_labels; threaded=false))

    train_times = Dict{String, Float64}()
    train_times["1"] = serial_train_time
    for w in worker_counts
        if w == 1
            continue
        end
        t = benchmark_seconds(() -> KaqPipeline.train_multinomial_nb(X_train, train_labels; threaded=true, worker_count=w))
        train_times[string(w)] = t
    end

    xs = sort(parse.(Int, collect(keys(train_times))))
    ys = [serial_train_time / train_times[string(x)] for x in xs]

    mkpath(joinpath(@__DIR__, "..", "paper", "images"))
    out_plot = joinpath(@__DIR__, "..", "paper", "images", "thread_scaling.png")
    plot(
        xs,
        ys;
        marker=:circle,
        linewidth=2,
        title="Naive Bayes Training Speedup vs Serial",
        xlabel="Worker Count",
        ylabel="Speedup",
        legend=false,
        xticks=xs,
    )
    savefig(out_plot)

    results = Dict(
        "threads_available" => max_threads,
        "worker_counts" => xs,
        "feature_extraction_seconds" => feature_times,
        "naive_bayes_train_seconds" => train_times,
        "naive_bayes_speedup_vs_serial" => Dict(string(x) => ys[i] for (i, x) in enumerate(xs)),
        "scaling_plot" => out_plot,
    )

    out_json = joinpath(@__DIR__, "results", "benchmark_results.json")
    mkpath(dirname(out_json))
    open(out_json, "w") do io
        JSON3.pretty(io, results)
    end

    println("Feature extraction (serial): $(round(serial_feature_time, digits=5))s")
    println("Naive Bayes train (serial): $(round(serial_train_time, digits=5))s")
    println("Saved benchmark JSON: $out_json")
    println("Saved scaling plot: $out_plot")
end

main()
