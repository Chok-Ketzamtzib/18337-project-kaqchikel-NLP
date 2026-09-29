using JSON3

include(joinpath(@__DIR__, "kaq_pipeline.jl"))
using .KaqPipeline

function arg_or_default(flag::String, default::String)
    idx = findfirst(==(flag), ARGS)
    if idx === nothing || idx == length(ARGS)
        return default
    end
    return ARGS[idx + 1]
end

function int_arg_or_default(flag::String, default::Int)
    idx = findfirst(==(flag), ARGS)
    if idx === nothing || idx == length(ARGS)
        return default
    end
    return parse(Int, ARGS[idx + 1])
end

function float_arg_or_default(flag::String, default::Float64)
    idx = findfirst(==(flag), ARGS)
    if idx === nothing || idx == length(ARGS)
        return default
    end
    return parse(Float64, ARGS[idx + 1])
end

function main()
    tang_path = arg_or_default("--tang-path", KaqPipeline.default_tang_path())
    seed = int_arg_or_default("--seed", 42)
    min_tokens = int_arg_or_default("--min-tokens", 3)
    test_fraction = float_arg_or_default("--test-fraction", 0.2)

    println("Preparing Kaqchikel register dataset...")
    println("  Tang/Bennett path: $tang_path")
    println("  Seed: $seed")
    println("  Min tokens: $min_tokens")
    println("  Test fraction: $test_fraction")

    stats = KaqPipeline.prepare_datasets(
        tang_path=tang_path,
        seed=seed,
        min_tokens=min_tokens,
        test_fraction=test_fraction,
    )

    println("\nData preparation complete.")
    println("  Kiwujil rows regenerated: $(stats.kiwujil_rows)")
    println("  Classical rows available: $(stats.classical_rows)")
    println("  Modern rows available: $(stats.modern_rows)")
    println("  Balanced rows used: $(stats.balanced_rows)")
    println("  Train rows: $(stats.train_rows)")
    println("  Test rows: $(stats.test_rows)")
    println("  Manifest: $(stats.manifest_path)")

    summary = Dict{String, Any}(string(k) => v for (k, v) in pairs(stats))
    summary["manifest_path"] = KaqPipeline.repo_relpath(stats.manifest_path)
    summary["kiwujil_csv_path"] = KaqPipeline.repo_relpath(stats.kiwujil_csv_path)

    summary_path = joinpath(@__DIR__, "results", "data_prep_summary.json")
    mkpath(dirname(summary_path))
    open(summary_path, "w") do io
        JSON3.pretty(io, summary)
    end
    println("  Summary JSON: $summary_path")
end

main()
