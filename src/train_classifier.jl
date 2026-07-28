using JSON3
using SparseArrays
using TextAnalysis

include(joinpath(@__DIR__, "kaq_pipeline.jl"))
using .KaqPipeline

function arg_or_default(flag::String, default::String)
    idx = findfirst(==(flag), ARGS)
    if idx === nothing || idx == length(ARGS)
        return default
    end
    return ARGS[idx + 1]
end

function _subset_vector(values::Vector{String}, indices::Vector{Int})
    out = Vector{String}(undef, length(indices))
    for (i, idx) in enumerate(indices)
        out[i] = values[idx]
    end
    return out
end

function _textanalysis_cross_check(train_texts, train_labels, test_texts, test_labels)
    # TextAnalysis NaiveBayesClassifier currently trains incrementally.
    try
        classes = unique(train_labels)
        model = NaiveBayesClassifier(classes)
        for (text, label) in zip(train_texts, train_labels)
            fit!(model, lowercase(text), label)
        end
        raw_preds = [predict(model, lowercase(text)) for text in test_texts]
        preds = [string(p) for p in raw_preds]
        acc = count(preds .== test_labels) / length(test_labels)
        status = acc == 0.0 ? "warning" : "ok"
        msg = acc == 0.0 ? "Classifier executed but output labels do not align with manifest labels." : "TextAnalysis NaiveBayesClassifier executed."
        return Dict(
            "status" => status,
            "accuracy" => acc,
            "message" => msg,
            "sample_prediction" => string(raw_preds[1]),
        )
    catch err
        return Dict(
            "status" => "unavailable",
            "message" => "TextAnalysis NaiveBayesClassifier check skipped due to API/runtime mismatch.",
            "error" => sprint(showerror, err),
        )
    end
end

function run_feature_experiment(feature_name::String, X::SparseMatrixCSC{Float64, Int}, terms, labels, train_idx, test_idx)
    X_train = X[train_idx, :]
    y_train = _subset_vector(labels, train_idx)
    X_test = X[test_idx, :]
    y_test = _subset_vector(labels, test_idx)

    serial_model = KaqPipeline.train_multinomial_nb(X_train, y_train; threaded=false)
    serial_pred = KaqPipeline.predict_multinomial_nb(serial_model, X_test)
    accuracy, confusion, per_class = KaqPipeline.classification_metrics(y_test, serial_pred, serial_model.classes)

    threaded_model = KaqPipeline.train_multinomial_nb(X_train, y_train; threaded=true, worker_count=Threads.nthreads())
    threaded_pred = KaqPipeline.predict_multinomial_nb(threaded_model, X_test)
    threaded_accuracy, _, _ = KaqPipeline.classification_metrics(y_test, threaded_pred, threaded_model.classes)

    return Dict(
        "feature_set" => feature_name,
        "test_accuracy_serial" => accuracy,
        "test_accuracy_threaded" => threaded_accuracy,
        "confusion_matrix" => confusion,
        "classes" => serial_model.classes,
        "per_class" => per_class,
        "top_discriminative_terms" => KaqPipeline.top_discriminative_terms(serial_model, terms; top_n=20),
    )
end

function main()
    tang_path = arg_or_default("--tang-path", KaqPipeline.DEFAULT_TANG_PATH)
    manifest_path = arg_or_default("--manifest", KaqPipeline.DEFAULT_MANIFEST_CSV)

    println("Loading manifest and corpus sources...")
    texts, labels, split = KaqPipeline.load_manifest_texts(tang_path=tang_path, manifest_path=manifest_path)
    println("Loaded $(length(texts)) samples from manifest.")

    train_idx = findall(==("train"), split)
    test_idx = findall(==("test"), split)
    println("Train rows: $(length(train_idx)) | Test rows: $(length(test_idx))")

    println("\nBuilding word unigram TF-IDF features...")
    word_X, word_terms = KaqPipeline.word_tfidf_matrix(texts)
    println("Word TF-IDF shape: $(size(word_X))")

    println("Building character trigram TF-IDF features...")
    char_X, char_terms = KaqPipeline.char_trigram_tfidf_matrix(texts; threaded=true, worker_count=Threads.nthreads())
    println("Char trigram TF-IDF shape: $(size(char_X))")

    println("\nTraining/evaluating Naive Bayes (word unigrams)...")
    word_results = run_feature_experiment("word_unigram", word_X, word_terms, labels, train_idx, test_idx)

    println("Training/evaluating Naive Bayes (char trigrams)...")
    char_results = run_feature_experiment("char_trigram", char_X, char_terms, labels, train_idx, test_idx)

    y_train = _subset_vector(texts, train_idx)
    l_train = _subset_vector(labels, train_idx)
    y_test = _subset_vector(texts, test_idx)
    l_test = _subset_vector(labels, test_idx)
    textanalysis_check = _textanalysis_cross_check(y_train, l_train, y_test, l_test)

    results = Dict(
        "manifest_path" => manifest_path,
        "sample_count" => length(texts),
        "train_count" => length(train_idx),
        "test_count" => length(test_idx),
        "threads_available" => Threads.nthreads(),
        "word_unigram" => word_results,
        "char_trigram" => char_results,
        "textanalysis_cross_check" => textanalysis_check,
    )

    println("\nWord unigram test accuracy (serial): $(round(word_results["test_accuracy_serial"], digits=4))")
    println("Word unigram test accuracy (threaded): $(round(word_results["test_accuracy_threaded"], digits=4))")
    println("Char trigram test accuracy (serial): $(round(char_results["test_accuracy_serial"], digits=4))")
    println("Char trigram test accuracy (threaded): $(round(char_results["test_accuracy_threaded"], digits=4))")

    out_path = joinpath(@__DIR__, "results", "train_metrics.json")
    mkpath(dirname(out_path))
    open(out_path, "w") do io
        JSON3.pretty(io, results)
    end
    println("\nSaved metrics to: $out_path")
end

main()
