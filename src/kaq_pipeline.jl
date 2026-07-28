module KaqPipeline

using CSV
using DataFrames
using LinearAlgebra
using Random
using SparseArrays
using StatsBase
using Statistics
using TextAnalysis
using Base.Threads

const DEFAULT_KIWUJIL_TXT = joinpath(@__DIR__, "datasets", "kiwujil xajila' final edit text ruk'isib'al q'ij 2022.txt")
const DEFAULT_KIWUJIL_CSV = joinpath(@__DIR__, "datasets", "Kiwujil.csv")
const DEFAULT_MANIFEST_CSV = joinpath(@__DIR__, "datasets", "register_manifest.csv")
const DEFAULT_TANG_PATH = raw"C:\Users\wakef\Documents\Mayanist\TangBennett_WrittenCorpus_Release_V01_April142023\April142023\tang_bennett_2018_corpus_v01_14042023.txt"

function read_text_with_fallback(path::AbstractString)::String
    raw = read(path)
    try
        return String(raw)
    catch
        # Fallback for legacy corpora that were stored in Latin-1/Windows-1252.
        return String(Char.(raw))
    end
end

function normalize_space(text::AbstractString)::String
    clean = replace(text, '\ufeff' => "")
    clean = strip(replace(clean, r"\s+" => " "))
    return clean
end

function split_into_sentences(text::AbstractString; min_tokens::Int=3)::Vector{String}
    chunks = split(text, r"[.!?\n]+")
    sentences = String[]
    for chunk in chunks
        candidate = normalize_space(chunk)
        if isempty(candidate)
            continue
        end
        if startswith(lowercase(candidate), "kiwujil xajila")
            continue
        end
        if length(split(candidate)) < min_tokens
            continue
        end
        push!(sentences, candidate)
    end
    return sentences
end

function regenerate_kiwujil_csv(
    kiwujil_txt_path::AbstractString=DEFAULT_KIWUJIL_TXT;
    output_csv_path::AbstractString=DEFAULT_KIWUJIL_CSV,
    min_tokens::Int=3,
)::DataFrame
    raw_text = read_text_with_fallback(kiwujil_txt_path)
    sentences = split_into_sentences(raw_text; min_tokens=min_tokens)
    df = DataFrame(Sentence=sentences)
    CSV.write(output_csv_path, df)
    return df
end

function load_kiwujil_sentences(kiwujil_csv_path::AbstractString=DEFAULT_KIWUJIL_CSV)::Vector{String}
    df = CSV.read(kiwujil_csv_path, DataFrame)
    if !("Sentence" in names(df))
        error("Expected a 'Sentence' column in $kiwujil_csv_path")
    end
    sentences = String[]
    for sentence in df.Sentence
        sentence isa String || continue
        clean = normalize_space(sentence)
        isempty(clean) && continue
        push!(sentences, clean)
    end
    return sentences
end

function load_tang_sentences(tang_path::AbstractString=DEFAULT_TANG_PATH; min_tokens::Int=3)::Vector{String}
    raw_text = read_text_with_fallback(tang_path)
    lines = split(raw_text, '\n')
    sentences = String[]
    for line in lines
        clean = normalize_space(line)
        isempty(clean) && continue
        if length(split(clean)) < min_tokens
            continue
        end
        push!(sentences, clean)
    end
    return sentences
end

function _stratified_split_indices(n_per_class::Int; seed::Int=42, test_fraction::Float64=0.2)
    rng = MersenneTwister(seed)
    all_idx = collect(1:n_per_class)
    shuffle!(rng, all_idx)
    n_test = max(1, round(Int, n_per_class * test_fraction))
    test_idx = Set(all_idx[1:n_test])
    train_idx = Set(all_idx[(n_test + 1):end])
    return train_idx, test_idx
end

function prepare_datasets(;
    kiwujil_txt_path::AbstractString=DEFAULT_KIWUJIL_TXT,
    kiwujil_csv_path::AbstractString=DEFAULT_KIWUJIL_CSV,
    tang_path::AbstractString=DEFAULT_TANG_PATH,
    manifest_path::AbstractString=DEFAULT_MANIFEST_CSV,
    min_tokens::Int=3,
    seed::Int=42,
    test_fraction::Float64=0.2,
)
    kiwujil_df = regenerate_kiwujil_csv(kiwujil_txt_path; output_csv_path=kiwujil_csv_path, min_tokens=min_tokens)
    classical_sentences = load_kiwujil_sentences(kiwujil_csv_path)
    modern_sentences = load_tang_sentences(tang_path; min_tokens=min_tokens)

    n_classical = length(classical_sentences)
    n_modern = length(modern_sentences)
    n_balanced = min(n_classical, n_modern)
    if n_balanced == 0
        error("No usable sentences found after filtering.")
    end

    rng = MersenneTwister(seed)
    classical_idx = sort(sample(rng, 1:n_classical, n_balanced; replace=false))
    modern_idx = sort(sample(rng, 1:n_modern, n_balanced; replace=false))
    train_local, test_local = _stratified_split_indices(n_balanced; seed=seed, test_fraction=test_fraction)

    rows = NamedTuple[]
    for i in 1:n_balanced
        split = i in test_local ? "test" : "train"
        push!(rows, (label="classical", source="kiwujil", source_index=classical_idx[i], split=split))
        push!(rows, (label="modern", source="tang_bennett", source_index=modern_idx[i], split=split))
    end

    manifest = DataFrame(rows)
    CSV.write(manifest_path, manifest)

    return (
        kiwujil_rows=size(kiwujil_df, 1),
        classical_rows=n_classical,
        modern_rows=n_modern,
        balanced_rows=n_balanced * 2,
        train_rows=count(==("train"), manifest.split),
        test_rows=count(==("test"), manifest.split),
        manifest_path=manifest_path,
        kiwujil_csv_path=kiwujil_csv_path,
    )
end

function load_manifest_texts(;
    manifest_path::AbstractString=DEFAULT_MANIFEST_CSV,
    kiwujil_csv_path::AbstractString=DEFAULT_KIWUJIL_CSV,
    tang_path::AbstractString=DEFAULT_TANG_PATH,
)
    manifest = CSV.read(manifest_path, DataFrame)
    classical = load_kiwujil_sentences(kiwujil_csv_path)
    modern = load_tang_sentences(tang_path)

    texts = Vector{String}(undef, size(manifest, 1))
    labels = Vector{String}(undef, size(manifest, 1))
    split = Vector{String}(undef, size(manifest, 1))

    for i in 1:size(manifest, 1)
        src = manifest.source[i]
        idx = Int(manifest.source_index[i])
        if src == "kiwujil"
            texts[i] = classical[idx]
        elseif src == "tang_bennett"
            texts[i] = modern[idx]
        else
            error("Unknown source '$src' in manifest.")
        end
        labels[i] = String(manifest.label[i])
        split[i] = String(manifest.split[i])
    end

    return texts, labels, split
end

function word_tfidf_matrix(texts::Vector{String})
    docs = StringDocument[]
    for text in texts
        doc = StringDocument(lowercase(text))
        prepare!(doc, strip_punctuation)
        push!(docs, doc)
    end
    crps = Corpus(docs)
    update_lexicon!(crps)
    dtm = DocumentTermMatrix(crps)
    tfidf = tf_idf(dtm)
    return sparse(tfidf), dtm.terms
end

function _line_char_trigrams(text::String)
    normalized = lowercase(normalize_space(text))
    padded = "^" * normalized * "\$"
    trigrams = String[]
    chars = collect(padded)
    if length(chars) < 3
        return trigrams
    end
    for i in 1:(length(chars) - 2)
        push!(trigrams, String(chars[i:(i + 2)]))
    end
    return trigrams
end

function _count_trigrams_chunk(texts::Vector{String}, idxs::UnitRange{Int})
    local_docs = Vector{Dict{String, Int}}(undef, length(idxs))
    local_df = Dict{String, Int}()
    for (offset, i) in enumerate(idxs)
        doc_counts = Dict{String, Int}()
        for tri in _line_char_trigrams(texts[i])
            doc_counts[tri] = get(doc_counts, tri, 0) + 1
        end
        for tri in keys(doc_counts)
            local_df[tri] = get(local_df, tri, 0) + 1
        end
        local_docs[offset] = doc_counts
    end
    return local_docs, local_df
end

function _chunk_ranges(n::Int, chunks::Int)
    chunk_size = max(1, ceil(Int, n / chunks))
    ranges = UnitRange{Int}[]
    start_idx = 1
    while start_idx <= n
        stop_idx = min(n, start_idx + chunk_size - 1)
        push!(ranges, start_idx:stop_idx)
        start_idx = stop_idx + 1
    end
    return ranges
end

function char_trigram_tfidf_matrix(texts::Vector{String}; threaded::Bool=false, worker_count::Int=nthreads())
    n_docs = length(texts)
    n_docs == 0 && error("Cannot build features for zero texts.")

    doc_counts = Vector{Dict{String, Int}}(undef, n_docs)
    df_counts = Dict{String, Int}()

    if threaded && worker_count > 1
        ranges = _chunk_ranges(n_docs, min(worker_count, n_docs))
        tasks = map(ranges) do chunk
            @spawn _count_trigrams_chunk(texts, chunk)
        end
        for (chunk, task) in zip(ranges, tasks)
            local_docs, local_df = fetch(task)
            for (offset, i) in enumerate(chunk)
                doc_counts[i] = local_docs[offset]
            end
            for (tri, cnt) in local_df
                df_counts[tri] = get(df_counts, tri, 0) + cnt
            end
        end
    else
        for i in 1:n_docs
            counts = Dict{String, Int}()
            for tri in _line_char_trigrams(texts[i])
                counts[tri] = get(counts, tri, 0) + 1
            end
            for tri in keys(counts)
                df_counts[tri] = get(df_counts, tri, 0) + 1
            end
            doc_counts[i] = counts
        end
    end

    vocab = sort!(collect(keys(df_counts)))
    vocab_index = Dict(term => idx for (idx, term) in enumerate(vocab))

    row_idx = Int[]
    col_idx = Int[]
    vals = Float64[]
    for i in 1:n_docs
        for (tri, count) in doc_counts[i]
            push!(row_idx, i)
            push!(col_idx, vocab_index[tri])
            push!(vals, float(count))
        end
    end

    tf = sparse(row_idx, col_idx, vals, n_docs, length(vocab))
    idf = zeros(Float64, length(vocab))
    for (j, term) in enumerate(vocab)
        idf[j] = log((n_docs + 1) / (df_counts[term] + 1)) + 1.0
    end
    tfidf = tf * Diagonal(idf)
    return tfidf, vocab
end

function train_multinomial_nb(
    X::SparseMatrixCSC{Float64, Int},
    y::Vector{String};
    alpha::Float64=1.0,
    threaded::Bool=false,
    worker_count::Int=nthreads(),
)
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
    Xt = sparse(transpose(X))

    if threaded && worker_count > 1
        ranges = _chunk_ranges(n_docs, min(worker_count, n_docs))
        tasks = map(ranges) do chunk
            @spawn begin
                local_sum = zeros(Float64, n_classes, n_features)
                for doc_idx in chunk
                    c = y_idx[doc_idx]
                    for ptr in nzrange(Xt, doc_idx)
                        feature = Xt.rowval[ptr]
                        value = Xt.nzval[ptr]
                        local_sum[c, feature] += value
                    end
                end
                local_sum
            end
        end
        for task in tasks
            feature_sums .+= fetch(task)
        end
    else
        for doc_idx in 1:n_docs
            c = y_idx[doc_idx]
            for ptr in nzrange(Xt, doc_idx)
                feature = Xt.rowval[ptr]
                value = Xt.nzval[ptr]
                feature_sums[c, feature] += value
            end
        end
    end

    log_likelihood = zeros(Float64, n_classes, n_features)
    for c in 1:n_classes
        denom = sum(view(feature_sums, c, :)) + alpha * n_features
        for f in 1:n_features
            log_likelihood[c, f] = log((feature_sums[c, f] + alpha) / denom)
        end
    end

    return (classes=classes, class_to_idx=class_to_idx, log_priors=log_priors, log_likelihood=log_likelihood)
end

function predict_multinomial_nb(model, X::SparseMatrixCSC{Float64, Int})
    n_docs, _ = size(X)
    n_classes = length(model.classes)
    preds = Vector{String}(undef, n_docs)
    Xt = sparse(transpose(X))
    for doc_idx in 1:n_docs
        scores = copy(model.log_priors)
        for ptr in nzrange(Xt, doc_idx)
            feature = Xt.rowval[ptr]
            value = Xt.nzval[ptr]
            for c in 1:n_classes
                scores[c] += value * model.log_likelihood[c, feature]
            end
        end
        preds[doc_idx] = model.classes[argmax(scores)]
    end
    return preds
end

function confusion_counts(y_true::Vector{String}, y_pred::Vector{String}, classes::Vector{String})
    class_to_idx = Dict(c => i for (i, c) in enumerate(classes))
    matrix = zeros(Int, length(classes), length(classes))
    for (truth, pred) in zip(y_true, y_pred)
        matrix[class_to_idx[truth], class_to_idx[pred]] += 1
    end
    return matrix
end

function classification_metrics(y_true::Vector{String}, y_pred::Vector{String}, classes::Vector{String})
    total = length(y_true)
    correct = count(y_true .== y_pred)
    accuracy = correct / total
    conf = confusion_counts(y_true, y_pred, classes)

    per_class = Dict{String, Dict{String, Float64}}()
    for (i, class_name) in enumerate(classes)
        tp = conf[i, i]
        fp = sum(conf[:, i]) - tp
        fn = sum(conf[i, :]) - tp
        precision = tp == 0 ? 0.0 : tp / (tp + fp)
        recall = tp == 0 ? 0.0 : tp / (tp + fn)
        per_class[class_name] = Dict(
            "precision" => precision,
            "recall" => recall,
            "support" => sum(conf[i, :]),
        )
    end
    return accuracy, conf, per_class
end

function top_discriminative_terms(model, terms::Vector{String}; top_n::Int=20)
    if length(model.classes) != 2
        return Dict("note" => "Top terms currently implemented for binary classification only.")
    end
    delta = model.log_likelihood[1, :] .- model.log_likelihood[2, :]
    order = sortperm(abs.(delta); rev=true)
    top = Dict{String, Dict{String, Any}}()
    for idx in order[1:min(top_n, length(order))]
        favored = delta[idx] > 0 ? model.classes[1] : model.classes[2]
        top[terms[idx]] = Dict("favor_class" => favored, "log_odds_delta" => delta[idx])
    end
    return top
end

export DEFAULT_KIWUJIL_TXT
export DEFAULT_KIWUJIL_CSV
export DEFAULT_MANIFEST_CSV
export DEFAULT_TANG_PATH
export prepare_datasets
export load_manifest_texts
export word_tfidf_matrix
export char_trigram_tfidf_matrix
export train_multinomial_nb
export predict_multinomial_nb
export classification_metrics
export top_discriminative_terms

end
