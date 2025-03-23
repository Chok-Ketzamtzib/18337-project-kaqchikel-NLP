using DataFrames 
using CSV
using TextAnalysis
using Chain
using Statistics
using StatsBase
using SparseArrays

println("Loading Kaqchikel text data...")

# Use the Kiwujil.csv file that we have available
kiwujil_path = joinpath(@__DIR__, "datasets", "Kiwujil.csv")

# Load the CSV file
println("Reading CSV file: $kiwujil_path")
df_kiw = CSV.read(kiwujil_path, DataFrame)

# Display basic info about the dataframe
println("Dataset shape: ", size(df_kiw))
println("Column names: ", names(df_kiw))

# Get the text column (assuming it's the first column)
text_column = first(names(df_kiw))
println("Using text from column: $text_column")

# Basic text analysis
println("\nBasic text analysis:")
println("Number of entries: ", size(df_kiw, 1))

# Display some example entries - using first words only to avoid unicode issues
println("\nSample entries (first 10 words):")
for i in 1:min(3, size(df_kiw, 1))
    text = df_kiw[i, text_column]
    if typeof(text) == String
        words = split(text)
        sample = join(words[1:min(10, length(words))], " ")
        println("  Sample $i: $sample...")
    else
        println("  Sample $i: <non-string content>")
    end
end

# Convert text to StringDocuments for text analysis
println("\nConverting text to documents for analysis...")
docs = StringDocument[]
for text in df_kiw[:, text_column]
    if typeof(text) == String && !isempty(text)
        push!(docs, StringDocument(text))
    end
end

println("Created $(length(docs)) valid text documents")

# Document length statistics
doc_lengths = [length(doc.text) for doc in docs]
println("\nDocument length statistics:")
println("  Min length: $(minimum(doc_lengths)) characters")
println("  Max length: $(maximum(doc_lengths)) characters")
println("  Mean length: $(round(mean(doc_lengths), digits=2)) characters")
println("  Median length: $(median(doc_lengths)) characters")

# Simple text preprocessing
println("\nPerforming text preprocessing...")
for doc in docs
    prepare!(doc, strip_punctuation)
end

# Create corpus
println("\nCreating corpus...")
crps = Corpus(docs)
update_lexicon!(crps)

# Print lexicon statistics
println("\nLexicon statistics:")
println("Vocabulary size: ", length(lexicon(crps)))

# Count term frequencies manually by splitting text into words
println("Top 10 most frequent terms:")
term_counts = Dict{String, Int}()
for doc in docs
    words = split(doc.text)
    for word in words
        if !isempty(word)
            term_counts[word] = get(term_counts, word, 0) + 1
        end
    end
end

sorted_terms = sort(collect(term_counts), by=x->x[2], rev=true)
for (i, (term, count)) in enumerate(sorted_terms[1:min(10, length(sorted_terms))])
    println("  $i. $term: $count")
end

# TF-IDF Matrix Calculation
println("\nCalculating Document-Term Matrix...")
m = DocumentTermMatrix(crps)
println("DTM dimensions: $(size(m.dtm))")

println("\nCalculating TF-IDF Matrix...")
tfidf_mat = tf_idf(m)
println("TF-IDF matrix dimensions: $(size(tfidf_mat))")

# Find most important terms by TF-IDF scores
println("\nMost important terms by TF-IDF scores:")

# Calculate the sum of TF-IDF scores for each term
tfidf_sums = zeros(length(m.terms))
for j in 1:length(m.terms)
    for i in 1:size(tfidf_mat, 1)
        tfidf_sums[j] += tfidf_mat[i, j]
    end
end

# Create a dictionary of terms and their total TF-IDF scores
term_importance = Dict{String, Float64}()
for (i, term) in enumerate(m.terms)
    term_importance[term] = tfidf_sums[i]
end

# Sort and print the most important terms
sorted_importance = sort(collect(term_importance), by=x->x[2], rev=true)
for (i, (term, score)) in enumerate(sorted_importance[1:min(15, length(sorted_importance))])
    println("  $i. $term: $(round(score, digits=4))")
end

# Check for sentiment column (if it exists)
if "Sentiment" in names(df_kiw)
    # Check if there's any non-empty sentiment data
    local has_sentiment_data = false
    for s in df_kiw.Sentiment
        if !ismissing(s) && !isempty(s) && typeof(s) == String
            has_sentiment_data = true
            break
        end
    end
    
    if has_sentiment_data
        println("\nSentiment analysis:")
        sentiment_counts = Dict{String, Int}()
        for s in df_kiw.Sentiment
            if !ismissing(s) && !isempty(s) && typeof(s) == String
                sentiment_counts[s] = get(sentiment_counts, s, 0) + 1
            end
        end
        
        for (sentiment, count) in sort(collect(sentiment_counts), by=x->x[2], rev=true)
            println("  $sentiment: $count documents")
        end
    else
        println("\nNo sentiment data available for analysis")
    end
else
    println("\nNo sentiment column found in the dataset")
end

# More detailed term analysis - word length distribution
println("\nWord length distribution:")
word_lengths = [length(term) for term in keys(term_counts)]
length_counts = Dict{Int, Int}()
for len in word_lengths
    length_counts[len] = get(length_counts, len, 0) + 1
end

total_words = length(word_lengths)
for length_val in sort(collect(keys(length_counts)))
    count = length_counts[length_val]
    percentage = round(100 * count / total_words, digits=2)
    println("  $length_val-letter words: $count ($(percentage)%)")
end

println("\nAnalysis complete!") 