using DataFrames 
using CSV
using Gadfly
using TextAnalysis
using MLJ
using Chain
using Pipe
using StableRNGs

# importing the dataset

df = CSV.read(joinpath((@__DIR__),"datasets", "spam_dataset.csv"), DataFrames.DataFrame)
first(df, 10) |> pretty

# converting the dataframe type

df = @chain df begin
    DataFrames.transform(:Message => ByRow(x -> StringDocument(x)) => :Message2)
end

# Text Preprocessing 

remove_case!.(df[:, :Message2])
prepare!.(df[:, :Message2], strip_html_tags| strip_punctuation| strip_numbers)
stem!.(df[:, :Message2])

# Preparing Corpus

crps = Corpus(df[:, :Message2])

# Build the vocabulary

update_lexicon!(crps)

# Dense 

dense_dtm = dtm(m, :dense)

# Build the tf-idf matrix

m = DocumentTermMatrix(crps)
tfidf_mat = tf_idf(m)

# Supervised Machine Learning Portion

X, y = tfidf_mat, df[:, :Category]

# Data Modeling

DecisionTreeClassifier = @load DecisionTreeClassifier pkg=DecisionTree
tree_model = DecisionTreeClassifier()

# model
tree = machine(tree_model, coerce(X, Continuous), coerce(y, Multiclass))

# partitioning the dataframe
rng = StableRNG(42)
train, test = partition(eachindex(y), 0.85, shuffle=true, rng=rng);

# fit
MLJ.fit!(tree, rows=train)