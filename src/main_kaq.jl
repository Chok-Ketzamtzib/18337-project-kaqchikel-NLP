using DataFrames 
using CSV
using Gadfly
using TextAnalysis
using MLJ
using Chain
using Pipe
using StableRNGs
using DelimitedFiles
using Plots
using Statistics
using TextSearch
using Languages

````
Converting the text to CSV
````

path = joinpath((@__DIR__),"datasets","TangBennett_WrittenCorpus_Release_V01_April142023","April142023", "tang_bennett_2018_corpus_v01_14042023.txt")
path2 = joinpath((@__DIR__),"datasets","kiwujil xajila' final edit text ruk'isib'al q'ij 2022.txt")

# let's try converting to a FileDocument from TextAnalysis.jl package
fd = FileDocument(read(path, String))
sd = StringDocument(path)

# Now CSV/Matrix Stuff 
#kaq_matrix = readdlm(path, ' ', String, '\n')
#kiwujil_matrix = readdlm(path2, '.', String, '.')

kaq_csv = CSV.File(path; delim=' ', ignorerepeated=false, header=false)
kiwujil_csv = CSV.File(path2; delim='.', ignorerepeated=true, header=false, quoted=false)

CSV.write(joinpath((@__DIR__),"datasets","teng_kaq.csv"), kaq_csv)
CSV.write(joinpath((@__DIR__),"datasets","Kiwujil.csv"), kiwujil_csv)

# kiwujil_string = read(path2, String)
# file = CSV.File(IOBuffer(kiwujil_string))

# CSV.write(joinpath((@__DIR__),"datasets","Kiwujil_2.csv"), file)


# wordlengths = zeros(Int64,0)
# @time wordlengths = [length(x) for x in kaq_matrix];
# lrange = minimum(wordlengths),maximum(wordlengths)
# histogram(wordlengths,
#     bins=20,
#     xaxis=("WORD LENGTH"),
#     yaxis=("COUNT"),
#     xticks=([1:1:20;]),
#     yticks=([0:400;],["$(x)k" for x=0:5:390]),
#     label=("Word count"),
#     xguidefontsize=8, yguidefontsize=8,
#     margin=5mm,
#     framestyle = :box,
#     fill = (0,0.5,:green),
#     size=(800,420))
    
```` 
Simple Naive Bayes Implementation of Subset of Dataset
````
# split matrix into vectors

tokens = tokenize(Tokenizer(TextConfig()), "HI!! this is fun!! http://something")

x1 = kaq_matrix2[1, 1]
x2 = kaq_matrix2[2, 1]
x3 = kaq_matrix2[3, 1]
x4 = kaq_matrix2[4, 1]
y = kaq_matrix2[:, 5]

# identify unique elements

uniq_x1 = unique(x1)
uniq_x2 = unique(x2)
uniq_x3 = unique(x3)
uniq_x4 = unique(x4)

uniq_y = unique(y)


```` 
Convert into DataFrame
````
# df = DataFrame(kaq_matrix, :auto)
# df2 = DataFrame(kaq_matrix2, :auto)

df_kiw = kiwujil_csv |> DataFrame

```` 
Understand DataFrame
````
describe(df_kiw)
first(df_kiw, 1) |> pretty
@view df_kiw[1:5,1]

# Start cleaning dataframe for manual labeling
deleteat!(df_kiw, [1])

# converting the dataframe type

# Try stacking all columns into one columns
df_kiw = stack(df_kiw, 1:96)
rows_with_sentences = completecases(df_kiw) # see number of rows with sentences
df_kiw = dropmissing(df_kiw)
df_kiw = select!(df_kiw, Not(:variable))
df_kiw[!, :Sentiment] .= String
rename!(df_kiw,:value => :Sentence)
df_kiw = filter(:Sentence => n -> !(n == "  "), df_kiw ) # Remove blank rows
df_kiw = filter(:Sentence => n -> !(n == " "), df_kiw ) # remove blank rows

CSV.write(joinpath((@__DIR__),"datasets","Kiwujil.csv"), df_kiw)
df = @chain df begin
    DataFrames.transform(:1 => ByRow(x -> StringDocument(x)) => :Message)
end

@view df[1:5, 1628:1630]
# Text Preprocessing 

# remove_case!.(df[:, :Message])
prepare!.(df[:, :Message], strip_html_tags| strip_punctuation| strip_numbers)
stem!.(df[:, :Message])

# Preparing Corpus

crps = Corpus(df[:, :Message])

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