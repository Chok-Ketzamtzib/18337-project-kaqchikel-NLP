using DataFrames 
using CSV
using Gadfly
using TextAnalysis
using MLJ
using Chain
using Pipe
using StableRNGs

df = CSV.read(joinpath((@__DIR__),"datasets", "spam_dataset.csv"), DataFrames.DataFrame)
first(df, 10) |> pretty