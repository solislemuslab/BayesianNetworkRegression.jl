module BayesianNetworkRegression

using Random, DataFrames, LinearAlgebra, StatsBase, InvertedIndices, ProgressMeter, Distributions

include("utils.jl")
include("gibbs.jl")

export GenerateSamples!

end
