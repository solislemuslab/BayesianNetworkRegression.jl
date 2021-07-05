module BayesianNetworkRegression

using Core: Typeof
using Base: Float64
using Random, DataFrames, LinearAlgebra, StatsBase, InvertedIndices, ProgressMeter, Distributions

include("utils.jl")
include("gibbs.jl")

export GenerateSamples!

end
