module BayesianNetworkRegression

using LinearAlgebra: iterate
using DataFrames: Vector
using Core: Typeof
using Base: Float64
using Random, DataFrames, LinearAlgebra, StatsBase, InvertedIndices, ProgressMeter, Distributions
using StaticArrays,TypedTables,GaussianDistributions

include("gig.jl")
include("utils.jl")
include("gibbs.jl")

export GenerateSamples!

end
