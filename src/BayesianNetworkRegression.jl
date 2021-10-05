module BayesianNetworkRegression
    using Random, DataFrames, LinearAlgebra, StatsBase, InvertedIndices, ProgressMeter, Distributions
    using StaticArrays, TypedTables, GaussianDistributions, MCMCChains, Distributed
    using Base.Threads

    include("gig.jl")
    include("utils.jl")
    include("gibbs.jl")

    export GenerateSamples!
end

