module BayesianNetworkRegression
    using Random, DataFrames, LinearAlgebra, StatsBase, InvertedIndices, ProgressMeter, Distributions
    using StaticArrays, TypedTables, GaussianDistributions, Distributed, Statistics
    using Base.Threads,KahanSummation
    using Dates
    using MCMCDiagnosticTools

    include("gig.jl")
    include("utils.jl")
    include("convergence.jl")
    include("gibbs.jl")

    export Fit!,Summary
end

