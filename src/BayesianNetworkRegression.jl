module BayesianNetworkRegression
    using Random, DataFrames, LinearAlgebra, StatsBase, InvertedIndices, ProgressMeter, Distributions
    using StaticArrays, TypedTables, GaussianDistributions, MCMCDiagnosticTools, Distributed, Statistics
    using Base.Threads
    import KahanSummation

    include("gig.jl")
    include("utils.jl")
    include("convergence.jl")
    include("gibbs.jl")

    export GenerateSamples!
end

