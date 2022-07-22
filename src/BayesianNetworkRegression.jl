module BayesianNetworkRegression
#=    # Don't use MKL on mac for now, perhaps related to https://github.com/JuliaLinearAlgebra/MKL.jl/issues/112
    if (occursin("Intel",Sys.cpu_info()[1].model))# && !Sys.isapple()) 
        using MKL 
    end    
=#
    using Random, DataFrames, LinearAlgebra, StatsBase, InvertedIndices, ProgressMeter, Distributions
    using StaticArrays, TypedTables, GaussianDistributions, MCMCDiagnosticTools, Distributed, Statistics
    using Base.Threads,KahanSummation

    include("gig.jl")
    include("utils.jl")
    include("convergence.jl")
    include("gibbs.jl")

    export Fit!
end

