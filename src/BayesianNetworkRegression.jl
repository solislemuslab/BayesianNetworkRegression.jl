module BayesianNetworkRegression
    # module BayesianNetworkRegressionHelper

    # using Random, DataFrames, LinearAlgebra, StatsBase, InvertedIndices, ProgressMeter, Distributions
    # using StaticArrays, TypedTables, GaussianDistributions, MCMCChains
    # #using FLoops
    # #using Polyester
    # using Base.Threads

    # include("gig.jl")
    # include("utils.jl")
    # include("gibbs.jl")


    # end

    # #module BNR
    # using Distributed

    # function include_everywhere(filepath)
    #     include(filepath) # Load on Node 1 first, triggering any precompile
    #     if nprocs() > 1
    #         fullpath = joinpath(@__DIR__, filepath)
    #         @sync for p in workers()
    #             @async remotecall_wait(include, p, fullpath)
    #         end
    #     end
    # end

    # #@everywhere using .BayesianNetworkRegressionHelper
    # #include_everywhere("imports.jl")

    # include("imports.jl")
    # include("main.jl")

    # export GenerateSamples!
    # #end
    using Random, DataFrames, LinearAlgebra, StatsBase, InvertedIndices, ProgressMeter, Distributions
    using StaticArrays, TypedTables, GaussianDistributions, MCMCChains, Distributed
    #using FLoops
    #using Polyester
    using Base.Threads

    include("gig.jl")
    include("utils.jl")
    include("gibbs.jl")
    #include("main.jl")

    export GenerateSamples!
end

