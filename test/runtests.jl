import InteractiveUtils: versioninfo
using BayesianNetworkRegression,Test,LinearAlgebra,Distributions
using CSV,DataFrames,StaticArrays,TypedTables,Random,Distributed

println(versioninfo())
@static if VERSION ≥ v"1.7.0-DEV.620"
    println(BLAS.get_config())
else
    @show BLAS.vendor()
    if startswith(string(BLAS.vendor()), "openblas")
        println(BLAS.openblas_get_config())
    end
end

@show BLAS.get_config()
@show Sys.isapple()

include("init-tests.jl")

