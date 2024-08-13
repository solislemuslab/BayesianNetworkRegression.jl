
### This file generates the "true" results used in init-tests.jl, toy-generate-samples-test.jl, and test1-generate-samples-test.jl

import InteractiveUtils: versioninfo
using BayesianNetworkRegression,Test,LinearAlgebra,Distributions
using CSV,DataFrames,StaticArrays,TypedTables,Random,Distributed,JLD2

## Initial tests for BayesianNetworkRegression.jl on toy data
println("----------------- INIT TESTS ------------------")

## global seed
rng = Xoshiro(1234)

X = [[0, 1, 0, 1,
     1, 0, 1, 1,
     0, 1, 0, 0,
     0, 1, 0, 0],
    [0, 1, 1, 1,
     1, 0, 1, 1,
     1, 1, 0, 0,
     1, 1, 0, 0],
    [0, 0, 1, 0,
     0, 0, 1, 1,
     1, 1, 0, 0,
     0, 1, 0, 0],
    [0, 0, 1, 0,
     0, 0, 1, 1,
     1, 1, 0, 1,
     0, 1, 1, 0]]

Z = BayesianNetworkRegression.symmetrize_matrices(X)

y = ones(size(Z,1))*12 + rand(rng, Normal(0,2),size(Z,1))

η  = 1.01
ζ  = 1.0
ι  = 1.0
R  = 7
aΔ = 1.0
bΔ = 1.0
V = size(Z,1)
q = floor(Int,V*(V+1)/2)
n = size(Z,1)
ν = 10
total = 20

st1 = Table(τ² = Array{Float64,3}(undef,(total,1,1)), u = Array{Float64,3}(undef,(total,R,V)),
                  ξ = Array{Float64,3}(undef,(total,V,1)), γ = Array{Float64,3}(undef,(total,q,1)),
                  S = Array{Float64,3}(undef,(total,q,1)), θ = Array{Float64,3}(undef,(total,1,1)),
                  Δ = Array{Float64,3}(undef,(total,1,1)), M = Array{Float64,3}(undef,(total,R,R)),
                  μ = Array{Float64,3}(undef,(total,1,1)), λ = Array{Float64,3}(undef,(total,R,1)),
                  πᵥ= Array{Float64,3}(undef,(total,R,3)));

X_new = Array{Float64,2}(undef,n,q)
##X_new = rand(rng, Normal(0,2),n,q) ## test added thinking undef was causing float issues, but no

println("InitTests - Dimensions and initializations") 
begin
    tmprng = Xoshiro(100)
    BayesianNetworkRegression.initialize_variables!(st1, η, R, ν,tmprng, V)

    @show st1.S[1,:,1]
    @show st1.πᵥ[1,:,:]
    @show st1.λ[1,:,1] 
    @show st1.ξ[1,:,1] 
    @show st1.u[1,:,:] 
    @show st1.M[1,:,:]
    @show st1.γ[1,:,1]
end


println("InitTests - Gibbs sampler") 
begin
    tmprng = Xoshiro(100)
    st2 = deepcopy(st1)
    BayesianNetworkRegression.gibbs_sample!(st2, 2, X_new, y, V, η, ζ, ι, R, aΔ, bΔ, ν, tmprng)

    @show st2.τ²[2,1,1]
    @show st2.ξ[2,:,1]
    @show st2.u[2,:,:]
    @show st2.γ[2,:,1]
    @show st2.θ[2,1,1]
    @show st2.Δ[2,1,1]
    @show st2.M[2,:,:] 
    @show st2.μ[2,1,1] 
    @show st2.S[2,:,1]
    @show st2.πᵥ[2,:,:]
    @show st2.λ[2,:,1]
end

println("InitTests - Deconstructed Gibbs sampler")
begin
    n = size(X_new,1)
    rng = Xoshiro(123)
    st3 = deepcopy(st2)

    BayesianNetworkRegression.update_τ²!(st3, 3, X_new, y, V, rng)
    @show st3.τ²[3,1,1]

    BayesianNetworkRegression.update_u_ξ!(st3, 3, V, rng)
    @show st3.ξ[3,:,1]
    @show st3.u[3,:,:]

    BayesianNetworkRegression.update_γ!(st3, 3, X_new, y, n, rng)
    @show st3.γ[3,:,1]

    BayesianNetworkRegression.update_D!(st3, 3, V, rng)
    BayesianNetworkRegression.update_θ!(st3, 3, ζ, ι, V, rng)
    @show st3.θ[3,1,1]

    BayesianNetworkRegression.update_Δ!(st3, 3, aΔ, bΔ, rng)
    @show st3.Δ[3,1,1]

    BayesianNetworkRegression.update_M!(st3, 3, ν, V, rng)
    @show st3.M[3,:,:] 

    BayesianNetworkRegression.update_μ!(st3, 3, X_new, y, n, rng)
    @show st3.μ[3,1,1] 
    
    BayesianNetworkRegression.update_Λ!(st3, 3, R, rng)
    @show st3.λ[3,:,1]

    BayesianNetworkRegression.update_π!(st3, 3, η, R, rng)
    @show st3.πᵥ[3,:,:]
    @show st3.S[3,:,1] 
end


println("--------TOY DATA--------------")
# Test of fit! in toy data

## global seed
rng = Xoshiro(1234)

## simulating 10 4x4 adjacency matrices
X = [rand(rng,Bernoulli(0.5),4,4)]
for i in 1:9
     push!(X,rand(rng,Bernoulli(0.5),4,4))
end

y = ones(size(X,1))*12 + rand(rng, Normal(0,2),size(X,1))

R  = 5

res = BayesianNetworkRegression.generate_samples!(X, y, R, η=1.01,ζ=1.0,ι=1.0,aΔ=1.0,bΔ=1.0,
    ν=10, nburn=300, nsamp=100, maxburn=300, x_transform=true, psrf_cutoff=1.2,
    suppress_timer=false, num_chains=1, seed=1234, purge_burn=nothing)

println("Toy Generate Samples - Parameters") 
begin
     @show res.state.τ²[1:10] 
     @show mean(res.state.τ²)
     @show res.state.u[1:10,1:5,1] 
     @show mean(res.state.u)
     @show res.state.ξ[1:10,:,1] 
     @show mean(res.state.ξ)
     @show res.state.γ[1:10,:,1]
     @show mean(res.state.γ)
     @show res.state.S[1:10,:,1] 
     @show mean(res.state.S)
     @show res.state.θ[1:10] 
     @show mean(res.state.θ) 
     @show res.state.Δ[1:10,:,1]
     @show mean(res.state.Δ) 
     @show res.state.M[1:10,:,1] 
     @show mean(res.state.M) 
     @show res.state.μ[1:10] 
     @show mean(res.state.μ) 
     @show res.state.πᵥ[1:10,:,1] 
     @show mean(res.state.πᵥ)
end


println("Toy Generate Samples - Rhat") 
begin
     @show res.rhatξ.ξ 
     @show res.rhatγ.γ 
end

println("Toy Generate Samples - Summary") 
begin
     out = Summary(res);
     @show out.edge_coef[!,:estimate] 
     @show out.prob_nodes[!,:probability]
end

# Test with test1.csv data

println("-------------TEST 1---------------")

data_in = DataFrame(CSV.File(joinpath(@__DIR__, "data", "test1.csv")))
X = Matrix(data_in[:,1:190])
y = data_in[:,191]
R = 5

res2 = BayesianNetworkRegression.generate_samples!(X, y, R, η=1.01,ζ=1.0,ι=1.0,aΔ=1.0,bΔ=1.0,
    ν=10, nburn=200, nsamp=200, maxburn=200, psrf_cutoff=1.2, x_transform=false, 
    suppress_timer=false, num_chains=1, seed=1234, purge_burn=nothing)

println("Test1 Generate Samples - Parameters")
begin
    @show res2.state.τ²[1:10]
    @show mean(res2.state.τ²)
    @show res2.state.u[1:10,1:5,1]
    @show mean(res2.state.u)
    @show res2.state.ξ[1:10,1:4,1]
    @show mean(res2.state.ξ) 
    @show res2.state.γ[1:10,1:6,1]
    @show mean(res2.state.γ) 
    @show res2.state.S[1:10,1:6,1] 
    @show mean(res2.state.S) 
    @show res2.state.θ[1:10] 
    @show mean(res2.state.θ) 
    @show res2.state.Δ[1:10,:,1] 
    @show mean(res2.state.Δ) 
    @show res2.state.M[1:10,:,1] 
    @show mean(res2.state.M) 
    @show res2.state.μ[1:10] 
    @show mean(res2.state.μ) 
    @show res2.state.πᵥ[1:10,:,1]
    @show mean(res2.state.πᵥ)
end

println("Test1 Generate Samples - Rhat") 
begin
    @show res2.rhatξ.ξ[1:10]
    @show res2.rhatγ.γ[1:10] 
end

println("Test1 Generate Samples - Summary") 
begin
    out2 = Summary(res2);
    @show out2.edge_coef[1:10,:estimate] 
    @show out2.prob_nodes[1:10,:probability] 
    []
end

jldsave("test/data/gen_test_results.jld2"; st1, st2, st3, res, out, res2, out2)