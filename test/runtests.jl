using Test,BayesianNetworkRegression,LinearAlgebra,Distributions
using CSV,DataFrames,StaticArrays,TypedTables

function symmetrize_matrices(X)
    X_new = Array{Array{Int8,2},1}(undef,0)
    for i in 1:size(X,1)
        B = convert(Matrix, reshape(X[i], 4, 4))
        push!(X_new,Symmetric(B))
    end
    X = X_new
end

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

Z = symmetrize_matrices(X)


y = ones(size(Z[1],1))*12 + rand(Normal(0,2),size(Z[1],1))


η  = 1.01
ζ  = 1.0
ι  = 1.0
R  = 5
aΔ = 0.0
bΔ = 0.0
V = size(Z,1)
q = floor(Int,V*(V-1)/2)
n = size(Z,1)
ν = 12
total = 20

state = Table(τ² = Array{Float64,3}(undef,(total,1,1)), u = Array{Float64,3}(undef,(total,R,V)),
                  ξ = Array{Float64,3}(undef,(total,V,1)), γ = Array{Float64,3}(undef,(total,q,1)),
                  S = Array{Float64,3}(undef,(total,q,1)), θ = Array{Float64,3}(undef,(total,1,1)),
                  Δ = Array{Float64,3}(undef,(total,1,1)), M = Array{Float64,3}(undef,(total,R,R)),
                  μ = Array{Float64,3}(undef,(total,1,1)), λ = Array{Float64,3}(undef,(total,R,1)),
                  πᵥ= Array{Float64,3}(undef,(total,R,3)))

X_new = Array{Float64,2}(undef,n,q)

@testset "InitTests" begin
    BayesianNetworkRegression.initialize_variables!(state, X_new, Z, η, ζ, ι, R, aΔ, bΔ, ν, V,true)

    @test size(X_new) == (n,q)
    @test size(state.S[1,:,1]) == (q,)
    @test size(state.πᵥ[1,:,:]) == (R,3)
    @test size(state.λ[1,:,1]) == (R,)
    @test issubset(state.ξ[1,:,1],[0,1])
    @test size(state.M[1,:,:]) == (R,R)
    @test size(state.u[1,:,:]) == (R,V)
    @test size(state.γ[1,:,1]) == (q,)
end

@testset "Sim tests" begin
    R  = 5
    V = 20
    nburn = 200
    nsamp = 200
    q = floor(Int,V*(V-1)/2)

    data_in = DataFrame(CSV.File(joinpath(@__DIR__, "data", "test1.csv")))

    X = Matrix(data_in[:,1:190])
    y = data_in[:,191]

    res = GenerateSamples!(X, y, R, nburn=nburn,nsamples=nsamp, V=V, aΔ=1.0, bΔ=1.0,ν=10 ,ι=1.0,ζ=1.0,x_transform=false,num_chains=1,in_seq=true)

    result = res.state
    
    @test size(result.γ) == (nsamp+nburn+1,q,1)
    @test size(result.ξ) == (nsamp+nburn+1,V,1)
    @test size(result.u) == (nsamp+nburn+1,R,V)
end