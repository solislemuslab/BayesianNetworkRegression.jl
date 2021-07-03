using Test,BayesianNetworkRegression,LinearAlgebra,Distributions
using CSV,DataFrames

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


@testset "InitTests" begin
    X_new, θ, D, πᵥ, Λ, Δ, ξ, M, u, μ, τ², γ = BayesianNetworkRegression.init_vars!(Z, η, ζ, ι, R, aΔ, bΔ, ν, V,true)

    @test size(X_new) == (n,q)
    @test size(D[1]) == (q,q)
    @test size(πᵥ[1]) == (R,3)
    @test size(Λ[1]) == (R,R)
    @test issubset(ξ[1],[0,1])
    @test size(M[1]) == (R,R)
    @test size(u[1]) == (R,V)
    @test size(γ[1]) == (q,)
end

@testset "Sim tests" begin
    R  = 5
    V = 20
    nburn = 200
    nsamp = 200

    data_in = DataFrame(CSV.File("test/data/test1.csv"))

    X = Matrix(data_in[:,1:190])
    y = data_in[:,191]

    result = GenerateSamples!(X, y, R, nburn=nburn,nsamples=nsamp, V=V, aΔ=1.0, bΔ=1.0,ν=10 ,ι=1.0,ζ=1.0,x_transform=false)

    @test size(result.Gammas) == (nsamp,)
    @test size(result.Xis) == (nsamp,)
    @test size(result.us) == (nsamp,)

    @test size(result.us[1]) == (R,V)
end