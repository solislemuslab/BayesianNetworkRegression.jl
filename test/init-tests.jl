## Initial tests for BayesianNetworkRegression.jl on toy data

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

truth = load("data/gen_test_results.jld2")

st1 = Table(τ² = Array{Float64,3}(undef,(total,1,1)), u = Array{Float64,3}(undef,(total,R,V)),
                  ξ = Array{Float64,3}(undef,(total,V,1)), γ = Array{Float64,3}(undef,(total,q,1)),
                  S = Array{Float64,3}(undef,(total,q,1)), θ = Array{Float64,3}(undef,(total,1,1)),
                  Δ = Array{Float64,3}(undef,(total,1,1)), M = Array{Float64,3}(undef,(total,R,R)),
                  μ = Array{Float64,3}(undef,(total,1,1)), λ = Array{Float64,3}(undef,(total,R,1)),
                  πᵥ= Array{Float64,3}(undef,(total,R,3)));

X_new = Array{Float64,2}(undef,n,q)
##X_new = rand(rng, Normal(0,2),n,q) ## test added thinking undef was causing float issues, but no

@testset "InitTests - Dimensions and initializations" begin
    tmprng = Xoshiro(100)
    BayesianNetworkRegression.initialize_variables!(st1, η, R, ν,tmprng, V)

    @test size(X_new) == (n,q)
    @test size(st1.S[1,:,1]) == (q,)
    @test size(st1.πᵥ[1,:,:]) == (R,3)
    @test size(st1.λ[1,:,1]) == (R,)
    @test issubset(st1.ξ[1,:,1],[0,1])
    @test size(st1.M[1,:,:]) == (R,R)
    @test size(st1.u[1,:,:]) == (R,V)
    @test size(st1.γ[1,:,1]) == (q,)

    @test st1.S[1,:,1] ≈ truth["st1"].S[1,:,1] rtol=1.0e-5
    @test st1.πᵥ[1,:,:] ≈ truth["st1"].πᵥ[1,:,:] rtol=1.0e-5
    @test st1.λ[1,:,1] ≈ truth["st1"].λ[1,:,1] rtol=1.0e-5
    @test st1.ξ[1,:,1] ≈ truth["st1"].ξ[1,:,1] rtol=1.0e-5
    @test st1.u[1,:,:] ≈ truth["st1"].u[1,:,:] rtol=1.0e-5
    @test st1.M[1,:,:] ≈ truth["st1"].M[1,:,:] rtol=1.0e-5   
    @test st1.γ[1,:,1] ≈ truth["st1"].γ[1,:,1] rtol=1.0e-5
end


@testset "InitTests - Gibbs sampler" begin
    tmprng = Xoshiro(100)
    BayesianNetworkRegression.gibbs_sample!(st1, 2, X_new, y, V, η, ζ, ι, R, aΔ, bΔ, ν, tmprng)

    @test st1.τ²[2,1,1] ≈  truth["st2"].τ²[2,1,1] rtol=1.0e-5
    @test st1.ξ[2,:,1] ≈ truth["st2"].ξ[2,:,1] rtol=1.0e-5
    @test st1.u[2,:,:] ≈ truth["st2"].u[2,:,:] rtol=1.0e-5
    @test st1.γ[2,:,1] ≈ truth["st2"].γ[2,:,1] rtol=1.0e-5
    @test st1.θ[2,1,1] ≈ truth["st2"].θ[2,1,1] rtol=1.0e-5
    @test st1.Δ[2,1,1] ≈ truth["st2"].Δ[2,1,1] rtol=1.0e-5
    @test st1.M[2,:,:] ≈ truth["st2"].M[2,:,:] rtol=1.0e-5
    @test st1.μ[2,1,1] ≈ truth["st2"].μ[2,1,1] rtol=1.0e-5

    @test st1.S[2,:,1] ≈ truth["st2"].S[2,:,1] rtol=1.0e-5
    @test st1.πᵥ[2,:,:] ≈ truth["st2"].πᵥ[2,:,:] rtol=1.0e-5
    @test st1.λ[2,:,1] ≈ truth["st2"].λ[2,:,1] rtol=1.0e-5
end

@testset "InitTests - Deconstructed Gibbs sampler" begin
    n = size(X_new,1)
    rng = Xoshiro(123)

    BayesianNetworkRegression.update_τ²!(st1, 3, X_new, y, V, rng)
    @test st1.τ²[3,1,1] ≈ truth["st3"].τ²[3,1,1] rtol=1.0e-5

    BayesianNetworkRegression.update_u_ξ!(st1, 3, V, rng)
    @test st1.ξ[3,:,1] ≈ truth["st3"].ξ[3,:,1] rtol=1.0e-5
    @test st1.u[3,:,:] ≈ truth["st3"].u[3,:,:] rtol=1.0e-5

    BayesianNetworkRegression.update_γ!(st1, 3, X_new, y, n, rng)
    @test st1.γ[3,:,1] ≈ truth["st3"].γ[3,:,1] rtol=1.0e-5

    BayesianNetworkRegression.update_D!(st1, 3, V, rng)
    BayesianNetworkRegression.update_θ!(st1, 3, ζ, ι, V, rng)
    @test st1.θ[3,1,1] ≈ truth["st3"].θ[3,1,1] rtol=1.0e-5

    BayesianNetworkRegression.update_Δ!(st1, 3, aΔ, bΔ, rng)
    @test st1.Δ[3,1,1] ≈ truth["st3"].Δ[3,1,1] rtol=1.0e-5

    BayesianNetworkRegression.update_M!(st1, 3, ν, V, rng)
    @test st1.M[3,:,:] ≈ truth["st3"].M[3,:,:] rtol=1.0e-5

    BayesianNetworkRegression.update_μ!(st1, 3, X_new, y, n, rng)
    @test st1.μ[3,1,1] ≈ truth["st3"].μ[3,1,1] rtol=1.0e-5

    BayesianNetworkRegression.update_Λ!(st1, 3, R, rng)
    @test st1.λ[3,:,1] ≈ truth["st3"].λ[3,:,1] rtol=1.0e-5

    BayesianNetworkRegression.update_π!(st1, 3, η, R, rng)
    @test st1.πᵥ[3,:,:] ≈ truth["st3"].πᵥ[3,:,:] rtol=1.0e-5
    @test st1.S[3,:,1] ≈ truth["st3"].S[3,:,1] rtol=1.0e-5
end