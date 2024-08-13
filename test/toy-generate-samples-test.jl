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

truth = load("data/gen_test_results.jld2")

res = BayesianNetworkRegression.generate_samples!(X, y, R, η=1.01,ζ=1.0,ι=1.0,aΔ=1.0,bΔ=1.0,
    ν=10, nburn=300, nsamp=100, maxburn=300, psrf_cutoff=1.2, x_transform=true, 
    suppress_timer=false, num_chains=1, seed=1234, purge_burn=nothing)

@testset "Toy Generate Samples - Parameters" begin
     @test res.state.τ²[1:10] ≈ truth["res"].state.τ²[1:10] rtol=1.0e-5
     @test mean(res.state.τ²) ≈ mean(truth["res"].state.τ²) rtol=1.0e-5
     @test res.state.u[1:10,1:5,1] ≈ truth["res"].state.u[1:10,1:5,1] rtol=1.0e-5
     @test mean(res.state.u) ≈ mean(truth["res"].state.u) rtol=1.0e-5
     @test res.state.ξ[1:10,:,1] ≈ truth["res"].state.ξ[1:10,:,1] rtol=1.0e-5
     @test mean(res.state.ξ) ≈ mean(truth["res"].state.ξ) rtol=1.0e-5
     @test res.state.γ[1:10,:,1] ≈ truth["res"].state.γ[1:10,:,1] rtol=1.0e-5
     @test mean(res.state.γ) ≈ mean(truth["res"].state.γ) rtol=1.0e-5
     @test res.state.S[1:10,:,1] ≈ truth["res"].state.S[1:10,:,1] rtol=1.0e-5
     @test mean(res.state.S) ≈ mean(truth["res"].state.S) rtol=1.0e-5
     @test res.state.θ[1:10] ≈ truth["res"].state.θ[1:10] rtol=1.0e-5
     @test mean(res.state.θ) ≈ mean(truth["res"].state.θ) rtol=1.0e-5
     @test res.state.Δ[1:10,:,1] ≈ truth["res"].state.Δ[1:10,:,1] rtol=1.0e-5
     @test mean(res.state.Δ) ≈ mean(truth["res"].state.Δ) rtol=1.0e-5
     @test res.state.M[1:10,:,1] ≈ truth["res"].state.M[1:10,:,1] rtol=1.0e-5
     @test mean(res.state.M) ≈ mean(truth["res"].state.M) rtol=1.0e-5
     @test res.state.μ[1:10] ≈ truth["res"].state.μ[1:10] rtol=1.0e-5
     @test mean(res.state.μ) ≈ mean(truth["res"].state.μ) rtol=1.0e-5
     @test res.state.πᵥ[1:10,:,1] ≈ truth["res"].state.πᵥ[1:10,:,1] rtol=1.0e-5
     @test mean(res.state.πᵥ) ≈ mean(truth["res"].state.πᵥ) rtol=1.0e-5
end


@testset "Toy Generate Samples - Rhat" begin
     @test res.rhatξ.ξ ≈ truth["res"].rhatξ.ξ rtol=1.0e-5
     @test res.rhatγ.γ ≈ truth["res"].rhatγ.γ rtol=1.0e-5
end

@testset "Toy Generate Samples - Summary" begin
     out = Summary(res);
     @test out.edge_coef[!,:estimate] ≈ truth["out"].edge_coef[!,:estimate] rtol=1.0e-5
     @test out.prob_nodes[!,:probability] ≈ truth["out"].prob_nodes[!,:probability] rtol=1.0e-5
end