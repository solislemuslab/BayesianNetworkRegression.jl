# Test with test1.csv data

data_in = DataFrame(CSV.File(joinpath(@__DIR__, "data", "test1.csv")))
X = Matrix(data_in[:,1:190])
y = data_in[:,191]
R = 5

truth = load("data/gen_test_results.jld2")

res = BayesianNetworkRegression.generate_samples!(X, y, R, η=1.01,ζ=1.0,ι=1.0,aΔ=1.0,bΔ=1.0,
    ν=10, nburn=200, nsamp=200, maxburn=200, psrf_cutoff=1.2, x_transform=false, 
    suppress_timer=false, num_chains=1, seed=1234, purge_burn=nothing)

@testset "Test1 Generate Samples - Parameters" begin
    @test res.state.τ²[1:10] ≈ truth["res2"].state.τ²[1:10] rtol=1.0e-5
     @test mean(res.state.τ²) ≈ mean(truth["res2"].state.τ²) rtol=1.0e-5
     @test res.state.u[1:10,1:5,1] ≈ truth["res2"].state.u[1:10,1:5,1] rtol=1.0e-5
     @test mean(res.state.u) ≈ mean(truth["res2"].state.u) rtol=1.0e-5
     @test res.state.ξ[1:10,:,1] ≈ truth["res2"].state.ξ[1:10,:,1] rtol=1.0e-5
     @test mean(res.state.ξ) ≈ mean(truth["res2"].state.ξ) rtol=1.0e-5
     @test res.state.γ[1:10,:,1] ≈ truth["res2"].state.γ[1:10,:,1] rtol=1.0e-5
     @test mean(res.state.γ) ≈ mean(truth["res2"].state.γ) rtol=1.0e-5
     @test res.state.S[1:10,:,1] ≈ truth["res2"].state.S[1:10,:,1] rtol=1.0e-5
     @test mean(res.state.S) ≈ mean(truth["res2"].state.S) rtol=1.0e-5
     @test res.state.θ[1:10] ≈ truth["res2"].state.θ[1:10] rtol=1.0e-5
     @test mean(res.state.θ) ≈ mean(truth["res2"].state.θ) rtol=1.0e-5
     @test res.state.Δ[1:10,:,1] ≈ truth["res2"].state.Δ[1:10,:,1] rtol=1.0e-5
     @test mean(res.state.Δ) ≈ mean(truth["res2"].state.Δ) rtol=1.0e-5
     @test res.state.M[1:10,:,1] ≈ truth["res2"].state.M[1:10,:,1] rtol=1.0e-5
     @test mean(res.state.M) ≈ mean(truth["res2"].state.M) rtol=1.0e-5
     @test res.state.μ[1:10] ≈ truth["res2"].state.μ[1:10] rtol=1.0e-5
     @test mean(res.state.μ) ≈ mean(truth["res2"].state.μ) rtol=1.0e-5
     @test res.state.πᵥ[1:10,:,1] ≈ truth["res2"].state.πᵥ[1:10,:,1] rtol=1.0e-5
     @test mean(res.state.πᵥ) ≈ mean(truth["res2"].state.πᵥ) rtol=1.0e-5
end

@testset "Test1 Generate Samples - Rhat" begin
    @test res.rhatξ.ξ ≈ truth["res2"].rhatξ.ξ rtol=1.0e-5
     @test res.rhatγ.γ ≈ truth["res2"].rhatγ.γ rtol=1.0e-5
end

@testset "Test1 Generate Samples - Summary" begin
    out = Summary(res);
    @test out.edge_coef[!,:estimate] ≈ truth["out2"].edge_coef[!,:estimate] rtol=1.0e-5
     @test out.prob_nodes[!,:probability] ≈ truth["out2"].prob_nodes[!,:probability] rtol=1.0e-5
end