# Old tests; need to include them as files into runtests.jl; one by one.
import InteractiveUtils: versioninfo
using BayesianNetworkRegression,Test,LinearAlgebra,Distributions
using CSV,DataFrames,StaticArrays,TypedTables,Random,Distributed

seed = 2358

η  = 1.01
ζ  = 1.0
ι  = 1.0
R  = 7
aΔ = 1.0
bΔ = 1.0
ν = 10

@testset "Result tests - master" begin
    R  = 7
    V = 30
    nburn = 30000
    nsamp = 20000
    total = nburn+nsamp
    q = floor(Int,V*(V-1)/2)
    seed = 2358
    num_chains = 1

    data_in = DataFrame(CSV.File(joinpath(@__DIR__, "data", "mu=0.8_n_microbes=15_out=XYs_pi=0.8_samplesize=100_simnum=1.csv")))
    edges_res = DataFrame(CSV.File(joinpath(@__DIR__,"data","R=7_mu=0.8_n_microbes=15_nu=10_out=edges_pi=0.8_samplesize=100_simnum=1.csv")))
    nodes_res = DataFrame(CSV.File(joinpath(@__DIR__,"data","R=7_mu=0.8_n_microbes=15_nu=10_out=nodes_pi=0.8_samplesize=100_simnum=1.csv")))

    X = Matrix(data_in[:,names(data_in,Not("y"))])
    y = SVector{size(X,1)}(data_in[:,:y])


    if V != convert(Int,(1 + sqrt(1 + 8*size(X,2)))/2)
        println("wrong V")
    end

    result2 = Fit!(X, y, R, η=η, nburn=nburn,nsamples=nsamp, aΔ=aΔ, 
                    bΔ=bΔ,ν=ν,ι=ι,ζ=ζ,x_transform=false,purge_burn=10000,
                    num_chains=num_chains,seed=seed)

    @show seed 
    nburn=10000
    total=nburn+nsamp

    γ_sorted = sort(result2.state.γ[nburn+1:total,:,:],dims=1)
    @show size(γ_sorted)
    lw = convert(Int64, round(nsamp * 0.025))
    hi = convert(Int64, round(nsamp * 0.975))
    
    ci_df = DataFrame(mean=mean(result2.state.γ[nburn+1:total,:,:],dims=1)[1,:])
    ci_df[:,"0.025"] = γ_sorted[lw,:,1]
    ci_df[:,"0.975"] = γ_sorted[hi,:,1]

    nburn = 30000
    addprocs(1, exeflags=["--optimize=0","--math-mode=ieee","--check-bounds=yes"])
    @everywhere begin
        using BayesianNetworkRegression,Test,LinearAlgebra,Distributions
        using CSV,DataFrames,StaticArrays,TypedTables,Random,Distributed
        R  = 7
        V = 30
        nburn = 30000
        nsamp = 20000
        total = nburn+nsamp
        q = floor(Int,V*(V-1)/2)
        seed = 2358
        num_chains = 1

        data_in = DataFrame(CSV.File(joinpath(@__DIR__, "data", "mu=0.8_n_microbes=15_out=XYs_pi=0.8_samplesize=100_simnum=1.csv")))
        X = Matrix(data_in[:,names(data_in,Not("y"))])
        y = SVector{size(X,1)}(data_in[:,:y])

    end
    result22 = Fit!(X, y, R, η=η, nburn=nburn,nsamples=nsamp, aΔ=aΔ, 
                    bΔ=bΔ,ν=ν,ι=ι,ζ=ζ,x_transform=false,purge_burn=10000,
                    num_chains=num_chains,seed=seed)

    @show seed 
    nburn=10000
    total=nburn+nsamp

    γ_sorted2 = sort(result22.state.γ[nburn+1:total,:,:],dims=1)
    @show size(γ_sorted2)
    lw = convert(Int64, round(nsamp * 0.025))
    hi = convert(Int64, round(nsamp * 0.975))
    
    ci_df2 = DataFrame(mean=mean(result22.state.γ[nburn+1:total,:,:],dims=1)[1,:])
    ci_df2[:,"0.025"] = γ_sorted2[lw,:,1]
    ci_df2[:,"0.975"] = γ_sorted2[hi,:,1]

    @test isapprox(mean(result2.state.γ[nburn+1:total,:,:],dims=1)[1,:],
                  mean(result22.state.γ[nburn+1:total,:,:],dims=1)[1,:])
    @test isapprox(ci_df[:,"0.025"], ci_df2[:,"0.025"])
    @test isapprox(ci_df[:,"0.975"], ci_df2[:,"0.975"])
    @test isapprox(mean(result2.state.ξ[nburn+1:total,:,:],dims=1)[1,:],nodes_res[:,"Xi posterior"])

    @test isapprox(mean(result2.state.γ[nburn+1:total,:,:],dims=1)[1,:], edges_res.mean)
    @test isapprox(ci_df[:,"0.025"], edges_res[:,"0.025"])
    @test isapprox(ci_df[:,"0.975"], edges_res[:,"0.975"])

    xis = zeros(30)
    for i=1:30 xis[i] = isapprox(mean(result2.state.ξ[nburn+1:total,:,:],dims=1)[1,i],nodes_res[i,"Xi posterior"]) end
    @test xis == ones(30)

    @show DataFrame(loc = mean(result2.state.ξ[nburn+1:total,:,:],dims=1)[1,:], 
                    worker = mean(result22.state.ξ[nburn+1:total,:,:],dims=1)[1,:],
                    real = nodes_res[:,"Xi posterior"])
end 

#addprocs(1,exeflags="--optimize=0")
rmprocs()

@testset "Result tests - worker" begin
    seed = 2358
    nburn=30000
    nsamp=20000
    total=nburn+nsamp
    R = 7

    data_in = DataFrame(CSV.File(joinpath(@__DIR__, "data", "mu=1.6_n_microbes=8_out=XYs_pi=0.8_samplesize=100_simnum=1.csv")))

    X = Matrix(data_in[:,names(data_in,Not("y"))])
    y = SVector{size(X,1)}(data_in[:,:y])

    V = 30
    q = floor(Int,V*(V-1)/2)
    num_chains = 1

    result33 = Fit!(X, y, R, η=η, nburn=nburn,nsamples=nsamp, aΔ=aΔ, 
    bΔ=bΔ,ν=ν,ι=ι,ζ=ζ,x_transform=false,
    num_chains=num_chains,seed=seed)

    γ_sorted33 = sort(result33.state.γ[nburn+1:total,:,:],dims=1)
    lw = convert(Int64, round(nsamp * 0.025))
    hi = convert(Int64, round(nsamp * 0.975))

    ci_df33 = DataFrame(mean=mean(result33.state.γ[nburn+1:total,:,:],dims=1)[1,:])
    ci_df33[:,"0.025"] = γ_sorted33[lw,:,1]
    ci_df33[:,"0.975"] = γ_sorted33[hi,:,1]


    addprocs(1, exeflags=["--optimize=0","--math-mode=ieee","--check-bounds=yes"])
    seed = 2358
    nburn=30000
    nsamp=20000
    total=nburn+nsamp
    R = 7

    data_in = DataFrame(CSV.File(joinpath(@__DIR__, "data", "mu=1.6_n_microbes=8_out=XYs_pi=0.8_samplesize=100_simnum=1.csv")))

    X = Matrix(data_in[:,names(data_in,Not("y"))])
    y = SVector{size(X,1)}(data_in[:,:y])

    V = 30
    q = floor(Int,V*(V-1)/2)
    num_chains = 1

    @everywhere begin
        using BayesianNetworkRegression,Test,LinearAlgebra,Distributions
        using CSV,DataFrames,StaticArrays,TypedTables,Random,Distributed
        R = 7
        nburn = 30000
        nsamp = 20000
        total = nburn+nsamp
        seed = 2358

        data_in = DataFrame(CSV.File(joinpath(@__DIR__, "data", "mu=1.6_n_microbes=8_out=XYs_pi=0.8_samplesize=100_simnum=1.csv")))

        X = Matrix(data_in[:,names(data_in,Not("y"))])
        y = SVector{size(X,1)}(data_in[:,:y])

        V = 30
        q = floor(Int,V*(V-1)/2)
        num_chains = 1
    end
    
    result3 = Fit!(X, y, R, η=η,nburn=nburn,nsamples=nsamp, aΔ=aΔ, 
                    bΔ=bΔ,ν=ν,ι=ι,ζ=ζ,x_transform=false,
                    num_chains=num_chains,seed=seed)

    γ_sorted3 = sort(result3.state.γ[nburn+1:total,:,:],dims=1)
    lw = convert(Int64, round(nsamp * 0.025))
    hi = convert(Int64, round(nsamp * 0.975))
    
    ci_df3 = DataFrame(mean=mean(result3.state.γ[nburn+1:total,:,:],dims=1)[1,:])
    ci_df3[:,"0.025"] = γ_sorted3[lw,:,1]
    ci_df3[:,"0.975"] = γ_sorted3[hi,:,1]

    edges_res3 = DataFrame(CSV.File(joinpath(@__DIR__,"data","R=7_mu=1.6_n_microbes=8_nu=10_out=edges_pi=0.8_samplesize=100_simnum=1.csv")))
    nodes_res3 = DataFrame(CSV.File(joinpath(@__DIR__,"data","R=7_mu=1.6_n_microbes=8_nu=10_out=nodes_pi=0.8_samplesize=100_simnum=1.csv")))

    @show DataFrame(loc = mean(result3.state.ξ[nburn+1:total,:,:],dims=1)[1,:], 
                    real = nodes_res3[:,"Xi posterior"],
                    master = mean(result33.state.ξ[nburn+1:total,:,:],dims=1)[1,:])

    @test isapprox(mean(result3.state.γ[nburn+1:total,:,:],dims=1)[1,:],
    mean(result33.state.γ[nburn+1:total,:,:],dims=1)[1,:])
    @test isapprox(ci_df3[:,"0.025"], ci_df33[:,"0.025"])
    @test isapprox(ci_df3[:,"0.975"], ci_df33[:,"0.975"])


    @test isapprox(mean(result3.state.γ[nburn+1:total,:,:],dims=1)[1,:], edges_res3.mean)
    @test isapprox(ci_df3[:,"0.025"], edges_res3[:,"0.025"])
    @test isapprox(ci_df3[:,"0.975"], edges_res3[:,"0.975"])

    xis = zeros(30)
    for i=1:30 xis[i] = isapprox(mean(result3.state.ξ[nburn+1:total,:,:],dims=1)[1,i],nodes_res3[i,"Xi posterior"]) end
    @test xis == ones(30)
end