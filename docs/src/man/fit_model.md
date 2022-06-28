# Fitting the Model

## Network Estimation

BayesianNetworkRegression implements the statistical inference method for
[Ozminkowski & Solís-Lemus 2022]

### Using Matrix-style input
After [Input Data](@ref), we can estimate the network using the
input data `matrix_networks`. 

We will run two chains in parallel in order to generate convergence statistics. If you don't want to run in parallel, delete the first two lines of the next code block, the `@everywhere` decorator, and change `in_seq=true` in the `Fit!` call below.

We first load the data and split it into the covariate matrix and the response vector

```julia
using Distributed
addprocs(2)

@everywhere begin
    using BayesianNetworkRegression,CSV,DataFrames,StaticArrays

    matrix_networks = joinpath(dirname(pathof(BayesianNetworkRegression)), "..","examples","matrix_networks.csv")
    data_in = DataFrame(CSV.File(matrix_networks))

    X = Matrix(data_in[:,names(data_in,Not("y"))])
    y = Vector{Float64}(data_in[:,:y])
end
```

Then we decide what the dimensionality of the latent node space (R) should be. Quality of inference will increase with increasing dimensionality up to a point, after which it will approximately plateau. The time it takes to fit the model will increase (worse than linearly) with increasing dimensionality. For our simulations, R=7 worked best. 

We will run the model with 20000 burn-in followed by 10000 retained samples. This will likely not be enough to achieve convergence (but the form of the results will be the same).

```julia
result = Fit!(X, y, 7, # we set R=7
    nburn=20000, nsamples=10000, x_transform=false, # x_transform=false indicates we're inputting the data in Matrix-style, it doesn't need to be re-organized
    num_chains=2, # we'll run two chains, allowing us to check for convergence
    seed=1234, # we'll set a random seed for reproducibility
    in_seq=false # if we did not want to use parallelization, there is an option to run the chains in sequence
    )
```

***Note: a progress-meter is utilized to keep you informed of the estimated completion time, but when parallelization is used updating it is very costly so it is only updated sporadically.*** 

### Using Vector-style input
After [Input Data](@ref), we can estimate the network using the
input data `vector_networks` and `vector_response`. 

We first load the data and split it into the covariate matrix and the response vector

```julia
using Distributed
addprocs(2)

@everywhere begin
    using BayesianNetworkRegression,JLD2,DataFrames,StaticArrays

    vector_networks = JLD2.load(joinpath(dirname(pathof(BayesianNetworkRegression)), "..","examples","vector_networks.jld2"))
    vector_response = JLD2.load(joinpath(dirname(pathof(BayesianNetworkRegression)), "..","examples","vector_response.jld2"))

    X = vector_networks["networks"]
    y = vector_response["response"]
end
```

Then we decide what the dimensionality of the latent node space (R) should be. Quality of inference will increase with increasing dimensionality up to a point, after which it will approximately plateau. The time it takes to fit the model will increase (worse than linearly) with increasing dimensionality. For our simulations, R=7 worked best. 

We will run the model with 20000 burn-in followed by 10000 retained samples. This will likely not be enough to achieve convergence (but the form of the results will be the same).

```julia
result = Fit!(X, y, 7, # we set R=7
    nburn=20000, nsamples=10000, x_transform=true, # x_transform=true indicates we're inputting the data in Vector-style, it needs to be re-organized
    num_chains=2, # we'll run two chains, allowing us to check for convergence
    seed=1234, # we'll set a random seed for reproducibility
    in_seq=false # if we did not want to use parallelization, there is an option to run the chains in sequence
    )
```

***Note: a progress-meter is utilized to keep you informed of the estimated completion time, but when parallelization is used updating it is very costly so it is only updated sporadically.*** 

## Error reporting

Please report any bugs and errors by opening an
[issue](https://github.com/samozm/BayesianNetworkRegression.jl/issues/new).