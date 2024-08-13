# Fitting the Model

## Single-thread run

The `BayesianNetworkRegression.jl` package implements the statistical inference method in [Ozminkowski & Solís-Lemus (2024)](https://doi.org/10.1002/ece3.11039) in its main function [`Fit!`](@ref) which takes three parameters:
- `X`: data matrix from microbial networks (networked covariates)
- `y`: response vector
- `R`: dimension of latent variables 

Quality of inference will increase with increasing `R` up to a point, after which it will approximately plateau. The time it takes to fit the model will increase (worse than linearly) with increasing `R`. For our simulations, `R=7` worked best. 

We will use the data read in the [Input Data](@ref) section:
- `X_a`: vector of adjacency matrices
- `X_v`: matrix of vectorized adjacency matrices

The type of input data matrix `X` will inform the argument `x_transform` so that:
- `x_transform=true` means that the input matrix `X` needs to be vectorized (as for `X_a`),
- `x_transform=false` means that the input matrix `X` has already been vectorized (as for `X_v`).

To fit the model, we type this in julia:
```julia
using BayesianNetworkRegression

result = Fit!(X_a, y_a, 5, # we set R=5
    nburn=200, nsamples=100, x_transform=true, 
    num_chains=1, 
    seed=1234)
```
Note that we are running a very small chain (300 generations: 200 burnin and 100 post burnin). For a real analysis, these number should be much larger (see the Simulation in the manuscript for more details).

The `result` variable is a [`Result`](@ref) type which contains five attributes:
- `state::Table`: Table with all the sampled parameters for all generations.
- `rhatξ::Table`: Table with convergence information ($\hat{R}$ values) for the parameters $\xi$ representing whether specific nodes are influential or not.
- `rhatγ::Table`: Table with convergence information ($\hat{R}$ values) for the parameters $\gamma$ representing the regression coefficients.
- `burn_in::Int`: number of burnin samples.
- `sampled::Int`: number of post burnin samples.

Each of these objects can be accessed with a `.`, for example, `results.state` will produce the table with all the samples.

We use the convergence criteria proposed in [Vehtari et al (2019)](https://arxiv.org/abs/1903.08008). Values close to 1 indicate convergence. Vehtari et al. suggest using a cutoff of $\hat{R} < 1.01$ to indicate convergence. Values are provided for all $\xi$ (in `rhatξ`) and $\gamma$ (in `rhatγ`) variables.

These attributes are not readily interpretable, and thus, they can be summarized with the [`Summary`](@ref) function:

```julia
out = Summary(result)
```

Note that the `show` function for a `Results` object is already calling `Summary(result)` internally, and thus, if you simply call:
```julia
result
```
you observe the same output as when calling
```julia
out = Summary(result)
```

The `out` object is now a `BNRSummary` object with two main data frames:

1. DataFrame with edge coefficient point estimates and endpoints of credible intervals (default 95%):
```
julia> out.edge_coef
465×5 DataFrame
 Row │ node1  node2  estimate  lower_bound  upper_bound 
     │ Int64  Int64  Float64   Float64      Float64     
─────┼──────────────────────────────────────────────────
   1 │     1      1     2.677       -0.753        5.392
   2 │     1      2     2.573       -1.27         4.881
   3 │     1      3     2.106        0.207        3.876
  ⋮  │   ⋮      ⋮       ⋮           ⋮            ⋮
 463 │    29     29     9.074        5.199       13.233
 464 │    29     30     3.015       -1.028        7.793
 465 │    30     30     1.262       -1.334        4.296
                                        459 rows omitted
```

2. DataFrame with probabilities of being influential for each node
```
julia> out.prob_nodes
30×1 DataFrame
 Row │ probability 
     │ Float64     
─────┼─────────────
   1 │        1.0
   2 │        1.0
   3 │        1.0
  ⋮  │      ⋮
  28 │        1.0
  29 │        1.0
  30 │        1.0
    24 rows omitted
```

The `BNRSummary` object also keeps the level for the credible interval, set at 95% by default:
```
julia> out.ci_level
95
```

The level can be changed when running the [`Summary`](@ref) function:
```julia
out2=Summary(result,interval=97)
```

We will use these summary data frames in the [Interpretation](@ref) section.

Note that we ran the case when we need to transform the data matrix. If we already have the adjacency matrices vectorized, we simply need to set `x_transform=false`:
```julia
result2 = Fit!(X_v, y_v, 5, # we set R=5
    nburn=200, nsamples=100, x_transform=false, 
    num_chains=1, 
    seed=1234)
```

## Multi-thread run

You can run multiple chains in parallel by setting up multiple processors as shown next.

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

Compared to the single-thread run, we only need to change `num_chains=2`:
```julia
result3 = Fit!(X, y, 5,
    nburn=200, nsamples=100, x_transform=false, 
    num_chains=2, 
    seed=1234)
```

## Error reporting

Please report any bugs and errors by opening an
[issue](https://github.com/solislemuslab/BayesianNetworkRegression.jl/issues/new).