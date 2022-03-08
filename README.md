# BayesianNetworkRegression.jl

## WARNING: THIS PROJECT IS STILL BEING TESTED AND IS NOT YET RELEASED. 
### RESULTS MAY NOT BE ACCURATE. 

-----------------

## Tests
`test/runtests.jl` contains some basic tests. 

A brief example can be explored by running the following in Julia (from the BayesianNetworkRegression directory):

```
using BayesianNetworkRegression,CSV,DataFrames
R  = 9
nburn = 30000
nsamp = 20000

data_in = DataFrame(CSV.File("test/data/test1.csv"))

X = Matrix(data_in[:,1:190])
y = data_in[:,191]

result = Fit!(X, y, R, nburn=nburn,nsamples=nsamp, x_transform=false)
```

`test/data/test1.csv` contains a small simulation as described (as simulation 2 case 1) in [Guha & Rodriguez (2020)](https://doi.org/10.1080/01621459.2020.1772079)


### Output is as follows:

`result.state` returns all samples for all 11 posterior variables (burn-in included)

`result.state.γ` returns an array of arrays (Array{Array{Float64,1},1}) - each inner array contains the gamma values (edge coefficients) for all edges for one Gibbs sample

`result.state.ξ` returns an array of arrays (Array{Array{Int64,1},1}) - each inner array contains the xi values (whether the node is influential) for all nodes for one Gibbs sample. 

`result.psrf.all_γ` returns PSRF convergence statistics for the γ variables

`result.psrf.all_ξ` returns PSRF convergence statistics for the ξ variables