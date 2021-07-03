# BayesianNetworkRegression.jl

## WARNING: THIS PROJECT IS STILL BEING TESTED AND IS NOT YET RELEASED. 
### RESULTS MAY NOT BE ACCURATE. 

-----------------

## Tests
`test/runtests.jl` contains some basic tests. 

A brief example can be explored by running the following in Julia (from the BayesianNetworkRegression directory):

```
using BayesianNetworkRegression,CSV,DataFrames
R  = 5
V = 20
nburn = 200
nsamp = 200

data_in = DataFrame(CSV.File("test/data/test1.csv"))

X = Matrix(data_in[:,1:190])
y = data_in[:,191]

result = GenerateSamples!(X, y, R, nburn=nburn,nsamples=nsamp, aΔ=1.0, bΔ=1.0,ν=10,ι=1.0,ζ=1.0, V=V, x_transform=false)
```

`test/data/test1.csv` contains a small simulation as described (as simulation 2 case 1) in [Guha & Rodriguez (2020)](https://doi.org/10.1080/01621459.2020.1772079)


### Output is as follows:

`result.Gammas` returns an array of arrays (Array{Array{Float64,1},1}) - each inner array contains the gamma values (edge coefficients) for all edges for one Gibbs sample

`result.Xis` returns an array of arrays (Array{Array{Int64,1},1}) - each inner array contains the xi values (whether the node is influential) for all nodes for one Gibbs sample. 

`result.us` returns an array of arrays (Array{Array{Float64,2},1}) - each inner array contains the u values (used in determing the influence of the edges on the response) for all nodes for one Gibbs sample