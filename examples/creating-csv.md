Code to save the csv files for the documentation:

```julia
using BayesianNetworkRegression
using JLD2
cd(joinpath(dirname(pathof(BayesianNetworkRegression)), "..","examples"))

vector_networks = JLD2.load("vector_networks.jld2")
vector_response = JLD2.load("vector_response.jld2")

using CSV, Tables
for i in 1:100
CSV.write(string("data",i,".csv"),Tables.table(vector_networks["networks"][i]))
end

CSV.write("responses.csv",Tables.table(vector_response["response"]))
```

Code to save the csv file for all networks in one file:

```julia
using BayesianNetworkRegression
using JLD2
cd(joinpath(dirname(pathof(BayesianNetworkRegression)), "..","examples"))

vector_networks = JLD2.load("vector_networks.jld2")
vector_response = JLD2.load("vector_response.jld2")

using CSV,DataFrames

n = 100
V = 30
q = floor(Int64,(V*(V+1)/2))
X_new = Matrix(undef, n, q)
BayesianNetworkRegression.setup_X!(X_new,vector_networks["networks"],true)
output = hcat(DataFrame(X_new,:auto),DataFrame(y=vector_response["response"]))
CSV.write("matrix_networks.csv",output)
```