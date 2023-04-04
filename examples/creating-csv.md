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
