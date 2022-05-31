# Input for BayesianNetworkRegression

BayesianNetworkRegression is the main method implemented in the package, to estimate the relationships between the edges of the microbiome network and the variable of interest. 
The variable of interest should be a vector of real numbers.
There are two alternatives for the network input data:

1. A vector of networks, where each item in the vector (of length $n$, where $n$ is the samplesize) is an $m \times m$ (where $m$ is the number of microbes in each sample) adjacency matrix describing the network. All adjacency matrices must be the same size.
2. A $n \times \frac{m(m-1)}{2}$ matrix, where each row in the matrix is the upper triangle of the adjacency matrix describing the network for that sample.

## Tutorial data: Matrix format

We suggest that you create a special directory for running these examples,
where input files can be downloaded and where output files will be
created (with estimated networks for instance). Enter this directory
and run Julia from there.

Suppose you have a file with n rows and m(m-1)/2 + 1 columns. For each row, the first m(m-1)/2 columns describe the upper triangle of an adjacency matrix and the last gives the response variable. 
You can access the example file of input networks (and response)
[here](https://github.com/crsl4/PhyloNetworks/blob/master/examples/matrix_networks.csv)


Do not copy-paste into a "smart" text-editor. Instead, save the file
directly into your working directory using "save link as" or "download linked file as".
This file contains 100 adjacency matrices and corresponding responses.

If `matrix_networks.csv` is in your working directory, you can view its content
within Julia:
```julia
less("matrix_networks.csv")
```
or like this, to view the version downloaded with the package:
```julia
matrix_networks = joinpath(dirname(pathof(BayesianNetworkRegression)), "..","examples","matrix_networks.csv")
less(matrix_networks)
```
Just type `q` to quit viewing this file.


## Tutorial data: Vector format

We suggest that you create a special directory for running these examples,
where input files can be downloaded and where output files will be
created (with estimated networks for instance). Enter this directory
and run Julia from there.

For this data format, we will utilize the JLD2 package, which handles julia formatting.

Suppose you have two files, one containing a vector of adjacency matrices and the other a vector of responses (real numbers).
You can access the example files for the networks 
[here](https://github.com/crsl4/PhyloNetworks/blob/master/examples/vector_networks.jld2)
and for the responses
[here](https://github.com/crsl4/PhyloNetworks/blob/master/examples/vector_response.jld2)

To load the data and view an example in julia do the following:
```julia
using JLD2
vector_networks = JLD2.load("vector_networks.jld2")
vector_response = JLD2.load("vector_response.jld2")

vector_networks["networks"][1] # shows the first adjancency matrix
vector_response["response"] # shows all responses
```