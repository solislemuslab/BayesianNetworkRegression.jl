# Bayesian Network Regression model for microbiome networks as covariates on biological phenotypes as response <picture> <source media="(prefers-color-scheme: dark)" srcset="docs/src/logo-dark_text.png"><img alt="bayesiannetworkregression logo" src="docs/src/logo_text.png" align=right></picture>

[![CI](https://github.com/samozm/BayesianNetworkRegression.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/samozm/BayesianNetworkRegression.jl/actions/workflows/CI.yml)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://solislemuslab.github.io/BayesianNetworkRegression.jl/dev)
[![codecov](https://codecov.io/gh/samozm/BayesianNetworkRegression.jl/branch/main/graph/badge.svg?token=BVZGYMWV1D)](https://codecov.io/gh/samozm/BayesianNetworkRegression.jl)



## Overview

`BayesianNetworkRegression.jl` is a [Julia](http://julialang.org/) package to perform (Bayesian) statistical inference of a regression model with networked covariates on a real response. 

The structure of the data assumes $n$ samples, each sample with a microbial network represented as an adjacency matrix $A_i$ and a measured biological phenotype $y_i$. This package will identify influential nodes (microbes) associated with the phenotype, as well as significant edges (interactions among microbes) that are also associated with the phenotype.

Examples of data that can be fit with this model:
- Samples of soil microbiome data represented as $n$ microbial networks and measured yield of $n$ plants on that soil. The model will identify microbes and interactions associated with plant yield.
- Samples of gut microbiome data represented as $n$ microbial networks, one per human patient, and BMI measurement per human patient. The model will identify microbes and interactions associated with BMI.

## Usage

`BayesianNetworkRegression.jl` is a julia package, so the user needs to install julia, and then install the package.

To install the package, 
1. [Clone](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository) [BayesianNetworkRegression.jl](https://github.com/samozm/BayesianNetworkRegression.jl) to your local machine.
2. Type inside Julia:
```julia
]
dev PATH/BayesianNetworkRegression.jl
```
where PATH is the path to the BayesianNetworkRegression.jl depository on your machine.

## Help and errors

To get help, check the documentation [here](https://solislemuslab.github.io/BayesianNetworkRegression.jl/dev). Please report any bugs and errors by opening an
[issue](https://github.com/samozm/BayesianNetworkRegression.jl/issues/new).

## Citation

If you use `BayesianNetworkRegression.jl` in your work, we kindly ask that you cite the following paper: 
```
@article{Ozminkowski2022,
author = {Ozminkowski, S. and Sol'{i}s-Lemus, C.},
year = {2022},
title = {{Identifying microbial drivers in biological phenotypes with a Bayesian Network Regression model}},
journal = {In preparation}
}
```

## License

`BayesianNetworkRegression.jl` is licensed under a
[GNU General Public License v2.0](https://github.com/samozm/BayesianNetworkRegression.jl/blob/main/LICENSE).

## Contributions

Users interested in expanding functionalities in `BayesianNetworkRegression.jl` are welcome to do so. See details on how to contribute in [CONTRIBUTING.md](https://github.com/samozm/BayesianNetworkRegression.jl/blob/main/CONTRIBUTING.md).