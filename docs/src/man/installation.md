# Installation

## Installation of Julia

Julia is a high-level and interactive programming language (like R or Matlab),
but it is also high-performance (like C).
To install Julia, follow instructions [here](http://julialang.org/downloads/).
For a quick & basic tutorial on Julia, see
[learn x in y minutes](http://learnxinyminutes.com/docs/julia/).

Editors:

- [Visual Studio Code](https://code.visualstudio.com) provides an editor
  and an integrated development environment (IDE) for Julia: highly recommended!
- you can also run Julia within a [Jupyter](http://jupyter.org) notebook
  (formerly IPython notebook).

IMPORTANT: Julia code is just-in-time compiled. This means that the
first time you run a function, it will be compiled at that moment. So,
please be patient! Future calls to the function will be much much
faster. Trying out toy examples for the first calls is a good idea.

## Installation of the package PhyloNetworks

To install the package, 
1. [Clone](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository) [BayesianNetworkRegression.jl](https://github.com/samozm/BayesianNetworkRegression.jl) to your local machine.
4. type inside Julia:
```julia
]
dev PATH/BayesianNetworkRegression.jl
```
where PATH is the path to the BayesianNetworkRegression.jl depository on your machine.

The first step can take a few minutes, be patient.

The BayesianNetworkRegression package has dependencies like
[Distributions](https://juliastats.org/Distributions.jl/stable/starting/) and
[DataFrames](http://juliadata.github.io/DataFrames.jl/stable/)
(see the `Project.toml` file for the full list), but everything is installed automatically.

## Loading the Package

To check that your installation worked, type this in Julia to load the package.
This is something to type every time you start a Julia session:
```@example install
using BayesianNetworkRegression;
```
This step can also take a while, if Julia needs to pre-compile the code (after a package
update for instance).

Press `?` inside Julia to switch to help mode, 
followed by the name of a function (or type) to get more details about it.

