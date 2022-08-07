# Contributing to `BayesianNetworkRegression.jl`

The following guidelines are designed for contributors to `BayesianNetworkRegression.jl`. 

## Reporting Issues

For reporting a bug or a failed function or requesting a new feature, you can simply open an issue in the [issue tracker](https://github.com/samozm/BayesianNetworkRegression.jl/issues). If you are reporting a bug, please also include a minimal code example or all relevant information for us to replicate the issue.

## Contributing Code

To make contributions to `BayesianNetworkRegression.jl`, you need to set up your [GitHub](https://github.com) 
account if you do not have and sign in, and request your change(s) or contribution(s) via 
a pull request against the ``develop``
branch of the [BayesianNetworkRegression.jl repository](https://github.com/samozm/BayesianNetworkRegression.jl). 

Please use the following steps:

1. Open a new issue for new feature or failed function in the [issue tracker](https://github.com/samozm/BayesianNetworkRegression.jl/issues)
2. Fork the [BayesianNetworkRegression.jl repository](https://github.com/samozm/BayesianNetworkRegression.jl) to your GitHub account
3. Clone your fork locally:
```
$ git clone https://github.com/your-username/BayesianNetworkRegression.jl.git
```   
4. Make your change(s) in the `master` (or `development`) branch of your cloned fork
5. Make sure that all tests (`test/runtests.jl`) are passed without any errors
6. Push your change(s) to your fork in your GitHub account
7. [Submit a pull request](https://github.com/samozm/BayesianNetworkRegression.jl/pulls) describing what problem has been solved and linking to the issue you had opened in step 1

Your contribution will be checked and merged into the original repository. You will be contacted if there is any problem in your contribution

Make sure to include the following information in your pull request:

* **Code** which you are contributing to this package

* **Documentation** of this code if it provides new functionality. This should be a
  description of new functionality added to the [docs](https://solislemuslab.github.io/BayesianNetworkRegression.jl/dev/). Check out the [docs folder](https://github.com/samozm/BayesianNetworkRegression.jl/tree/main/docs) for instructions on how to update the documentation.

- **Tests** of this code to make sure that the previously failed function or the new functionality now works properly


---

_These Contributing Guidelines have been adapted from the [Contributing Guidelines](https://github.com/atomneb/AtomNeb-py/blob/master/CONTRIBUTING.md) of [The Turing Way](https://github.com/atomneb/AtomNeb-py)! (License: MIT)_