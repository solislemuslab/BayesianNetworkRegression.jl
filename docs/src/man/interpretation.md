# Interpreting the Model Results

## Result format

The `Fit!` function will return a `Results` object with the following members:
- state
- rhat

### State

result.state contains the (post-burn-in) outputs of the sampling algorithm, with focus on the following two variables
-  $\xi$: a vector describing whether each node (microbe) is influential on the response. Set to 1 if the microbe is influential and 0 if it is not. 
-  $\gamma$: a vector of coefficients describing the effect of each edge (relationship) on the response. 

### $\hat{R}$

result.rhat contains r-hat statistics ([Vehtari et al. 2020](http://www.stat.columbia.edu/~gelman/research/unpublished/1903.08008.pdf)), which are used to assess whether the sampling algorithm has converged. Values close to 1 indicate convergence (Vehtari et al. suggest using a cutoff of $\hat{R} < 1.01$ to indicate convergence). Values are provided for all $\xi$ and $\gamma$ variables.

## Interpretation

### State

To aid in interpretation of results, run the following (in julia):
```julia
using Statistics
mean(result.state.ξ[:,:],dims=1)
mean(result.state.γ[:,:],dims=1)
```
The first mean calculates the posterior probability that each node is influential. 
The second mean calculates the point-estimate for the coefficients.

In order to generate (95\%) credible intervals for the coefficients, the following can be run. `nsamp` should be the number of post-burn-in samples.

```julia
using DataFrames

nsamp = 10000
γ_sorted = sort(result.γ,dims=1)
lw = convert(Int64, round(nsamp * 0.025))
hi = convert(Int64, round(nsamp * 0.975))

DataFrame(L=γ_sorted[lw,:,1],U=γ_sorted[hi,:,1])
```