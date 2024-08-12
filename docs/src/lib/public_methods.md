```@meta
CurrentModule = BayesianNetworkRegression
```

```@docs
Fit!(X::AbstractArray{T}, y::AbstractVector{U}, R; η=1.01, V=30, ζ=1.0, ι=1.0, aΔ=1.0, bΔ=1.0, ν=10, nburn=30000, nsamp=20000, mingen=0, maxgen=0, psrf_cutoff=1.01, x_transform=true, suppress_timer=false, num_chains=2, seed=nothing, purge_burn=nothing, filename="parameters.log") where {T,U}
```

```@docs
Summary(results::Results;interval::Int=95,digits::Int=3)
```
