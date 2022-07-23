
Pkg mode:

```
julia -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.test("BayesianNetworkRegression")'
```


Script directly:

```
julia test/runtests.jl
```