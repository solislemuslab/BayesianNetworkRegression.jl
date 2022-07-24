# Steps to check the testing errors

1. Git pull the `develop` branch:
```
git checkout -b develop
git pull origin develop
```
I had to fix two conflicts with the current status of master (logo.png and readme).

2. In julia
```julia
]
dev PATH/BayesianNetworkRegression.jl
```

```
(@v1.8) pkg> status
Status `~/.julia/environments/v1.8/Project.toml`
  [69666777] Arrow v2.3.0
  [9eeb066a] BayesianNetworkRegression v0.1.0 `~/Dropbox/Documents/solislemus-lab/lab-members/grad-student-projects/sam-projects/BayesianNetworkRegression.jl`
  [336ed68f] CSV v0.10.4
  [8f4d0f93] Conda v1.7.0
  [a93c6f00] DataFrames v1.3.4
  [7073ff75] IJulia v1.23.3
  [524e6230] IntervalTrees v1.0.0
  [33ad39ac] PhyloNetworks v0.15.0
  [c0d5b6db] PhyloPlots v0.3.2
  [438e738f] PyCall v1.93.1
  [6f49c342] RCall v0.13.13
  [bd369af6] Tables v1.7.0
```

## Running the test within julia:

This takes more than 2 hours:
```julia
(@v1.8) pkg> test BayesianNetworkRegression
     Testing BayesianNetworkRegression
      Status `/private/var/folders/mx/r17jmtg12315_ptcj74tkqq80000gq/T/jl_D70okr/Project.toml`
  [9eeb066a] BayesianNetworkRegression v0.1.0 `~/Dropbox/Documents/solislemus-lab/lab-members/grad-student-projects/sam-projects/BayesianNetworkRegression.jl`
  [336ed68f] CSV v0.10.4
  [a93c6f00] DataFrames v1.3.4
⌃ [31c24e10] Distributions v0.25.62
  [43dcc890] GaussianDistributions v0.5.2
  [41ab1584] InvertedIndices v1.1.0
  [8e2b3108] KahanSummation v0.3.0
  [2ab3a3ac] LogExpFunctions v0.3.15
  [c7f686f2] MCMCChains v5.3.1
  [be115224] MCMCDiagnosticTools v0.1.3
  [33e6dc65] MKL v0.5.0
  [92933f4c] ProgressMeter v1.7.2
⌃ [90137ffa] StaticArrays v1.5.0
⌃ [2913bbd2] StatsBase v0.33.16
  [9d95f2ec] TypedTables v1.4.0
  [8ba89e20] Distributed `@stdlib/Distributed`
  [b77e0a4c] InteractiveUtils `@stdlib/InteractiveUtils`
  [37e2e46d] LinearAlgebra `@stdlib/LinearAlgebra`
  [9a3f8284] Random `@stdlib/Random`
  [1a1011a3] SharedArrays `@stdlib/SharedArrays`
  [10745b16] Statistics `@stdlib/Statistics`
  [8dfed614] Test `@stdlib/Test`
      Status `/private/var/folders/mx/r17jmtg12315_ptcj74tkqq80000gq/T/jl_D70okr/Manifest.toml`
  [621f4979] AbstractFFTs v1.2.1
  [80f14c24] AbstractMCMC v4.1.3
⌅ [1520ce14] AbstractTrees v0.3.4
  [79e6a3ab] Adapt v3.3.3
  [dce04be8] ArgCheck v2.3.0
  [13072b0f] AxisAlgorithms v1.0.1
  [39de3d68] AxisArrays v0.4.6
  [198e06fe] BangBang v0.3.36
  [9718e550] Baselet v0.1.1
  [9eeb066a] BayesianNetworkRegression v0.1.0 `~/Dropbox/Documents/solislemus-lab/lab-members/grad-student-projects/sam-projects/BayesianNetworkRegression.jl`
  [336ed68f] CSV v0.10.4
  [49dc2e85] Calculus v0.5.1
⌃ [d360d2e6] ChainRulesCore v1.15.0
  [9e997f8a] ChangesOfVariables v0.1.3
  [944b1d66] CodecZlib v0.7.0
⌅ [34da2185] Compat v3.45.0
  [a33af91c] CompositionsBase v0.1.1
  [88cd18e8] ConsoleProgressMonitor v0.1.2
  [187b0558] ConstructionBase v1.4.0
  [a8cc5b0e] Crayons v4.1.1
  [9a962f9c] DataAPI v1.10.0
  [a93c6f00] DataFrames v1.3.4
  [864edb3b] DataStructures v0.18.13
  [e2d170a0] DataValueInterfaces v1.0.0
  [244e2a9f] DefineSingletons v0.1.2
  [b429d917] DensityInterface v0.4.0
  [85a47980] Dictionaries v0.3.21
⌃ [31c24e10] Distributions v0.25.62
⌅ [ffbed154] DocStringExtensions v0.8.6
  [fa6b7ba4] DualNumbers v0.6.8
  [7a1cc6ca] FFTW v1.5.0
  [48062228] FilePathsBase v0.9.18
  [1a297f60] FillArrays v0.13.2
  [59287772] Formatting v0.4.2
  [43dcc890] GaussianDistributions v0.5.2
⌃ [34004b35] HypergeometricFunctions v0.3.10
  [313cdc1a] Indexing v1.1.1
  [22cec73e] InitialValues v0.3.1
⌃ [842dd82b] InlineStrings v1.1.2
⌅ [a98d9a8b] Interpolations v0.13.6
  [8197267c] IntervalSets v0.7.1
⌃ [3587e190] InverseFunctions v0.1.6
  [41ab1584] InvertedIndices v1.1.0
  [92d709cd] IrrationalConstants v0.1.1
  [c8e1da08] IterTools v1.4.0
  [82899510] IteratorInterfaceExtensions v1.0.0
  [692b3bcd] JLLWrappers v1.4.1
  [8e2b3108] KahanSummation v0.3.0
  [5ab0869b] KernelDensity v0.6.3
⌅ [1d6d02ad] LeftChildRightSiblingTrees v0.1.3
  [2ab3a3ac] LogExpFunctions v0.3.15
  [e6f89c97] LoggingExtras v0.4.9
  [c7f686f2] MCMCChains v5.3.1
  [be115224] MCMCDiagnosticTools v0.1.3
  [33e6dc65] MKL v0.5.0
  [e80e1ace] MLJModelInterface v1.6.0
  [1914dd2f] MacroTools v0.5.9
  [128add7d] MicroCollections v0.1.2
  [e1d29d7a] Missings v1.0.2
⌃ [77ba4419] NaNMath v1.0.0
  [c020b1a1] NaturalSort v1.0.0
  [6fe1bfb0] OffsetArrays v1.12.7
  [bac558e1] OrderedCollections v1.4.1
⌃ [90014a1f] PDMats v0.11.14
⌃ [69de0a69] Parsers v2.3.1
  [2dfb63ee] PooledArrays v1.4.2
  [21216c6a] Preferences v1.3.0
  [08abe8d2] PrettyTables v1.3.1
  [33c8b6b6] ProgressLogging v0.1.4
  [92933f4c] ProgressMeter v1.7.2
  [1fd47b50] QuadGK v2.4.2
  [b3c3ace0] RangeArrays v0.3.2
  [c84ed2f1] Ratios v0.4.3
  [3cdcf5f2] RecipesBase v1.2.1
  [189a3867] Reexport v1.2.2
  [ae029012] Requires v1.3.0
  [79098fc4] Rmath v0.7.0
  [30f210dd] ScientificTypesBase v3.0.0
  [91c51154] SentinelArrays v1.3.13
⌅ [efcf1570] Setfield v0.8.2
  [a2af1166] SortingAlgorithms v1.0.1
⌃ [276daf66] SpecialFunctions v2.1.6
  [03a91e81] SplitApplyCombine v1.2.2
  [171d559e] SplittablesBase v0.1.14
⌃ [90137ffa] StaticArrays v1.5.0
⌃ [1e83bf80] StaticArraysCore v1.0.0
  [64bff920] StatisticalTraits v3.2.0
⌃ [82ae8749] StatsAPI v1.3.0
⌃ [2913bbd2] StatsBase v0.33.16
  [4c63d2b9] StatsFuns v1.0.1
  [3783bdb8] TableTraits v1.0.1
  [bd369af6] Tables v1.7.0
  [5d786b92] TerminalLoggers v0.1.5
  [3bb67fe8] TranscodingStreams v0.9.6
  [28d57a85] Transducers v0.4.73
  [9d95f2ec] TypedTables v1.4.0
  [ea10d353] WeakRefStrings v1.4.2
  [efce3f68] WoodburyMatrices v0.5.5
  [700de1a5] ZygoteRules v0.2.2
  [f5851436] FFTW_jll v3.3.10+0
  [1d5cc7b8] IntelOpenMP_jll v2018.0.3+2
  [856f044c] MKL_jll v2022.0.0+0
  [efe28fd5] OpenSpecFun_jll v0.5.5+0
  [f50d1b31] Rmath_jll v0.3.0+0
  [0dad84c5] ArgTools v1.1.1 `@stdlib/ArgTools`
  [56f22d72] Artifacts `@stdlib/Artifacts`
  [2a0f44e3] Base64 `@stdlib/Base64`
  [ade2ca70] Dates `@stdlib/Dates`
  [8bb1440f] DelimitedFiles `@stdlib/DelimitedFiles`
  [8ba89e20] Distributed `@stdlib/Distributed`
  [f43a241f] Downloads v1.6.0 `@stdlib/Downloads`
  [7b1f6079] FileWatching `@stdlib/FileWatching`
  [9fa8497b] Future `@stdlib/Future`
  [b77e0a4c] InteractiveUtils `@stdlib/InteractiveUtils`
  [4af54fe1] LazyArtifacts `@stdlib/LazyArtifacts`
  [b27032c2] LibCURL v0.6.3 `@stdlib/LibCURL`
  [76f85450] LibGit2 `@stdlib/LibGit2`
  [8f399da3] Libdl `@stdlib/Libdl`
  [37e2e46d] LinearAlgebra `@stdlib/LinearAlgebra`
  [56ddb016] Logging `@stdlib/Logging`
  [d6f4376e] Markdown `@stdlib/Markdown`
  [a63ad114] Mmap `@stdlib/Mmap`
  [ca575930] NetworkOptions v1.2.0 `@stdlib/NetworkOptions`
  [44cfe95a] Pkg v1.8.0 `@stdlib/Pkg`
  [de0858da] Printf `@stdlib/Printf`
  [3fa0cd96] REPL `@stdlib/REPL`
  [9a3f8284] Random `@stdlib/Random`
  [ea8e919c] SHA v0.7.0 `@stdlib/SHA`
  [9e88b42a] Serialization `@stdlib/Serialization`
  [1a1011a3] SharedArrays `@stdlib/SharedArrays`
  [6462fe0b] Sockets `@stdlib/Sockets`
  [2f01184e] SparseArrays `@stdlib/SparseArrays`
  [10745b16] Statistics `@stdlib/Statistics`
  [4607b0f0] SuiteSparse `@stdlib/SuiteSparse`
  [fa267f1f] TOML v1.0.0 `@stdlib/TOML`
  [a4e569a6] Tar v1.10.0 `@stdlib/Tar`
  [8dfed614] Test `@stdlib/Test`
  [cf7118a7] UUIDs `@stdlib/UUIDs`
  [4ec0a83e] Unicode `@stdlib/Unicode`
  [e66e0078] CompilerSupportLibraries_jll v0.5.2+0 `@stdlib/CompilerSupportLibraries_jll`
  [deac9b47] LibCURL_jll v7.81.0+0 `@stdlib/LibCURL_jll`
  [29816b5a] LibSSH2_jll v1.10.2+0 `@stdlib/LibSSH2_jll`
  [c8ffd9c3] MbedTLS_jll v2.28.0+0 `@stdlib/MbedTLS_jll`
  [14a3606d] MozillaCACerts_jll v2022.2.1 `@stdlib/MozillaCACerts_jll`
  [4536629a] OpenBLAS_jll v0.3.20+0 `@stdlib/OpenBLAS_jll`
  [05823500] OpenLibm_jll v0.8.1+0 `@stdlib/OpenLibm_jll`
  [83775a58] Zlib_jll v1.2.12+3 `@stdlib/Zlib_jll`
  [8e850b90] libblastrampoline_jll v5.1.0+0 `@stdlib/libblastrampoline_jll`
  [8e850ede] nghttp2_jll v1.41.0+1 `@stdlib/nghttp2_jll`
  [3f19e933] p7zip_jll v17.4.0+0 `@stdlib/p7zip_jll`
        Info Packages marked with ⌃ and ⌅ have new versions available, but those with ⌅ cannot be upgraded. To see why use `status --outdated`
Precompiling project...
  31 dependencies successfully precompiled in 26 seconds. 80 already precompiled.
     Testing Running tests...
Julia Version 1.8.0-rc1
Commit 6368fdc6565 (2022-05-27 18:33 UTC)
Platform Info:
  OS: macOS (x86_64-apple-darwin21.4.0)
  CPU: 8 × Intel(R) Core(TM) i7-8569U CPU @ 2.80GHz
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-13.0.1 (ORCJIT, skylake)
  Threads: 1 on 8 virtual cores
Environment:
  JULIA_LOAD_PATH = @:/var/folders/mx/r17jmtg12315_ptcj74tkqq80000gq/T/jl_D70okr
nothing
LBTConfig([ILP64] libopenblas64_.0.3.20.dylib)
BLAS.get_config() = LBTConfig([ILP64] libopenblas64_.0.3.20.dylib)
Sys.isapple() = true
┌ Warning: `dim(a::AbstractMatrix)` is deprecated, use `LinearAlgebra.checksquare(a)` instead.
│   caller = InverseWishart at inversewishart.jl:32 [inlined]
└ @ Core ~/.julia/packages/Distributions/O5xl5/src/matrix/inversewishart.jl:32
┌ Warning: `dim(a::AbstractMatrix)` is deprecated, use `LinearAlgebra.checksquare(a)` instead.
│   caller = invwishart_logc0(df::Float64, Ψ::PDMats.PDMat{Float64, Matrix{Float64}}) at inversewishart.jl:112
└ @ Distributions ~/.julia/packages/Distributions/O5xl5/src/matrix/inversewishart.jl:112
┌ Warning: `dim(a::AbstractMatrix)` is deprecated, use `LinearAlgebra.checksquare(a)` instead.
│   caller = dim at inversewishart.jl:77 [inlined]
└ @ Core ~/.julia/packages/Distributions/O5xl5/src/matrix/inversewishart.jl:77
┌ Warning: `dim(a::AbstractMatrix)` is deprecated, use `LinearAlgebra.checksquare(a)` instead.
│   caller = dim at inversewishart.jl:77 [inlined]
└ @ Core ~/.julia/packages/Distributions/O5xl5/src/matrix/inversewishart.jl:77
┌ Warning: `dim(a::AbstractMatrix)` is deprecated, use `LinearAlgebra.checksquare(a)` instead.
│   caller = Wishart(df::Float64, S::PDMats.PDMat{Float64, Matrix{Float64}}) at wishart.jl:47
└ @ Distributions ~/.julia/packages/Distributions/O5xl5/src/matrix/wishart.jl:47
┌ Warning: `dim(a::AbstractMatrix)` is deprecated, use `LinearAlgebra.checksquare(a)` instead.
│   caller = wishart_logc0 at wishart.jl:151 [inlined]
└ @ Core ~/.julia/packages/Distributions/O5xl5/src/matrix/wishart.jl:151
┌ Warning: `dim(a::AbstractMatrix)` is deprecated, use `LinearAlgebra.checksquare(a)` instead.
│   caller = MvNormal at mvnormal.jl:186 [inlined]
└ @ Core ~/.julia/packages/Distributions/O5xl5/src/multivariate/mvnormal.jl:186
┌ Warning: `dim(a::AbstractMatrix)` is deprecated, use `LinearAlgebra.checksquare(a)` instead.
│   caller = MvNormal at mvnormal.jl:186 [inlined]
└ @ Core ~/.julia/packages/Distributions/O5xl5/src/multivariate/mvnormal.jl:186
Test Summary:          | Pass  Total  Time
InitTests - Dimensions |    8      8  5.1s
┌ Warning: `dim(a::AbstractMatrix)` is deprecated, use `LinearAlgebra.checksquare(a)` instead.
│   caller = MvNormal at mvnormal.jl:186 [inlined]
└ @ Core ~/.julia/packages/Distributions/O5xl5/src/multivariate/mvnormal.jl:186
┌ Warning: `dim(a::AbstractMatrix)` is deprecated, use `LinearAlgebra.checksquare(a)` instead.
│   caller = MvNormal at mvnormal.jl:186 [inlined]
└ @ Core ~/.julia/packages/Distributions/O5xl5/src/multivariate/mvnormal.jl:186
┌ Warning: `dim(a::AbstractMatrix)` is deprecated, use `LinearAlgebra.checksquare(a)` instead.
│   caller = MvNormal at mvnormal.jl:186 [inlined]
└ @ Core ~/.julia/packages/Distributions/O5xl5/src/multivariate/mvnormal.jl:186
┌ Warning: `dim(a::AbstractMatrix)` is deprecated, use `LinearAlgebra.checksquare(a)` instead.
│   caller = MvNormal at mvnormal.jl:186 [inlined]
└ @ Core ~/.julia/packages/Distributions/O5xl5/src/multivariate/mvnormal.jl:186
Progress: 100%|███████████████████████████████████████████████████████████████████████████████████████████| Time: 0:00:19 ( 6.67  s/it)
Test Summary:   | Pass  Total   Time
Dimension tests |    3      3  35.4s
┌ Warning: `dim(a::AbstractMatrix)` is deprecated, use `LinearAlgebra.checksquare(a)` instead.
│   caller = MvNormal at mvnormal.jl:186 [inlined]
└ @ Core ~/.julia/packages/Distributions/O5xl5/src/multivariate/mvnormal.jl:186
Progress: 100%|███████████████████████████████████████████████████████████████████████████████████████████| Time: 0:48:37 ( 5.85  s/it)
seed = 2358
size(γ_sorted) = (20000, 435, 1)
      From worker 2:	┌ Warning: `dim(a::AbstractMatrix)` is deprecated, use `LinearAlgebra.checksquare(a)` instead.
      From worker 2:	│   caller = InverseWishart at inversewishart.jl:32 [inlined]
      From worker 2:	└ @ Core ~/.julia/packages/Distributions/O5xl5/src/matrix/inversewishart.jl:32
      From worker 2:	┌ Warning: `dim(a::AbstractMatrix)` is deprecated, use `LinearAlgebra.checksquare(a)` instead.
      From worker 2:	│   caller = invwishart_logc0(df::Float64, Ψ::PDMats.PDMat{Float64, Matrix{Float64}}) at inversewishart.jl:112
      From worker 2:	└ @ Distributions ~/.julia/packages/Distributions/O5xl5/src/matrix/inversewishart.jl:112
      From worker 2:	┌ Warning: `dim(a::AbstractMatrix)` is deprecated, use `LinearAlgebra.checksquare(a)` instead.
      From worker 2:	│   caller = dim at inversewishart.jl:77 [inlined]
      From worker 2:	└ @ Core ~/.julia/packages/Distributions/O5xl5/src/matrix/inversewishart.jl:77
      From worker 2:	┌ Warning: `dim(a::AbstractMatrix)` is deprecated, use `LinearAlgebra.checksquare(a)` instead.
      From worker 2:	│   caller = dim at inversewishart.jl:77 [inlined]
      From worker 2:	└ @ Core ~/.julia/packages/Distributions/O5xl5/src/matrix/inversewishart.jl:77
      From worker 2:	┌ Warning: `dim(a::AbstractMatrix)` is deprecated, use `LinearAlgebra.checksquare(a)` instead.
      From worker 2:	│   caller = Wishart(df::Float64, S::PDMats.PDMat{Float64, Matrix{Float64}}) at wishart.jl:47
      From worker 2:	└ @ Distributions ~/.julia/packages/Distributions/O5xl5/src/matrix/wishart.jl:47
      From worker 2:	┌ Warning: `dim(a::AbstractMatrix)` is deprecated, use `LinearAlgebra.checksquare(a)` instead.
      From worker 2:	│   caller = wishart_logc0 at wishart.jl:151 [inlined]
      From worker 2:	└ @ Core ~/.julia/packages/Distributions/O5xl5/src/matrix/wishart.jl:151
      From worker 2:	┌ Warning: `dim(a::AbstractMatrix)` is deprecated, use `LinearAlgebra.checksquare(a)` instead.
      From worker 2:	│   caller = MvNormal at mvnormal.jl:186 [inlined]
      From worker 2:	└ @ Core ~/.julia/packages/Distributions/O5xl5/src/multivariate/mvnormal.jl:186
      From worker 2:	┌ Warning: `dim(a::AbstractMatrix)` is deprecated, use `LinearAlgebra.checksquare(a)` instead.
      From worker 2:	│   caller = MvNormal at mvnormal.jl:186 [inlined]
      From worker 2:	└ @ Core ~/.julia/packages/Distributions/O5xl5/src/multivariate/mvnormal.jl:186
      From worker 2:	┌ Warning: `dim(a::AbstractMatrix)` is deprecated, use `LinearAlgebra.checksquare(a)` instead.
      From worker 2:	│   caller = MvNormal at mvnormal.jl:186 [inlined]
      From worker 2:	└ @ Core ~/.julia/packages/Distributions/O5xl5/src/multivariate/mvnormal.jl:186
      From worker 2:	┌ Warning: `dim(a::AbstractMatrix)` is deprecated, use `LinearAlgebra.checksquare(a)` instead.
      From worker 2:	│   caller = MvNormal at mvnormal.jl:186 [inlined]
      From worker 2:	└ @ Core ~/.julia/packages/Distributions/O5xl5/src/multivariate/mvnormal.jl:186
      From worker 2:	┌ Warning: `dim(a::AbstractMatrix)` is deprecated, use `LinearAlgebra.checksquare(a)` instead.
      From worker 2:	│   caller = MvNormal at mvnormal.jl:186 [inlined]
      From worker 2:	└ @ Core ~/.julia/packages/Distributions/O5xl5/src/multivariate/mvnormal.jl:186
Progress: 100%|███████████████████████████████████████████████████████████████████████████████████████████| Time: 0:46:25 ( 5.58  s/it)
seed = 2358
size(γ_sorted2) = (20000, 435, 1)
DataFrame(loc = (mean(result2.state.ξ[nburn + 1:total, :, :], dims = 1))[1, :], worker = (mean(result22.state.ξ[nburn + 1:total, :, :], dims = 1))[1, :], real = nodes_res[:, "Xi posterior"]) = 30×3 DataFrame
 Row │ loc      worker   real
     │ Float64  Float64  Float64
─────┼───────────────────────────
   1 │ 0.50945  0.50945  0.50945
   2 │ 0.49975  0.49975  0.49975
   3 │ 0.5171   0.5171   0.5171
   4 │ 0.4972   0.4972   0.4972
   5 │ 0.4878   0.4878   0.4878
   6 │ 0.5022   0.5022   0.5022
   7 │ 0.4831   0.4831   0.4831
   8 │ 0.4727   0.4727   0.4727
   9 │ 0.535    0.535    0.535
  10 │ 0.49015  0.49015  0.49015
  11 │ 0.48755  0.48755  0.48755
  12 │ 0.4673   0.4673   0.4673
  13 │ 0.4763   0.4763   0.4763
  14 │ 0.50575  0.50575  0.50575
  15 │ 0.50365  0.50365  0.50365
  16 │ 0.5061   0.5061   0.5061
  17 │ 0.4654   0.4654   0.4654
  18 │ 0.49725  0.49725  0.49725
  19 │ 0.4803   0.4803   0.4803
  20 │ 0.48315  0.48315  0.48315
  21 │ 0.4843   0.4843   0.4843
  22 │ 0.4818   0.4818   0.4818
  23 │ 0.48135  0.48135  0.48135
  24 │ 0.51985  0.51985  0.51985
  25 │ 0.49245  0.49245  0.49245
  26 │ 0.49505  0.49505  0.49505
  27 │ 0.4935   0.4935   0.4935
  28 │ 0.47745  0.47745  0.47745
  29 │ 0.55765  0.55765  0.55765
  30 │ 0.49595  0.49595  0.49595
Test Summary:         | Pass  Total      Time
Result tests - master |    8      8  95m36.9s
Progress: 100%|███████████████████████████████████████████████████████████████████████████████████████████| Time: 0:46:18 ( 5.57  s/it)
Progress: 100%|███████████████████████████████████████████████████████████████████████████████████████████| Time: 0:47:04 ( 5.66  s/it)
DataFrame(loc = (mean(result3.state.ξ[nburn + 1:total, :, :], dims = 1))[1, :], real = nodes_res3[:, "Xi posterior"], master = (mean(result33.state.ξ[nburn + 1:total, :, :], dims = 1))[1, :]) = 30×3 DataFrame
 Row │ loc      real     master
     │ Float64  Float64  Float64
─────┼───────────────────────────
   1 │ 0.99775  0.99845  0.99775
   2 │ 1.0      1.0      1.0
   3 │ 1.0      1.0      1.0
   4 │ 1.0      1.0      1.0
   5 │ 0.6186   0.5932   0.6186
   6 │ 0.9991   0.99995  0.9991
   7 │ 1.0      1.0      1.0
   8 │ 0.64845  0.63035  0.64845
   9 │ 1.0      1.0      1.0
  10 │ 0.53755  0.48335  0.53755
  11 │ 1.0      1.0      1.0
  12 │ 0.63095  0.61875  0.63095
  13 │ 0.60365  0.5826   0.60365
  14 │ 1.0      1.0      1.0
  15 │ 1.0      1.0      1.0
  16 │ 1.0      1.0      1.0
  17 │ 0.45935  0.44705  0.45935
  18 │ 0.59905  0.5713   0.59905
  19 │ 1.0      1.0      1.0
  20 │ 0.9998   1.0      0.9998
  21 │ 0.5681   0.53835  0.5681
  22 │ 0.5856   0.5749   0.5856
  23 │ 0.99975  1.0      0.99975
  24 │ 1.0      1.0      1.0
  25 │ 0.5024   0.48875  0.5024
  26 │ 1.0      1.0      1.0
  27 │ 0.99995  1.0      0.99995
  28 │ 1.0      1.0      1.0
  29 │ 1.0      1.0      1.0
  30 │ 1.0      1.0      1.0
Result tests - worker: Test Failed at /Users/useradmin/Dropbox/Documents/solislemus-lab/lab-members/grad-student-projects/sam-projects/BayesianNetworkRegression.jl/test/runtests.jl:295
  Expression: isapprox((mean(result3.state.γ[nburn + 1:total, :, :], dims = 1))[1, :], edges_res3.mean)
   Evaluated: isapprox([2.250680999802882, 2.3238595666748587, 2.4585828906863427, -0.23040558281101603, 1.8661037522223252, 2.1859332835468033, -0.06381318102067537, 3.1896043827294567, 0.03105085032630252, 2.1026620047757327  …  2.4973857615546557, 1.782393154516049, 4.186198965369134, 2.5701085059257887, 1.9676811448855926, 3.6024886616456615, 2.436392253164013, 3.767763605275002, 1.990956043816715, 3.7752428522047934], [2.4204138236227823, 2.4544310282913817, 2.615027005327588, -0.20209427807363955, 1.9545007965780903, 2.3632716687444133, -0.08884169267151838, 3.4189163289504862, 0.04376711699482108, 2.298728515796466  …  2.503027484978233, 1.8165260874608502, 4.163769413802103, 2.6398087662989127, 1.9688029087593653, 3.58002016007935, 2.5004819721005864, 3.7185806477885968, 2.0808080074222017, 3.8514477256028963])
Stacktrace:
 [1] macro expansion
   @ ~/.julia/juliaup/julia-1.8.0-rc1+0~x64/share/julia/stdlib/v1.8/Test/src/Test.jl:464 [inlined]
 [2] macro expansion
   @ ~/Dropbox/Documents/solislemus-lab/lab-members/grad-student-projects/sam-projects/BayesianNetworkRegression.jl/test/runtests.jl:295 [inlined]
 [3] macro expansion
   @ ~/.julia/juliaup/julia-1.8.0-rc1+0~x64/share/julia/stdlib/v1.8/Test/src/Test.jl:1357 [inlined]
 [4] top-level scope
   @ ~/Dropbox/Documents/solislemus-lab/lab-members/grad-student-projects/sam-projects/BayesianNetworkRegression.jl/test/runtests.jl:207
Result tests - worker: Test Failed at /Users/useradmin/Dropbox/Documents/solislemus-lab/lab-members/grad-student-projects/sam-projects/BayesianNetworkRegression.jl/test/runtests.jl:296
  Expression: isapprox(ci_df3[:, "0.025"], edges_res3[:, "0.025"])
   Evaluated: isapprox([-0.5149151416405771, -0.09764165640739408, -0.2626785801117093, -2.4110201912910028, -0.49337593558646375, -0.48825297364898157, -2.212016234146599, 0.21774147669825317, -2.081425749065621, -0.49801366521701596  …  0.313326745190875, -0.677074614358335, 2.2490204061661188, 0.2309009624238405, -0.10429555714656868, 1.1821886548531828, 0.45153696581214176, 1.5355370696865176, -0.2906686398784787, 1.146781140458779], [-0.3253468835375697, 0.03456576397813649, -0.13727617500611067, -2.3130924911316164, -0.356571299293175, -0.24584651451679163, -2.1411421415118954, 0.413661316258787, -2.0214674783329394, -0.33902945040302396  …  0.24648664619520755, -0.5583554899309382, 2.2047140883634286, 0.3726351301296593, -0.07779661480915934, 1.1514151730452964, 0.5798336184332088, 1.502497053304905, -0.14084919637617999, 1.429356916615177])
Stacktrace:
 [1] macro expansion
   @ ~/.julia/juliaup/julia-1.8.0-rc1+0~x64/share/julia/stdlib/v1.8/Test/src/Test.jl:464 [inlined]
 [2] macro expansion
   @ ~/Dropbox/Documents/solislemus-lab/lab-members/grad-student-projects/sam-projects/BayesianNetworkRegression.jl/test/runtests.jl:296 [inlined]
 [3] macro expansion
   @ ~/.julia/juliaup/julia-1.8.0-rc1+0~x64/share/julia/stdlib/v1.8/Test/src/Test.jl:1357 [inlined]
 [4] top-level scope
   @ ~/Dropbox/Documents/solislemus-lab/lab-members/grad-student-projects/sam-projects/BayesianNetworkRegression.jl/test/runtests.jl:207
Result tests - worker: Test Failed at /Users/useradmin/Dropbox/Documents/solislemus-lab/lab-members/grad-student-projects/sam-projects/BayesianNetworkRegression.jl/test/runtests.jl:297
  Expression: isapprox(ci_df3[:, "0.975"], edges_res3[:, "0.975"])
   Evaluated: isapprox([5.004287971684317, 4.57384097170168, 5.339650880683134, 1.9282860640963724, 4.610877474099999, 4.789830040967984, 2.3362282481423247, 6.0053711320630265, 2.36119258031042, 4.7922795436593315  …  4.8566970854071645, 3.781776771649758, 6.217604513685389, 4.930911389582056, 4.055942631059299, 5.965730505584005, 4.510710858518116, 6.2700132451907375, 4.088022854916261, 6.173192662191308], [5.035357583510649, 4.649649263876709, 5.4288961650312535, 1.9281712262046373, 4.570145552566375, 4.967990639370721, 2.1790725464365526, 6.216630692935116, 2.275588134632959, 4.938705548683524  …  4.808089178173465, 3.8006834891958547, 6.136917035196203, 4.909378979755079, 3.9598598404382694, 5.920149377449734, 4.574941150563422, 6.2085874082054655, 4.154605282492215, 6.237668798309739])
Stacktrace:
 [1] macro expansion
   @ ~/.julia/juliaup/julia-1.8.0-rc1+0~x64/share/julia/stdlib/v1.8/Test/src/Test.jl:464 [inlined]
 [2] macro expansion
   @ ~/Dropbox/Documents/solislemus-lab/lab-members/grad-student-projects/sam-projects/BayesianNetworkRegression.jl/test/runtests.jl:297 [inlined]
 [3] macro expansion
   @ ~/.julia/juliaup/julia-1.8.0-rc1+0~x64/share/julia/stdlib/v1.8/Test/src/Test.jl:1357 [inlined]
 [4] top-level scope
   @ ~/Dropbox/Documents/solislemus-lab/lab-members/grad-student-projects/sam-projects/BayesianNetworkRegression.jl/test/runtests.jl:207
Result tests - worker: Test Failed at /Users/useradmin/Dropbox/Documents/solislemus-lab/lab-members/grad-student-projects/sam-projects/BayesianNetworkRegression.jl/test/runtests.jl:301
  Expression: xis == ones(30)
   Evaluated: [0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0  …  0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0] == [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0  …  1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
Stacktrace:
 [1] macro expansion
   @ ~/.julia/juliaup/julia-1.8.0-rc1+0~x64/share/julia/stdlib/v1.8/Test/src/Test.jl:464 [inlined]
 [2] macro expansion
   @ ~/Dropbox/Documents/solislemus-lab/lab-members/grad-student-projects/sam-projects/BayesianNetworkRegression.jl/test/runtests.jl:301 [inlined]
 [3] macro expansion
   @ ~/.julia/juliaup/julia-1.8.0-rc1+0~x64/share/julia/stdlib/v1.8/Test/src/Test.jl:1357 [inlined]
 [4] top-level scope
   @ ~/Dropbox/Documents/solislemus-lab/lab-members/grad-student-projects/sam-projects/BayesianNetworkRegression.jl/test/runtests.jl:207
Test Summary:         | Pass  Fail  Total      Time
Result tests - worker |    3     4      7  93m51.4s
ERROR: LoadError: Some tests did not pass: 3 passed, 4 failed, 0 errored, 0 broken.
in expression starting at /Users/useradmin/Dropbox/Documents/solislemus-lab/lab-members/grad-student-projects/sam-projects/BayesianNetworkRegression.jl/test/runtests.jl:206
ERROR: Package BayesianNetworkRegression errored during testing
```


# Other tests

Doug has a test to check if replicates from single thread or multiple threads are the same:
```julia
## Test from MixedModels.jl/utilities.jl
using LinearAlgebra
using MixedModels
using StableRNGs
using SparseArrays
using Test
using StatsModels

@testset "threaded_replicate" begin
	rng = StableRNG(42);
	single_thread = replicate(10;use_threads=false) do; only(randn(rng, 1)) ; end
	rng = StableRNG(42);
	multi_thread = replicate(10;use_threads=true) do
		if Threads.threadid() % 2 == 0
			sleep(0.001)
		end
		r = only(randn(rng, 1));
	end

	@test all(sort!(single_thread) .≈ sort!(multi_thread))
end
```

Very interesting [thread](https://discourse.julialang.org/t/random-numbers-and-threads/77364/19):

```julia
using Random

println("no spawning")
Random.seed!(0)
@show rand()
println("no spawning")
Random.seed!(0)
@show rand()
println("spawning once from main")
Random.seed!(0)
@sync begin
    Threads.@spawn nothing
end
@show rand()

println("spawning once from main")
Random.seed!(0)
@sync begin
    Threads.@spawn nothing
end
@show rand()
println("spawning twice from main")
Random.seed!(0)
@sync begin
    Threads.@spawn nothing
    Threads.@spawn nothing
end
@show rand()

println("spawning once from child")
task = Task() do 
    Threads.@spawn nothing
    rand()
end
Random.seed!(0)
@sync schedule(task)

@show rand()
```

From the link, "f I spawn things in my main task, its rng state changes. If I spawn things in another task, that other tasks rng state changes, but the main task rng state is untouched."
```
no spawning
rand() = 0.4056994708920292
no spawning
rand() = 0.4056994708920292
spawning once from main
rand() = 0.6616126907308237
spawning once from main
rand() = 0.6616126907308237
spawning twice from main
rand() = 0.2895098423219379
spawning once from child
rand() = 0.4056994708920292
0.4056994708920292
```

```
 -0.0   0.403434   0.415839    0.0
 -0.0   0.711698   0.299316    0.0
  0.0  -0.589814  -0.776966   -0.0
 -0.0   1.58773    0.618515    0.0
 -0.0   0.682053   0.312038    0.0
  0.0  -0.895464  -1.1891     -0.0
  0.0  -0.236236   0.0635545   0.0
```

# Sam notes

Pkg mode:

```
julia -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.test("BayesianNetworkRegression")'
```


Script directly:

```
julia test/runtests.jl
```

To test writing to file:
R=5
η=1.01
ζ=1.0
ι=1.0
aΔ=1.0
bΔ=1.0
ν=10
nburn=30000
nsamples=20000
V=0
x_transform=true
suppress_timer=false
num_chains=2
seed=nothing
full_results=false
purge_burn=nothing
filename="parameters.log"