var documenterSearchIndex = {"docs":
[{"location":"man/fit_model/#Fitting-the-Model","page":"Model Fitting","title":"Fitting the Model","text":"","category":"section"},{"location":"man/fit_model/#Single-thread-run","page":"Model Fitting","title":"Single-thread run","text":"","category":"section"},{"location":"man/fit_model/","page":"Model Fitting","title":"Model Fitting","text":"The BayesianNetworkRegression.jl package implements the statistical inference method in Ozminkowski & Solís-Lemus (2022) in its main function Fit! which takes three parameters:","category":"page"},{"location":"man/fit_model/","page":"Model Fitting","title":"Model Fitting","text":"X: data matrix from microbial networks (networked covariates)\ny: response vector\nR: dimension of latent variables ","category":"page"},{"location":"man/fit_model/","page":"Model Fitting","title":"Model Fitting","text":"Quality of inference will increase with increasing R up to a point, after which it will approximately plateau. The time it takes to fit the model will increase (worse than linearly) with increasing R. For our simulations, R=7 worked best. ","category":"page"},{"location":"man/fit_model/","page":"Model Fitting","title":"Model Fitting","text":"We will use the data read in the Input Data section:","category":"page"},{"location":"man/fit_model/","page":"Model Fitting","title":"Model Fitting","text":"X_a: vector of adjacency matrices\nX_v: matrix of vectorized adjacency matrices","category":"page"},{"location":"man/fit_model/","page":"Model Fitting","title":"Model Fitting","text":"The type of input data matrix X will inform the argument x_transform so that:","category":"page"},{"location":"man/fit_model/","page":"Model Fitting","title":"Model Fitting","text":"x_transform=true means that the input matrix X needs to be vectorized (as for X_a),\nx_transform=false means that the input matrix X has already been vectorized (as for X_v).","category":"page"},{"location":"man/fit_model/","page":"Model Fitting","title":"Model Fitting","text":"To fit the model, we type this in julia:","category":"page"},{"location":"man/fit_model/","page":"Model Fitting","title":"Model Fitting","text":"using BayesianNetworkRegression\n\nresult = Fit!(X_a, y_a, 5, # we set R=5\n    nburn=200, nsamples=100, x_transform=true, \n    num_chains=1, \n    seed=1234)","category":"page"},{"location":"man/fit_model/","page":"Model Fitting","title":"Model Fitting","text":"Note that we are running a very small chain (300 generations: 200 burnin and 100 post burnin). For a real analysis, these number should be much larger (see the Simulation in the manuscript for more details).","category":"page"},{"location":"man/fit_model/","page":"Model Fitting","title":"Model Fitting","text":"The result variable is a Result type which contains five attributes:","category":"page"},{"location":"man/fit_model/","page":"Model Fitting","title":"Model Fitting","text":"state::Table: Table with all the sampled parameters for all generations.\nrhatξ::Table: Table with convergence information (hatR values) for the parameters xi representing whether specific nodes are influential or not.\nrhatγ::Table: Table with convergence information (hatR values) for the parameters gamma representing the regression coefficients.\nburn_in::Int: number of burnin samples.\nsampled::Int: number of post burnin samples.","category":"page"},{"location":"man/fit_model/","page":"Model Fitting","title":"Model Fitting","text":"Each of these objects can be accessed with a ., for example, results.state will produce the table with all the samples.","category":"page"},{"location":"man/fit_model/","page":"Model Fitting","title":"Model Fitting","text":"We use the convergence criteria proposed in Vehtari et al (2019). Values close to 1 indicate convergence. Vehtari et al. suggest using a cutoff of hatR  101 to indicate convergence. Values are provided for all xi (in rhatξ) and gamma (in rhatγ) variables.","category":"page"},{"location":"man/fit_model/","page":"Model Fitting","title":"Model Fitting","text":"These attributes are not readily interpretable, and thus, they can be summarized with the Summary function:","category":"page"},{"location":"man/fit_model/","page":"Model Fitting","title":"Model Fitting","text":"out = Summary(result)","category":"page"},{"location":"man/fit_model/","page":"Model Fitting","title":"Model Fitting","text":"Note that the show function for a Results object is already calling Summary(result) internally, and thus, if you simply call:","category":"page"},{"location":"man/fit_model/","page":"Model Fitting","title":"Model Fitting","text":"result","category":"page"},{"location":"man/fit_model/","page":"Model Fitting","title":"Model Fitting","text":"you observe the same output as when calling","category":"page"},{"location":"man/fit_model/","page":"Model Fitting","title":"Model Fitting","text":"out = Summary(result)","category":"page"},{"location":"man/fit_model/","page":"Model Fitting","title":"Model Fitting","text":"The out object is now a BNRSummary object with two main data frames:","category":"page"},{"location":"man/fit_model/","page":"Model Fitting","title":"Model Fitting","text":"DataFrame with edge coefficient point estimates and endpoints of credible intervals (default 95%):","category":"page"},{"location":"man/fit_model/","page":"Model Fitting","title":"Model Fitting","text":"julia> out.edge_coef\n435×5 DataFrame\n Row │ node1  node2  estimate  lower_bound  upper_bound \n     │ Int64  Int64  Float64   Float64      Float64     \n─────┼──────────────────────────────────────────────────\n   1 │     1      2     2.385       -1.976        6.157\n   2 │     1      3     1.823       -5.257        7.595\n   3 │     1      4     1.763       -4.514        6.009\n  ⋮  │   ⋮      ⋮       ⋮           ⋮            ⋮\n 433 │    28     29     3.51        -1.251        7.376\n 434 │    28     30     1.949       -3.425        6.135\n 435 │    29     30     2.785       -0.772        6.042\n                                        429 rows omitted","category":"page"},{"location":"man/fit_model/","page":"Model Fitting","title":"Model Fitting","text":"DataFrame with probabilities of being influencial for each node","category":"page"},{"location":"man/fit_model/","page":"Model Fitting","title":"Model Fitting","text":"julia> out.prob_nodes\n30×1 DataFrame\n Row │ probability \n     │ Float64     \n─────┼─────────────\n   1 │        0.86\n   2 │        0.92\n   3 │        0.78\n  ⋮  │      ⋮\n  28 │        0.93\n  29 │        0.89\n  30 │        0.87\n    24 rows omitted","category":"page"},{"location":"man/fit_model/","page":"Model Fitting","title":"Model Fitting","text":"The BNRSummary object also keeps the level for the credible interval, set at 95% by default:","category":"page"},{"location":"man/fit_model/","page":"Model Fitting","title":"Model Fitting","text":"julia> out.ci_level\n95","category":"page"},{"location":"man/fit_model/","page":"Model Fitting","title":"Model Fitting","text":"The level can be changed when running the Summary function:","category":"page"},{"location":"man/fit_model/","page":"Model Fitting","title":"Model Fitting","text":"out2=Summary(result,interval=97)","category":"page"},{"location":"man/fit_model/","page":"Model Fitting","title":"Model Fitting","text":"We will use these summary data frames in the Interpretation section.","category":"page"},{"location":"man/fit_model/","page":"Model Fitting","title":"Model Fitting","text":"Note that we ran the case when we need to transform the data matrix. If we already have the adjacency matrices vectorized, we simply need to set x_transform=false:","category":"page"},{"location":"man/fit_model/","page":"Model Fitting","title":"Model Fitting","text":"result2 = Fit!(X_v, y_v, 5, # we set R=5\n    nburn=200, nsamples=100, x_transform=false, \n    num_chains=1, \n    seed=1234)","category":"page"},{"location":"man/fit_model/#Multi-thread-run","page":"Model Fitting","title":"Multi-thread run","text":"","category":"section"},{"location":"man/fit_model/","page":"Model Fitting","title":"Model Fitting","text":"You can run multiple chains in parallel by setting up multiple processors as shown next.","category":"page"},{"location":"man/fit_model/","page":"Model Fitting","title":"Model Fitting","text":"using Distributed\naddprocs(2)\n\n@everywhere begin\n    using BayesianNetworkRegression,CSV,DataFrames,StaticArrays\n\n    matrix_networks = joinpath(dirname(pathof(BayesianNetworkRegression)), \"..\",\"examples\",\"matrix_networks.csv\")\n    data_in = DataFrame(CSV.File(matrix_networks))\n\n    X = Matrix(data_in[:,names(data_in,Not(\"y\"))])\n    y = Vector{Float64}(data_in[:,:y])\nend","category":"page"},{"location":"man/fit_model/","page":"Model Fitting","title":"Model Fitting","text":"Compared to the single-thread run, we only need to change num_chains=2:","category":"page"},{"location":"man/fit_model/","page":"Model Fitting","title":"Model Fitting","text":"result3 = Fit!(X, y, 5,\n    nburn=200, nsamples=100, x_transform=false, \n    num_chains=2, \n    seed=1234\n    )","category":"page"},{"location":"man/fit_model/#Error-reporting","page":"Model Fitting","title":"Error reporting","text":"","category":"section"},{"location":"man/fit_model/","page":"Model Fitting","title":"Model Fitting","text":"Please report any bugs and errors by opening an issue.","category":"page"},{"location":"man/interpretation/#Interpreting-the-Model-Results","page":"Interpretation","title":"Interpreting the Model Results","text":"","category":"section"},{"location":"man/interpretation/#Interpretation","page":"Interpretation","title":"Interpretation","text":"","category":"section"},{"location":"man/interpretation/","page":"Interpretation","title":"Interpretation","text":"From the Fit! and Summary functions in previous section, we end up with two tables summarizing the mean estimates for the regression coefficients for the edge effects (out.edge_coef) and the posterior probabilities that nodes are influential (out.prob_nodes):","category":"page"},{"location":"man/interpretation/","page":"Interpretation","title":"Interpretation","text":"julia> out.edge_coef\n435×5 DataFrame\n Row │ node1  node2  estimate  lower_bound  upper_bound \n     │ Int64  Int64  Float64   Float64      Float64     \n─────┼──────────────────────────────────────────────────\n   1 │     1      2     2.385       -1.976        6.157\n   2 │     1      3     1.823       -5.257        7.595\n   3 │     1      4     1.763       -4.514        6.009\n  ⋮  │   ⋮      ⋮       ⋮           ⋮            ⋮\n 433 │    28     29     3.51        -1.251        7.376\n 434 │    28     30     1.949       -3.425        6.135\n 435 │    29     30     2.785       -0.772        6.042\n                                        429 rows omitted\n\njulia> out.prob_nodes\n30×1 DataFrame\n Row │ probability \n     │ Float64     \n─────┼─────────────\n   1 │        0.86\n   2 │        0.92\n   3 │        0.78\n  ⋮  │      ⋮\n  28 │        0.93\n  29 │        0.89\n  30 │        0.87\n    24 rows omitted","category":"page"},{"location":"man/interpretation/#Plot","page":"Interpretation","title":"Plot","text":"","category":"section"},{"location":"man/interpretation/","page":"Interpretation","title":"Interpretation","text":"The following R code can be used to plot the credible intervals of the edge regression coefficients. ","category":"page"},{"location":"man/interpretation/","page":"Interpretation","title":"Interpretation","text":"First, we pass the data frame to R:","category":"page"},{"location":"man/interpretation/","page":"Interpretation","title":"Interpretation","text":"edges = out.edge_coef\n\nusing RCall\n@rput edges","category":"page"},{"location":"man/interpretation/","page":"Interpretation","title":"Interpretation","text":"An alternative path is to save the data frame to file with CSV.write(\"edges.csv\",out.edge_coef) and then read in R.","category":"page"},{"location":"man/interpretation/","page":"Interpretation","title":"Interpretation","text":"In R, we type","category":"page"},{"location":"man/interpretation/","page":"Interpretation","title":"Interpretation","text":"library(ggplot2)\nlibrary(tidyr)\nnn <- length(edges$node1)\nedges$edge <- 1:nn\nedges <- transform(edges,rej=ifelse(lower_bound > 0 | upper_bound < 0,TRUE,FALSE))\n\nplt <- edges %>% ggplot() + geom_errorbar(aes(x=factor(edge),ymin=lower_bound,\n                                                                ymax=upper_bound,\n                                                                color=rej)) +\n        xlab(\"\") + ylab(\"\")+ ggtitle(\"95% Credible intervals for edge effects\")+\n        scale_color_manual(values=c(\"#82AFBC\",\"#0E4B87\")) + \n        theme(\n            plot.title = element_text(hjust=0.5, size=rel(2)),\n            axis.title.x = element_text(size=rel(1.2)),\n            axis.title.y = element_text(size=rel(1.9), angle=90, vjust=0.5, hjust=0.5),\n            axis.text.x = element_text(colour=\"grey\", size=rel(1.8), hjust=.5, vjust=.5, face=\"plain\"),\n            axis.ticks.y = element_blank(),\n            panel.background = element_rect(fill = NA, color = \"black\"),\n            axis.line = element_line(colour = \"grey\"),\n            strip.text = element_text(size = rel(1.5)),\n            legend.position = \"none\"\n        ) +\n        coord_flip() + scale_x_discrete(labels=NULL,expand=expansion(add=4)) +\n        geom_point(aes(x=factor(edge),y=estimate, color=rej),shape=18, size=2) +\n        geom_hline(aes(yintercept=0),linetype=\"solid\",color=\"#696969\", size=1)\nplt","category":"page"},{"location":"man/interpretation/","page":"Interpretation","title":"Interpretation","text":"(Image: edges)","category":"page"},{"location":"man/interpretation/","page":"Interpretation","title":"Interpretation","text":"The dark credible intervals are those that do not intersect zero which point at potential edges (interactions among microbes) that have a significant effect on the response.","category":"page"},{"location":"man/interpretation/","page":"Interpretation","title":"Interpretation","text":"The following R code can be used to plot posterior probabilities of being influencial nodes (micrones) on the response.","category":"page"},{"location":"man/interpretation/","page":"Interpretation","title":"Interpretation","text":"First, we pass the data frame to R:","category":"page"},{"location":"man/interpretation/","page":"Interpretation","title":"Interpretation","text":"nodes = out.prob_nodes\n\n@rput nodes","category":"page"},{"location":"man/interpretation/","page":"Interpretation","title":"Interpretation","text":"An alternative path is to save the data frame to file with CSV.write(\"nodes.csv\",out.prob_nodes) and then read in R.","category":"page"},{"location":"man/interpretation/","page":"Interpretation","title":"Interpretation","text":"nodes$microbe <- 1:length(nodes$probability)\n\naxisTextY = \n\n## Plot:\nplt2 <- ggplot(data=nodes,aes(x=microbe,y=probability)) + \n    geom_bar(stat=\"Identity\", fill=\"#82AFBC\") + xlab(\"Microbes\") + ylab(\"\") + \n    ggtitle(\"Posterior probability of influential node\") +\n    theme(\n        plot.title = element_text(hjust=0.5, size=rel(2)),\n        axis.title.x = element_text(size=rel(1.2)),\n        axis.title.y = element_text(size=rel(1.9), angle=90, vjust=0.5, hjust=0.5),\n        axis.text.y = element_text(colour=\"grey\", size=rel(1.5), angle=0, hjust=.5, vjust=.5, face=\"plain\"),\n        panel.background = element_rect(fill = NA, color = \"black\"),\n        axis.line = element_line(colour = \"grey\"),\n        strip.text = element_text(size = rel(2))\n    ) +\n    geom_hline(aes(yintercept=0.5),linetype=\"solid\",color=\"#696969\", size=1)+\n    scale_y_continuous(breaks = c(0,0.5,1),limits=c(0,1))\nplt2","category":"page"},{"location":"man/interpretation/","page":"Interpretation","title":"Interpretation","text":"(Image: nodes)","category":"page"},{"location":"man/interpretation/","page":"Interpretation","title":"Interpretation","text":"Each bar corresponds to the posterior probability of being an influential node (microbe) on the response. A horizontal line is drawn at 0.5, so that nodes with bars taller than the line can be considered influential.","category":"page"},{"location":"lib/public_methods/","page":"Public Methods","title":"Public Methods","text":"CurrentModule = BayesianNetworkRegression","category":"page"},{"location":"lib/public_methods/","page":"Public Methods","title":"Public Methods","text":"Fit!(X::AbstractArray{T,2}, y::AbstractVector{U}, R; η=1.01,ζ=1.0,ι=1.0,aΔ=1.0,bΔ=1.0, ν=10, nburn=30000, nsamples=20000, V=0, x_transform=true, suppress_timer=false, num_chains=2, seed=nothing, in_seq=false, full_results=false, purge_burn=nothing) where {T,U}","category":"page"},{"location":"lib/public_methods/#BayesianNetworkRegression.Fit!-Union{Tuple{U}, Tuple{T}, Tuple{AbstractMatrix{T}, AbstractVector{U}, Any}} where {T, U}","page":"Public Methods","title":"BayesianNetworkRegression.Fit!","text":"Fit!(X::AbstractArray{T}, y::AbstractVector{U}, R; η=1.01,ζ=1.0,ι=1.0,aΔ=1.0,bΔ=1.0, \n     ν=10, nburn=30000, nsamples=20000, x_transform=true, suppress_timer=false, num_chains=2, seed=nothing, purge_burn=nothing) where {T,U}\n\nFit the Bayesian Network Regression model, generating nsamples Gibbs samples after nburn burn-in are discarded\n\nRoad map of fit!:\n\nCalls generate_samples! directly\ngenerate_samples! calls initialize_and_run! on every chain\ninitialize_and_run! calls initialize_variables! and gibbs_sample!\n\nArguments\n\nX: matrix, required, matrix of unweighted symmetric adjacency matrices to be used as predictors. Two options:        2D matrix with each row the upper triangle of the adjacency matrix associated with one sample       1D matrix with each row the adjacency matrix relating the nodes to one another\ny: vector, required, vector of response variables\nR: integer, required, the dimensionality of the latent variables u, a hyperparameter\nη: float, default=1.01, hyperparameter used for sampling the 0 value of the πᵥ parameter, must be > 1\nζ: float, default=1.0, hyperparameter used for sampling θ\nι: float, default=1.0, hyperparameter used for sampling θ\naΔ: float, default=1.0, hyperparameter used for sampling Δ\nbΔ: float, default=1.0, hyperparameter used for sampling Δ \nν: integer, default=10, hyperparameter used for sampling M, must be > R\nnburn: integer, default=30000, number of burn-in samples to generate and discard\nnsamples: integer, default=20000, number of Gibbs samples to generate after burn-in\nx_transform: boolean, default=true, set to false if X has been pre-transformed into one row per sample. Otherwise the X will be transformed automatically.\nsuppress_timer: boolean, default=false, set to true to suppress \"progress meter\" output\nnum_chains: integer, default=2, number of separate sampling chains to run (for checking convergence)\nseed: integer, default=nothing, random seed used for repeatability\npurge_burn: integer, default=nothing, if set must be less than the number of burn-in samples (and ideally burn-in is a multiple of this value). After how many burn-in samples to delete previous burn-in samples.\nfilename: logfile with the parameters used for the fit, default=\"parameters.log\". The file will be overwritten if a new name is not specified.\n\nReturns\n\nResults object with the state table from the first chain and PSRF r-hat values for  γ and ξ \n\n\n\n\n\n","category":"method"},{"location":"lib/public_methods/","page":"Public Methods","title":"Public Methods","text":"Summary(results::Results;interval::Int=95,digits::Int=3)","category":"page"},{"location":"lib/public_methods/#BayesianNetworkRegression.Summary-Tuple{BayesianNetworkRegression.Results}","page":"Public Methods","title":"BayesianNetworkRegression.Summary","text":"Summary(results::Results;interval::Int=95,digits::Int=3)\n\nGenerate summary statistics for results: point estimates and credible intervals for edge coefficients, probabilities of influence for individual nodes\n\nArguments\n\nresults: a Results object, returned from running Fit!\ninterval: (optional) Integer, level for credible intervals. Default is 95%.\ndigits: (optional) Integer, number of digits (after the decimal) to round results to. Default is 3.\n\nReturns\n\nA BNRSummary object containing a matrix of edge coefficient point estimates (coef_matrix), a matrix of edge coefficient credible intervals (ci_matrix), and a DataFrame  containing the probability of influence of each node (pi_nodes).\n\n\n\n\n\n","category":"method"},{"location":"#BayesianNetworkRegression.jl","page":"Home","title":"BayesianNetworkRegression.jl","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"BayesianNetworkRegression.jl is a Julia package to perform (Bayesian) statistical inference of a regression model with networked covariates on a real response.","category":"page"},{"location":"#References","page":"Home","title":"References","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"If you use BayesianNetworkRegression.jl in your work, we kindly ask that you cite the following paper: ","category":"page"},{"location":"","page":"Home","title":"Home","text":"Ozminkowski, S., Solís-Lemus, C. (2022). Identifying microbial drivers in biological phenotypes with a Bayesian Network Regression model.  arXiv: 2208.05600.","category":"page"},{"location":"man/inputdata/#Input-for-BayesianNetworkRegression.jl","page":"Input Data","title":"Input for BayesianNetworkRegression.jl","text":"","category":"section"},{"location":"man/inputdata/","page":"Input Data","title":"Input Data","text":"The function Fit! within BayesianNetworkRegression.jl is the main method to estimate the relationships between the edges of the microbiome network (covariates) and the variable of interest (response). ","category":"page"},{"location":"man/inputdata/","page":"Input Data","title":"Input Data","text":"There are two alternatives for the network input data:","category":"page"},{"location":"man/inputdata/","page":"Input Data","title":"Input Data","text":"A vector of matrices, where each item in the vector (of length n, where n is the sample size) is an V times V (where V is the number of microbes in each sample) adjacency matrix describing the microbiome network. All adjacency matrices must be the same size.\nA n times fracV(V-1)2 matrix, where each row in the matrix is the upper triangle of the adjacency matrix describing the network for that sample.","category":"page"},{"location":"man/inputdata/","page":"Input Data","title":"Input Data","text":"Note that the package does not assume any specific inference procedure for the estimation of the adjacency matrices (and thus, of the microbiome networks). This means that the microbiome networks can be obtained using the user's preferred methodology and simply input them into the package as described below.","category":"page"},{"location":"man/inputdata/#Tutorial-data:-Adjacency-matrices","page":"Input Data","title":"Tutorial data: Adjacency matrices","text":"","category":"section"},{"location":"man/inputdata/","page":"Input Data","title":"Input Data","text":"Suppose that you design an experiment with n different samples, and for each sample, you estimate a microbial network for V microbes and a measured phenotype.","category":"page"},{"location":"man/inputdata/","page":"Input Data","title":"Input Data","text":"The n microbial networks can be stored as n adjacency matrices. We have a toy example where we stored the adjacency matrices as a JLD2 file with the JLD2.jl package.","category":"page"},{"location":"man/inputdata/","page":"Input Data","title":"Input Data","text":"You have two files:","category":"page"},{"location":"man/inputdata/","page":"Input Data","title":"Input Data","text":"vector_networks.jld2 contains a vector of adjacency matrices and\nvector_response.jld2 contains a vector of responses (real numbers).","category":"page"},{"location":"man/inputdata/","page":"Input Data","title":"Input Data","text":"You can access the example files for the networks  here and for the responses here","category":"page"},{"location":"man/inputdata/","page":"Input Data","title":"Input Data","text":"To load the data and view an example in julia do the following:","category":"page"},{"location":"man/inputdata/","page":"Input Data","title":"Input Data","text":"using BayesianNetworkRegression\nusing JLD2\ncd(joinpath(dirname(pathof(BayesianNetworkRegression)), \"..\",\"examples\"))\n\nvector_networks = JLD2.load(\"vector_networks.jld2\")\nvector_response = JLD2.load(\"vector_response.jld2\")\n\nvector_networks[\"networks\"][1] # shows the first adjancency matrix\nvector_response[\"response\"] # shows all responses\n\nX_a = vector_networks[\"networks\"]\ny_a = vector_response[\"response\"]","category":"page"},{"location":"man/inputdata/","page":"Input Data","title":"Input Data","text":"We will use the X_a and y_a objects in the Fit! function in the next section.","category":"page"},{"location":"man/inputdata/#Reading-adjacency-matrices-from-csv-files","page":"Input Data","title":"Reading adjacency matrices from csv files","text":"","category":"section"},{"location":"man/inputdata/","page":"Input Data","title":"Input Data","text":"Most of the times, researchers will have stored the adjacency matrices as csv files (rather than JLD files as in the previous section). In this example, you have 100 adjacency matrices stored as data1.csv,...,data100.csv, as well as a csv file for the 100-dimension vector of the responses: responses.csv.","category":"page"},{"location":"man/inputdata/","page":"Input Data","title":"Input Data","text":"using BayesianNetworkRegression\nusing CSV, Tables, DataFrames\n\nvector_response = CSV.read(\"responses.csv\", DataFrame)\n\nvector_networks = Matrix{Float64}[]\nfor i in 1:100\ndat = CSV.read(string(\"data\",i,\".csv\"),DataFrame)\npush!(vector_networks, Matrix(dat))\nend\n\nX_a = vector_networks\ny_a = vector_response[:,1]","category":"page"},{"location":"man/inputdata/","page":"Input Data","title":"Input Data","text":"We will use the X_a and y_a objects in the Fit! function in the next section.","category":"page"},{"location":"man/inputdata/#Tutorial-data:-Vectorized-adjacency-matrices","page":"Input Data","title":"Tutorial data: Vectorized adjacency matrices","text":"","category":"section"},{"location":"man/inputdata/","page":"Input Data","title":"Input Data","text":"Suppose that you already converted each adjacency matrix into a vector corresponding to its upper triangle (see image below).","category":"page"},{"location":"man/inputdata/","page":"Input Data","title":"Input Data","text":"(Image: transformA)","category":"page"},{"location":"man/inputdata/","page":"Input Data","title":"Input Data","text":"That is, you have a file with n rows and fracV(V-1)2 + 1 columns. For each row, the first fracV(V-1)2 columns describe the upper triangle of an adjacency matrix and the last column gives the response variable. ","category":"page"},{"location":"man/inputdata/","page":"Input Data","title":"Input Data","text":"You can access the example file of input networks (and response) here.","category":"page"},{"location":"man/inputdata/","page":"Input Data","title":"Input Data","text":"Do not copy-paste into a \"smart\" text-editor. Instead, save the file directly into your working directory using \"save link as\" or \"download linked file as\". This file contains 100 adjacency matrices and corresponding responses.","category":"page"},{"location":"man/inputdata/","page":"Input Data","title":"Input Data","text":"To load the data and view an example in julia do the following:","category":"page"},{"location":"man/inputdata/","page":"Input Data","title":"Input Data","text":"using CSV, DataFrames\ncd(joinpath(dirname(pathof(BayesianNetworkRegression)), \"..\",\"examples\"))\n\ndat = DataFrame(CSV.File(\"matrix_networks.csv\"))\nX_v = Matrix(dat[:,1:435])\ny_v = dat[:,436]","category":"page"},{"location":"man/inputdata/","page":"Input Data","title":"Input Data","text":"We will use the X_v and y_v objects in the Fit! function in the next section.","category":"page"},{"location":"man/installation/#Installation","page":"Installation","title":"Installation","text":"","category":"section"},{"location":"man/installation/#Installation-of-Julia","page":"Installation","title":"Installation of Julia","text":"","category":"section"},{"location":"man/installation/","page":"Installation","title":"Installation","text":"Julia is a high-level and interactive programming language (like R or Matlab), but it is also high-performance (like C). To install Julia, follow instructions here. For a quick & basic tutorial on Julia, see learn x in y minutes.","category":"page"},{"location":"man/installation/","page":"Installation","title":"Installation","text":"Editors:","category":"page"},{"location":"man/installation/","page":"Installation","title":"Installation","text":"Visual Studio Code provides an editor and an integrated development environment (IDE) for Julia: highly recommended!\nYou can also run Julia within a Jupyter notebook (formerly IPython notebook).","category":"page"},{"location":"man/installation/","page":"Installation","title":"Installation","text":"IMPORTANT: Julia code is just-in-time compiled. This means that the first time you run a function, it will be compiled at that moment. So, please be patient! Future calls to the function will be much much faster. Trying out toy examples for the first calls is a good idea.","category":"page"},{"location":"man/installation/#Installation-of-the-BayesianNetworkRegression.jl-package","page":"Installation","title":"Installation of the BayesianNetworkRegression.jl package","text":"","category":"section"},{"location":"man/installation/","page":"Installation","title":"Installation","text":"To install the package, type inside Julia:","category":"page"},{"location":"man/installation/","page":"Installation","title":"Installation","text":"]\nadd BayesianNetworkRegression","category":"page"},{"location":"man/installation/","page":"Installation","title":"Installation","text":"The first step can take a few minutes, be patient.","category":"page"},{"location":"man/installation/","page":"Installation","title":"Installation","text":"The BayesianNetworkRegression.jl package has dependencies like Distributions and DataFrames (see the Project.toml file for the full list), but everything is installed automatically.","category":"page"},{"location":"man/installation/#Loading-the-Package","page":"Installation","title":"Loading the Package","text":"","category":"section"},{"location":"man/installation/","page":"Installation","title":"Installation","text":"To check that your installation worked, type this in Julia to load the package. This is something to type every time you start a Julia session:","category":"page"},{"location":"man/installation/","page":"Installation","title":"Installation","text":"using BayesianNetworkRegression","category":"page"},{"location":"man/installation/","page":"Installation","title":"Installation","text":"This step can also take a while, if Julia needs to pre-compile the code (after a package update for instance).","category":"page"},{"location":"man/installation/","page":"Installation","title":"Installation","text":"Press ? inside Julia to switch to help mode,  followed by the name of a function (or type) to get more details about it.","category":"page"}]
}