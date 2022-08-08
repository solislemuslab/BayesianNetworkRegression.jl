using Documenter, BayesianNetworkRegression

makedocs(
    sitename="BayesianNetworkRegression.jl",
    authors="Samuel Ozminkowski, Claudia Solís-Lemus, and contributors",
    modules=[BayesianNetworkRegression],
    format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true"),
    pages = [
        "Home" => "index.md",
        "Manual" => [
            "Installation" => "man/installation.md",
            "Input Data" => "man/inputdata.md",
            "Model Fitting" => "man/fit_model.md",
            "Interpretation" => "man/interpretation.md",
        ],
        "Library" => [
            "Public Methods" => "lib/public_methods.md",
        ]
    ]
)

deploydocs(
    repo = "github.com/solislemuslab/BayesianNetworkRegression.jl.git",
)