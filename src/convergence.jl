## taken from https://github.com/TuringLang/MCMCDiagnosticTools.jl

function rhat(
    chains::AbstractArray{<:Union{Missing,Real},3}
)
    # compute size of matrices (each chain is split!)
    niter = size(chains, 1) ÷ 2
    nparams = size(chains, 2)
    nchains = 2 * size(chains, 3)

    # do not compute estimates if there is only one sample or lag
    maxlag = niter - 1
    maxlag > 0 || return fill(missing, nparams), fill(missing, nparams)

    # define caches for mean and variance
    U = typeof(zero(eltype(chains)) / 1)
    T = promote_type(eltype(chains), typeof(zero(eltype(chains)) / 1))
    chain_mean = Array{T}(undef, 1, nchains)
    chain_var = Array{T}(undef, nchains)
    samples = Array{T}(undef, niter, nchains)

    # compute correction factor
    correctionfactor = (niter - 1) / niter

    # define output arrays
    rhat = Vector{T}(undef, nparams)

    # for each parameter
    for (i, chains_slice) in enumerate((view(chains, :, i, :) for i in axes(chains, 2)))
        # check that no values are missing
        if any(x -> x === missing, chains_slice)
            rhat[i] = missing
            continue
        end

        # split chains
        MCMCDiagnosticTools.copyto_split!(samples, chains_slice)

        # calculate mean of chains
        Statistics.mean!(chain_mean, samples)

        # calculate within-chain variance
        #@inbounds for j in 1:nchains
        for j in 1:nchains
            chain_var[j] = Statistics.var(
                view(samples, :, j); mean=chain_mean[j], corrected=true
            )
        end
        W = Statistics.mean(chain_var)

        # compute variance estimator var₊, which accounts for between-chain variance as well
        var₊ = correctionfactor * W + Statistics.var(chain_mean; corrected=true)

        # estimate the potential scale reduction
        if var₊ == 0 && W == 0
            rhat[i] = 1
        elseif W == 0
            rhat[i] = Inf
        else
            rhat[i] = sqrt(var₊ / W)
        end
    end

    return rhat
end