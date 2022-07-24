#region output
struct Results
    state::Table
    rhat::Table
    burn_in::Int
    sampled::Int
end

struct ConfInt
    lower::AbstractFloat 
    upper::AbstractFloat
    confidence::Int
end

struct BNRSummary
    mean_matrix::Matrix
    ci_matrix::Matrix
    xi::DataFrame
end

Base.show(io::IO, ci::ConfInt) = print(io, "(",ci.lower,",",ci.upper,")")
Base.round(x::ConfInt;digits=3) = ConfInt(round(x.lower;digits),round(x.upper;digits),x.confidence)
#endregion


#region custom sampling
"""
    sample_u!(state::Table, i, j, R, rng)

Sample rows of the u matrix, either from MVN with mean 0 and covariance matrix M or a row of 0s

# Arguments
- `state` : all states, a vector of tuples (row-table) of all variables
- `i` : index of current state, used to index state variable
- `j` : index of the row of the u matrix to sample for
- `R` : dimension of u vectors, length of u to set
- `rng` : random number generator to be used for sampling
"""
function sample_u!(state::Table, i, j, R, rng)
    if (state.ξ[1,i] == 1)
        state.u[i,:,j] = rand(rng,MultivariateNormal(zeros(R), Matrix(state.M[i,:,:])))
    else
        state.u[i,:,j] = zeros(R)
    end
    nothing
end

"""
    sample_rgig(a,b,rng)

Sample from the GeneralizedInverseGaussian distribution with p=1/2, b=b, a=a

# Arguments
- `a` : shape and scale parameter a, sometimes also called ψ
- `b` : shape and scale parameter b, sometimes also called χ
- `rng` : random number generator to be used for sampling

# Returns
one sample from the GIG distribution with p=1/2, b=b, a=a
"""
function sample_rgig(a,b, rng)::Float64
    return sample_gig(rng,1/2,b,a)
end

"""
    sample_Beta(a,b,rng)

Sample from the Beta distribution, with handling for a=0 and/or b=0

#Arguments
- `a` : shape parameter a ≥ 0
- `b` : shape parameter b ≥ 0
- `rng` : random number generator to be used for sampling
"""
function sample_Beta(a,b, rng)
    if a > 0.0 && b > 0.0
        return rand(rng,Beta(a, b))
    elseif a > 0.0
        return 1.0
    elseif b > 0.0
        return 0.0
    else
        return sample(rng,[0.0,1.0])
    end
end

"""
    sample_π_dirichlet!(ret::AbstractVector{U},r,η,λ::AbstractVector{T},rng)

Sample from the 3-variable doirichlet distribution with weights
[r^η,1,1] + [#{λ[r] == 0}, #{λ[r] == 1}, #{λ[r] = -1}]

# Arguments
- `state` : all states, a vector of tuples (row-table) of all variables
- `i` : index of current state, used to index state variable
- `r` : integer, base term for the first weight and index for λ vector
- `η` : real number, power term for the first weight
- `λ` : 1d array of -1,0,1s, used to determine which weight is added to
- `rng` : random number generator to be used for sampling

# Returns
A vector of length 3 drawn from the Dirichlet distribution
"""
function sample_π_dirichlet!(state::Table,i,r,η,λ::AbstractVector{T}, rng) where {T}
    if λ[r] == 1
        state.πᵥ[i,r,1:3] = rand(rng,Dirichlet([r^η,2,1]))
    elseif λ[r] == 0
        state.πᵥ[i,r,1:3] = rand(rng,Dirichlet([r^η + 1,1,1]))
    else
        state.πᵥ[i,r,1:3] = rand(rng,Dirichlet([r^η,1,2]))
    end
    
    nothing
end
#endregion



"""
    initialize_variables!(state::Table, X_new::AbstractArray{U}, X::AbstractArray{T}, η, ζ, ι, R, aΔ, bΔ, ν, rng, V=0, x_transform::Bool=true)

    Initialize all variables using prior distributions. Note, if x_transform is true V will be ignored and overwritten with the implied value from X.
    All initializations done in place on the state argument.

    # Arguments
    - `state` : a row-table structure containing all past states, the current state, and space for the future states
    - `X_new` : 2-dimensional n × V(V-1)/2 matrix - will hold reshaped X
    - `X` : vector of unweighted symmetric adjacency matrices to be used as predictors. each element of the array should be 1 matrix
    - `η` : hyperparameter used to sample from the Dirichlet distribution (r^η)
    - `ζ` : hyperparameter used as the shape parameter in the gamma distribution used to sample θ
    - `ι` : hyperparameter used as the scale parameter in the gamma distribution used to sample θ
    - `R` : the dimensionality of the latent variables u, a hyperparameter
    - `aΔ`: hyperparameter used as the a parameter in the beta distribution used to sample Δ.
    - `bΔ`: hyperparameter used as the b parameter in the beta distribution used to sample Δ. aΔ and bΔ values causing the Beta distribution to have mass concentrated closer to 0 will cause more zeros in ξ
    - `rng` : random number generator to be used for sampling
    - `ν` : hyperparameter used as the degrees of freedom parameter in the InverseWishart distribution used to sample M.
    - `V`: Value of V, the number of nodes in the original X matrix. Only input when x_transform is false. Always output.
    - `x_transform`: boolean, set to false if X has been pre-transformed into one row per sample. True by default.

    # Returns
    nothing
"""
function initialize_variables!(state::Table, X_new::AbstractArray{U}, X::AbstractArray{T}, η, ζ, ι, R, aΔ, bΔ, ν, rng , V=0, x_transform::Bool=true) where {T,U}
    # η must be greater than 1, if it's not set it to its default value of 1.01
    if (η <= 1)
        η = 1.01
        println("η value invalid, reset to default of 1.01")
    end

    if x_transform
        V = Int64(size(X[1],1))
    end
    q = floor(Int,V*(V-1)/2)

    if x_transform
        for i in 1:size(X,1)
            X_new[i,:] = lower_triangle(X[i])
        end
    else
        X_new[:,:] = X
    end

    state.θ[1] = 0.5

    state.S[1,:] = map(k -> rand(rng,Exponential(state.θ[1]/2)), 1:q)

    state.πᵥ[1,:,:] = zeros(R,3)
    for r in 1:R
        state.πᵥ[1,r,:] = rand(rng,Dirichlet([r^η,1,1]))
    end
    state.λ[1,:] = map(r -> sample(rng,[0,1,-1], StatsBase.weights(state.πᵥ[1,r,:]),1)[1], 1:R)
    state.Δ[1] = 0.5
    
    state.ξ[1,:] = rand(rng,Binomial(1,state.Δ[1]),V)
    state.M[1,:,:] = rand(rng,InverseWishart(ν,cholesky(Matrix(I,R,R))))
    for i in 1:V
        state.u[1,:,i] = rand(rng,MultivariateNormal(zeros(R), I(R)))
    end
    state.μ[1] = 1.0
    state.τ²[1] = 1.0

    state.γ[1,:] = rand(rng,MultivariateNormal(reshape(lower_triangle(transpose(state.u[1,:,:]) * Diagonal(state.λ[1,:]) * state.u[1,:,:]),(q,)), state.τ²[1]*Diagonal(state.S[1,:,1])))
    X_new
end


#region update variables

"""
    update_τ²!(state::Table, i, X::AbstractArray{T,2}, y::AbstractVector{U}, V, rng)

Sample the next τ² value from the InverseGaussian distribution with mean n/2 + V(V-1)/4 and variance ((y - μ1 - Xγ)ᵀ(y - μ1 - Xγ) + (γ - W)ᵀD⁻¹(γ - W)

# Arguments
- `state` : all states, a vector of tuples (row-table) of all variables
- `i` : index of current state, used to index state variable
- `X` : 2 dimensional array of predictor values, 1 row per sample (upper triangle of original X)
- `y` : vector of response values
- `V` : dimension of original symmetric adjacency matrices
- `rng` : random number generator to be used for sampling

# Returns
nothing - updates are done in place
"""
function update_τ²!(state::Table, i, X::AbstractArray{T,2}, y::AbstractVector{U}, V, rng) where {T,U}
    n  = size(y,1)

    yμ1Xγ = (y) .- state.μ[i-1] - X*state.γ[i-1,:]
    γW = (state.γ[i-1,:] - lower_triangle(transpose(state.u[i-1,:,:]) * Diagonal(state.λ[i-1,:]) * state.u[i-1,:,:]))

    σₜ² = ((transpose(yμ1Xγ) * yμ1Xγ)[1])/2 + sum_kbn(((γW.^2)./2) ./ state.S[i-1,:])

    state.τ²[i] = rand(rng,InverseGamma((n/2) + (V*(V-1)/4), σₜ²))
    nothing
end

"""
    update_u_ξ!(state::Table, i, V, rng)

Sample the next u and ξ values

# Arguments
- `state` : all states, a vector of tuples (row-table) of all variables
- `i` : index of current state, used to index state variable
- `V` : dimension of original symmetric adjacency matrices
- `rng` : random number generator to be used for sampling

# Returns
nothing - updates are done in place
"""
function update_u_ξ!(state::Table, i, V, rng)

    for k in 1:V
        U = transpose(state.u[i-1,:,Not(k)]) * Diagonal(state.λ[i-1,:])
        Uᵀ = transpose(U)
        s = create_lower_tri(state.S[i-1,:], V)
        Γ = create_lower_tri(state.γ[i-1,:], V)
        if k == 1
            γk=vcat(Γ[2:V,1])
            H = Diagonal(vcat(s[2:V,1]))
        elseif k == V
            γk=vcat(Γ[V,1:(V-1)])
            H = Diagonal(vcat(s[V,1:(V-1)]))
        else
            γk= vcat(Γ[k,1:(k-1)],Γ[(k+1):V,k])
            H = Diagonal(vcat(s[k,1:(k-1)],s[(k+1):V,k]))
        end

        d = size(state.u,2)

        Σ⁻¹ = ((Uᵀ)*(H\U))/state.τ²[i] + inv(state.M[i-1,:,:])
        C = cholesky(Hermitian(Σ⁻¹))

        w_top = (1-state.Δ[i-1]) * pdf(MultivariateNormal(zeros(size(H,1)),Symmetric(state.τ²[i]*H)),γk)
        w_bot = state.Δ[i-1] * pdf( MultivariateNormal(zeros(size(H,1)), Symmetric(state.τ²[i] * H + U * state.M[i-1,:,:] * Uᵀ)),γk)
        w = w_top / (w_bot + w_top)


        state.ξ[i,k] = update_ξ(w, rng)
        μₜ = Σ⁻¹ \ (Uᵀ*inv(H)*γk)/state.τ²[i] 
        u_tmp = μₜ + inv(C.U) * rand(rng,MultivariateNormal(zeros(d),I(d)))

        state.u[i,:,k] = state.ξ[i,k] .* u_tmp

    end
    nothing
end

"""
    update_ξ(w, rng)

Sample the next ξ value from the Bernoulli distribution with parameter 1-w

# Arguments
- `w` : parameter to use for sampling, probability that 0 is drawn
- `rng` : random number generator to be used for sampling

# Returns
the new value of ξ
"""
function update_ξ(w, rng)
    if w == 0
        return 1
    elseif w == 1
        return 0
    end
    return Int64(rand(rng,Bernoulli(1 - w)))
end

"""
    update_γ!(state::Table, i, X::AbstractArray{T,2}, y::AbstractVector{U}, n, rng)

Sample the next γ value from the normal distribution, decomposed as described in Guha & Rodriguez 2018

# Arguments
- `state` : all states, a vector of tuples (row-table) of all variables
- `i` : index of current state, used to index state variable
- `X` : 2 dimensional array of predictor values, 1 row per sample (upper triangle of original X)
- `y` : response values
- `n` : number of samples
- `rng` : random number generator to be used for sampling

# Returns
nothing - all updates are done in place
"""
function update_γ!(state::Table, i, X::AbstractArray{T,2}, y::AbstractVector{U}, n, rng) where {T,U}
    W = lower_triangle( transpose(state.u[i,:,:]) * Diagonal(state.λ[i-1,:]) * state.u[i,:,:] )

    D = Diagonal(state.S[i-1,:])
    τ²D = state.τ²[i]*D
    Xτ = X ./ sqrt(state.τ²[i])
    q = size(D,1)

    τ = sqrt(state.τ²[i])
    Δᵧ₁ = rand(rng,MultivariateNormal(zeros(q), τ²D))
    Δᵧ₂ = rand(rng,MultivariateNormal(zeros(n), I(n)))
    
    a1 = ((y) - X*W .- state.μ[i-1]) / τ
    a3 = (Xτ * Δᵧ₁) + Δᵧ₂
    a4 = (Xτ*τ²D*transpose(Xτ)+I(n)) \ (a1 - a3)
    a5 = Δᵧ₁ + τ²D * transpose(Xτ) * a4
    state.γ[i,:] = a5 + W
    nothing
end

"""
    update_D!(state::Table, i, V, rng)

Sample the next D value from the GeneralizedInverseGaussian distribution with p = 1/2, a=((γ - uᵀΛu)^2)/τ², b=θ

# Arguments
- `state` : all states, a vector of tuples (row-table) of all variables
- `i` : index of current state, used to index state variable
- `V` : dimension of original symmetric adjacency matrices
- `rng` : random number generator to be used for sampling

# Returns
nothing - all updates are done in place
"""
function update_D!(state::Table, i, V, rng)
    a_ = (state.γ[i,:] - (lower_triangle( transpose(state.u[i,:,:]) * Diagonal(state.λ[i-1,:]) * state.u[i,:,:] ))).^2 / state.τ²[i]
    state.S[i,:] = map(k -> sample_rgig(state.θ[i-1],a_[k], rng), 1:size(state.S,2))
    nothing
end

"""
    update_θ!(state::Table, i, ζ, ι, V, rng)

Sample the next θ value from the Gamma distribution with a = ζ + V(V-1)/2 and b = ι + ∑(s[k,l]/2)

# Arguments
- `state` : all states, a vector of tuples (row-table) of all variables
- `i` : index of current state, used to index state variable
- `ζ` : hyperparameter, used to construct `a` parameter
- `ι` : hyperparameter, used to construct `b` parameter
- `V` : dimension of original symmetric adjacency matrices
- `rng` : random number generator to be used for sampling

# Returns
nothing - all updates are done in place
"""
function update_θ!(state::Table, i, ζ, ι, V, rng)
    state.θ[i] = rand(rng,Gamma(ζ + (V*(V-1))/2,2/(2*ι + sum_kbn(state.S[i,:]))))
    nothing
end

"""
    update_Δ!(state::Table, i, aΔ, bΔ, rng)

Sample the next Δ value from the Beta distribution with parameters a = aΔ + ∑ξ and b = bΔ + V - ∑ξ

# Arguments
- `state` : all states, a vector of tuples (row-table) of all variables
- `i` : index of current state, used to index state variable
- `aΔ`: hyperparameter used as part of the a parameter in the beta distribution used to sample Δ.
- `bΔ`: hyperparameter used as part of the b parameter in the beta distribution used to sample Δ.
- `rng` : random number generator to be used for sampling

# Returns
nothing - all updates are done in place
"""
function update_Δ!(state::Table, i, aΔ, bΔ, rng)
    state.Δ[i] = sample_Beta(aΔ + sum_kbn(state.ξ[i,:]),bΔ + sum_kbn(1 .- state.ξ[i,:]), rng)
    nothing
end

"""
    update_M!(state::Table, i, ν, V, rng)

Sample the next M value from the InverseWishart distribution with df = V + # of nonzero columns in u and Ψ = I + ∑ uΛuᵀ

# Arguments
- `state` : all states, a vector of tuples (row-table) of all variables
- `i` : index of current state, used to index state variable
- `ν` : hyperparameter, base df for IW distribution (to be added to by sum of ξs)
- `V` : dimension of original symmetric adjacency matrices
- `rng` : random number generator to be used for sampling

# Returns
nothing - all updates are done in place
"""
function update_M!(state::Table, i, ν, V, rng)
    R = size(state.u[i,:,:],1)
    uuᵀ = zeros(Float64,R,R)
    num_nonzero = 0
    for v in 1:V
        uuᵀ = uuᵀ + (state.u[i,:,v]) * transpose(state.u[i,:,v])
        if !isapprox(state.ξ[i,v],0,atol=0.1)
            num_nonzero = num_nonzero + 1
        end
    end

    state.M[i,:,:] = rand(rng,InverseWishart(ν + num_nonzero,cholesky(Matrix(I(R) + uuᵀ))))
    nothing
end

"""
    update_μ!(state::Table, i, X::AbstractArray{T,2}, y::AbstractVector{U}, n, rng)

Sample the next μ value from the normal distribution with mean 1ᵀ(y - Xγ)/n and variance τ²/n

# Arguments
- `state` : all states, a vector of tuples (row-table) of all variables
- `i` : index of current state, used to index state variable
- `X` : 2 dimensional array of predictor values, 1 row per sample (upper triangle of original X)
- `y` : response values
- `n` : number of samples (length of y)
- `rng` : random number generator to be used for sampling

# Returns
nothing - all updates are done in place
"""
function update_μ!(state::Table, i, X::AbstractArray{T,2}, y::AbstractVector{U}, n, rng) where {T,U}
    μₘ = mean(y - (X * state.γ[i,:]))
    σₘ = sqrt(state.τ²[i]/n)
    state.μ[i] = rand(rng,Normal(μₘ,σₘ))
    nothing
end

"""
    update_Λ!(state::Table, i, R, rng)

Sample the next values of λ from [1,0,-1] with probabilities determined from a normal mixture

# Arguments
- `state` : all states, a vector of tuples (row-table) of all variables
- `i` : index of current state, used to index state variable
- `R` : the dimensionality of the latent variables u
- `rng` : random number generator to be used for sampling

# Returns
nothing - all updates are done in place
"""
function update_Λ!(state::Table, i, R, rng)
    Λ = Diagonal(state.λ[i-1,:])
    q = size(state.γ[i,:],1)
    τ²D = state.τ²[i] * Diagonal(state.S[i,:])

    for r in 1:R
        Λ₋₁= deepcopy(Λ)
        Λ₋₁[r,r] = -1
        Λ₀ = deepcopy(Λ)
        Λ₀[r,r] = 0
        Λ₁ = deepcopy(Λ)
        Λ₁[r,r] = 1
        u_tr = transpose(state.u[i,:,:])
        W₋₁= lower_triangle(u_tr * Λ₋₁ * state.u[i,:,:])
        W₀ = lower_triangle(u_tr * Λ₀ * state.u[i,:,:])
        W₁ = lower_triangle(u_tr * Λ₁ * state.u[i,:,:])

        n₀ = sum_kbn(map(j -> logpdf(Normal(W₀[j],sqrt(τ²D[j,j])),state.γ[i,j]),1:q))
        n₁ = sum_kbn(map(j -> logpdf(Normal(W₁[j],sqrt(τ²D[j,j])),state.γ[i,j]),1:q))
        n₋₁ = sum_kbn(map(j -> logpdf(Normal(W₋₁[j],sqrt(τ²D[j,j])),state.γ[i,j]),1:q))

        probs = [n₀,n₁,n₋₁]
        pmax = max(probs...)
        a1 = exp.(probs .- pmax)
        state.λ[i,r] = sample(rng,[0,1,-1],StatsBase.weights(state.πᵥ[i-1,r,:] .* a1))
    end
    nothing
end

"""
    update_π!(state::Table,i,η,R, rng)

Sample the new values of πᵥ from the Dirichlet distribution with parameters [1 + #{r: λᵣ= 1}, #{r: λᵣ = 0} + r^η, 1 + #{r: λᵣ = -1 }]

# Arguments
- `state` : all states, a vector of tuples (row-table) of all variables
- `i` : index of current state, used to index state variable
- `η` : hyperparameter used for sampling the 0 value (r^η)
- `R` : dimension of u vectors
- `rng` : random number generator to be used for sampling

# Returns
nothing, all updating is done in place
"""
function update_π!(state::Table, i, η, R, rng)

    for r in 1:R
        sample_π_dirichlet!(state,i,r,η,state.λ[i,:],rng)
    end
    nothing
end

#endregion

"""
    gibbs_sample!(state::Table, iteration, X::AbstractArray{U,2}, y::AbstractVector{S}, V, η, ζ, ι, R, aΔ, bΔ, ν, rng)

Take one Gibbs Sample (update the state table in place)

# Arguments
- `state`: a row-table structure containing all past states, the current state, and space for the future states
- `iteration`: the number of the current sample (for accessing the current state)
- `X` : matrix of unweighted symmetric adjacency matrices to be used as predictors. each row should be the upper triangle of the adjacency matrix associated with one sample.
- `y` : vector of response variables
- `V` : number of nodes in adjacency matrices
- `η` : hyperparameter used for sampling the 0 value of the πᵥ parameter
- `ζ` : hyperparameter used for sampling θ
- `ι` : hyperparameter used for sampling θ
- `R` : hyperparameter - depth of u vectors (and others)
- `aΔ`: hyperparameter used for sampling Δ
- `bΔ`: hyperparameter used for sampling Δ
- `ν` : hyperparameter used for sampling M
- `rng` : random number generator to be used for sampling

# Returns:
nothing, all updating is done in place
"""
function gibbs_sample!(state::Table, iteration, X::AbstractArray{U,2}, y::AbstractVector{S}, V, η, ζ, ι, R, aΔ, bΔ, ν, rng) where {S,U}
    n = size(X,1)

    update_τ²!(state, iteration, X, y, V, rng)
    update_u_ξ!(state, iteration, V, rng)
    update_γ!(state, iteration, X, y, n, rng)
    update_D!(state, iteration, V, rng)
    update_θ!(state, iteration, ζ, ι, V, rng)
    update_Δ!(state, iteration, aΔ, bΔ, rng)
    update_M!(state, iteration, ν, V, rng)
    update_μ!(state, iteration, X, y, n, rng)
    update_Λ!(state, iteration, R, rng)
    update_π!(state, iteration, η, R, rng)
    nothing
end

"""
    Fit!(X::AbstractArray{T}, y::AbstractVector{U}, R; η=1.01,ζ=1.0,ι=1.0,aΔ=1.0,bΔ=1.0, 
         ν=10, nburn=30000, nsamples=20000, x_transform=true, suppress_timer=false, num_chains=2, seed=nothing, purge_burn=nothing) where {T,U}

Fit the Bayesian Network Regression model, generating `nsamples` Gibbs samples after `nburn` burn-in are discarded

Road map of fit!:
- Calls [`generate_samples!`](@ref) directly
- `generate_samples!` calls [`initialize_and_run!`](@ref) on every chain
- `initialize_and_run!` calls [`initialize_variables!`](@ref) and [`gibbs_sample!`](@ref)

# Arguments
- `X`: matrix, required, matrix of unweighted symmetric adjacency matrices to be used as predictors. Two options: 
        2D matrix with each row the upper triangle of the adjacency matrix associated with one sample
        1D matrix with each row the adjacency matrix relating the nodes to one another
- `y`: vector, required, vector of response variables
- `R`: integer, required, the dimensionality of the latent variables u, a hyperparameter
- `η`: float, default=1.01, hyperparameter used for sampling the 0 value of the πᵥ parameter, must be > 1
- `ζ`: float, default=1.0, hyperparameter used for sampling θ
- `ι`: float, default=1.0, hyperparameter used for sampling θ
- `aΔ`: float, default=1.0, hyperparameter used for sampling Δ
- `bΔ`: float, default=1.0, hyperparameter used for sampling Δ 
- `ν`: integer, default=10, hyperparameter used for sampling M, must be > R
- `nburn`: integer, default=30000, number of burn-in samples to generate and discard
- `nsamples`: integer, default=20000, number of Gibbs samples to generate after burn-in
- `x_transform`: boolean, default=true, set to false if X has been pre-transformed into one row per sample. Otherwise the X will be transformed automatically.
- `suppress_timer`: boolean, default=false, set to true to suppress "progress meter" output
- `num_chains`: integer, default=2, number of separate sampling chains to run (for checking convergence)
- `seed`: integer, default=nothing, random seed used for repeatability
- `purge_burn`: integer, default=nothing, if set must be less than the number of burn-in samples, and a divisor of burn-in. The maximum number of burn-in samples stored at any time.

# Returns

    `Results` object with the state table from the first chain and PSRF r-hat values for  γ and ξ

"""
function Fit!(X::AbstractArray{T}, y::AbstractVector{U}, R; η=1.01,ζ=1.0,ι=1.0,aΔ=1.0,bΔ=1.0, 
    ν=10, nburn=30000, nsamples=20000, x_transform=true, suppress_timer=false, 
    num_chains=2, seed=nothing, purge_burn=nothing) where {T,U}

    generate_samples!(X, y, R; η=η,ζ=ζ,ι=ι,aΔ=aΔ,bΔ=bΔ,ν=ν,nburn=nburn,nsamples=nsamples,x_transform=x_transform, 
    suppress_timer=suppress_timer,num_chains=num_chains,seed=seed,purge_burn=purge_burn)
end


"""
    return_psrf_VOI(states,num_chains,nburn,nsamples,V,q)

organize and return ALL (burn-in and post-burn-in) samples for the first state, calculate and return PSRF r-hat for ξ and γ

# Arguments 
- `states`: a vector of states, each of which is a row-table structure containing all past states (or only post-burn-in)
- `num_chains`: the number of chains run (length of states vector)
- `nburn`: the number of burn-in samples. If the states in `states` only contain post-burn, set this to 0
- `nsamples`: the number of post-burn-in samples to use
- `V`: the dimension of the original adjacency matrix, used to size the return table
- `q`: the dimension of the model matrix, used to size the return table

# Returns
A `Results` object with the state table from the first chain and PSRF r-hat values for  γ and ξ

"""
function return_psrf_VOI(states,num_chains,nburn,nsamples,V,q)
    all_ξs = Array{Float64,3}(undef,(nsamples,V,num_chains))
    all_γs = Array{Float64,3}(undef,(nsamples,q,num_chains))

    total = nsamples + nburn

    for c=1:num_chains
        all_ξs[:,:,c] = states[c].ξ[nburn+1:total,:,1]
        all_γs[:,:,c] = states[c].γ[nburn+1:total,:,1]
    end

    psrf = Table(ξ = Vector{Float64}(undef,q), γ = Vector{Float64}(undef,q))

    psrf.γ[1:q] = rhat(all_γs)
    psrf.ξ[1:V] = rhat(all_ξs)

    return Results(states[1],psrf,nburn,nsamples)
end

"""
    initialize_and_run!(X::AbstractArray{T},y::AbstractVector{U},c,total,V,R,η,ζ,ι,aΔ,bΔ,ν,rng,x_transform,suppress_timer,prog_freq,purge_burn,nsamples,channel) where {T,U}

Initialize a new state table with all variables with [`initialize_variables!`](@ref) and generate `total` samples with [`gibbs_sample!`](@ref).

# Arguments
- `X`: matrix of unweighted symmetric adjacency matrices to be used as predictors. each row should be the upper triangle of the adjacency matrix associated with one sample.
- `y`: vector of response variables
- `c`: index of the current chain.
- `total`: total number of Gibbs samples to take (burn in and retained)
- `V`: dimensionality of the adjacency matrices (number of nodes)
- `R` : the dimensionality of the latent variables u, a hyperparameter
- `η`: hyperparameter used for sampling the 0 value of the πᵥ parameter
- `ζ` : hyperparameter used for sampling θ
- `ι` : hyperparameter used for sampling θ
- `aΔ`: hyperparameter used for sampling Δ
- `bΔ`: hyperparameter used for sampling Δ
- `ν` : hyperparameter used for sampling M
- `rng` : random number generator to be used for sampling.
- `x_transform`: boolean, set to false if X has been pre-transformed into one row per sample. Otherwise the X will be transformed automatically.
- `suppress_timer`: boolean, set to true to suppress "progress meter" output
- `prog_freq`: integer, how many samples to run between each update of the progress-meter. Lower values will give a more accurate reporting of time remaining but may slow execution of the program (especially when run in parallel).
- `purge_burn`: integer, if set must be less than the number of burn-in samples (and ideally burn-in is a multiple of this value). After how many burn-in samples to delete previous burn-in samples.
- `nsamples`: integer, the number of post burn-in samples to retain. only necessary to provide when purge_burn is not nothing
- `channel`: channel between worker and manager, used to update the progress meter when running parallel chains

# Returns

The complete `state` table with all samples of all variables.

"""
function initialize_and_run!(X::AbstractArray{T},y::AbstractVector{U},c,total,V,R,η,ζ,ι,aΔ,bΔ, 
                             ν,rng,x_transform,suppress_timer,prog_freq,purge_burn,nsamples,channel) where {T,U}

    p = Progress(floor(Int64,(total-1)/10);dt=1,showspeed=true, enabled = !suppress_timer)
    n = size(X,1)
    q = Int64(V*(V-1)/2)

    tot_save = total
    if (!isnothing(purge_burn))
        tot_save = nsamples + purge_burn;
    end

    nburn = total - nsamples

    X_new = Matrix{eltype(T)}(undef, n, q)
    state = Table(τ² = Array{Float64,3}(undef,(tot_save,1,1)), u = Array{Float64,3}(undef,(tot_save,R,V)),
            ξ = Array{Float64,3}(undef,(tot_save,V,1)), γ = Array{Float64,3}(undef,(tot_save,q,1)),
            S = Array{Float64,3}(undef,(tot_save,q,1)), θ = Array{Float64,3}(undef,(tot_save,1,1)),
            Δ = Array{Float64,3}(undef,(tot_save,1,1)), M = Array{Float64,3}(undef,(tot_save,R,R)),
            μ = Array{Float64,3}(undef,(tot_save,1,1)), λ = Array{Float64,3}(undef,(tot_save,R,1)),
            πᵥ= Array{Float64,3}(undef,(tot_save,R,3)), Σ⁻¹= Array{Float64,3}(undef,(tot_save,R,R)),
            invC = Array{Float64,3}(undef,(tot_save,R,R)), μₜ = Array{Float64,3}(undef,(tot_save,R,1)))
            
    initialize_variables!(state, X_new, X, η, ζ, ι, R, aΔ, bΔ, ν, rng, V, x_transform)

    j = 2
    for i in 2:total
        gibbs_sample!(state, j, X_new, y, V, η, ζ, ι, R, aΔ, bΔ, ν, rng)
        if c==1 && (i % prog_freq == 0) 
            put!(channel,true) 
        end
        if !isnothing(purge_burn) && i < nburn && j == purge_burn+1
            copy_table!(state,1,j)
            j = 1
        end
        j = j+1
    end
    return state
end


"""
    generate_samples!(X::AbstractArray{T}, y::AbstractVector{U}, R; η=1.01,ζ=1.0,ι=1.0,aΔ=1.0,bΔ=1.0, V=0,
        ν=10, nburn=30000, nsamples=20000, x_transform=true, suppress_timer=false, num_chains=2, seed=nothing, purge_burn=nothing) where {T,U}

Main function for the program. Calls [`initialize_and_run!`](@ref) for each chain. 

# Arguments
- `X`: 2d matrix, required, matrix of unweighted symmetric adjacency matrices to be used as predictors. each row should be the upper triangle of the adjacency matrix associated with one sample.
- `y`: vector, required, vector of response variables
- `R`: integer, required, the dimensionality of the latent variables u, a hyperparameter
- `η`: float, default=1.01, hyperparameter used for sampling the 0 value of the πᵥ parameter, must be > 1
- `ζ`: float, default=1.0, hyperparameter used for sampling θ
- `ι`: float, default=1.0, hyperparameter used for sampling θ
- `aΔ`: float, default=1.0, hyperparameter used for sampling Δ
- `bΔ`: float, default=1.0, hyperparameter used for sampling Δ 
- `ν`: integer, default=10, hyperparameter used for sampling M, must be > R
- `nburn`: integer, default=30000, number of burn-in samples to generate and discard
- `nsamples`: integer, default=20000, number of Gibbs samples to generate after burn-in
- `x_transform`: boolean, default=true, set to false if X has been pre-transformed into one row per sample. Otherwise the X will be transformed automatically.
- `suppress_timer`: boolean, default=false, set to true to suppress "progress meter" output
- `num_chains`: integer, default=2, number of separate sampling chains to run (for checking convergence)
- `seed`: integer, default=nothing, random seed used for repeatability
- `purge_burn`: integer, default=nothing, if set must be less than the number of burn-in samples (and ideally burn-in is a multiple of this value). After how many burn-in samples to delete previous burn-in samples.

# Returns

`Results` object with the state table from the first chain and PSRF r-hat values for  γ and ξ

"""
function generate_samples!(X::AbstractArray{T}, y::AbstractVector{U}, R; η=1.01,ζ=1.0,ι=1.0,aΔ=1.0,bΔ=1.0,
    ν=10, nburn=30000, nsamples=20000, x_transform=true, suppress_timer=false, num_chains=2, seed=nothing,purge_burn=nothing) where {T,U}

    if ν < R
        ArgumentError("ν value ($ν) must be greater than R value ($R)")
    elseif ν == R
        println("Warning: ν==R may give poor accuracy. Consider increasing ν")
    end

    if x_transform
        V = size(X[1,:],2)
        q = floor(Int,V*(V-1)/2)
    else
        q = size(X,2)
        V = Int64((1 + sqrt( 1 + 8 * q))/2)
    end

    states = Vector{Table}(undef,num_chains)
    total = nburn + nsamples

    prog_freq = 100
    rngs = [ (isnothing(seed) ? Xoshiro() : Xoshiro(seed*c)) for c = 1:num_chains ]

    if !isnothing(purge_burn) && (purge_burn < nburn) && purge_burn != 0
        if nburn % purge_burn != 0 
            purge_burn = purge_burn - (purge_burn % nburn)
        end
    else
        purge_burn = nothing
    end
     
    p = Progress(Int(floor((total-1)/prog_freq));dt=1,showspeed=true, enabled = !suppress_timer)
    channel = RemoteChannel(()->Channel{Bool}(), 1)
        
    @sync begin 
        @async while take!(channel)
            next!(p)
        end
        @async begin
            states = pmap(1:num_chains) do c
                return initialize_and_run!(X,y,c,total,V,R,η,ζ,ι,aΔ,bΔ,ν,rngs[c],x_transform,suppress_timer,prog_freq,purge_burn,nsamples,channel)
            end
            put!(channel, false)
        end
    end

    return return_psrf_VOI(states,num_chains,!isnothing(purge_burn) ? purge_burn : nburn,nsamples,V,q)
end


function Summary(results::Results;interval=95,digits=3)
    nburn = results.burn_in
    nsamples = results.sampled
    total = nburn+nsamples

    lower_bound = (100-interval)/200
    upper_bound = 1-lower_bound
    γ_sorted = sort(results.state.γ[nburn+1:total,:,:],dims=1)
    lw = convert(Int64, round(nsamples * lower_bound))
    hi = convert(Int64, round(nsamples * upper_bound))
    
    ci_df = DataFrame(mean=mean(results.state.γ[nburn+1:total,:,:],dims=1)[1,:])
    ci_df[:,:lower_bound] = γ_sorted[lw,:,1]
    ci_df[:,:upper_bound] = γ_sorted[hi,:,1]

    q = size(ci_df,1)
    V = Int64((1 + sqrt( 1 + 8 * q))/2)

    CI_MAT = Matrix{ConfInt}(undef,V,V)
    MN_MAT = Matrix{Float64}(undef,V,V)

    i = 1
    for k = 1:V
        CI_MAT[k,k] = ConfInt(0,0,0)
        for l = k+1:V
            CI_MAT[l,k] = CI_MAT[k,l] = ConfInt(ci_df[i,:lower_bound], ci_df[i,:upper_bound],interval)
            MN_MAT[l,k] = MN_MAT[k,l] = ci_df[i,:mean]
            i += 1
        end
    end

    xi_df = DataFrame(probability=mean(results.state.ξ[nburn+1:total,:,:],dims=1)[1,:])

    println("Node Probabilities")
    display(round.(xi_df;digits))
    println("")
    println("---------------")
    println("Edge Coefficients Estimates")
    display(round.(MN_MAT;digits))
    println("")
    println("---------------")
    println("Edge Coefficients, $interval% C.I.s")
    display(round.(CI_MAT))

    return BNRSummary(MN_MAT,CI_MAT,xi_df)

end