#region BNRPosteriors
struct BNRPosteriors{T<:AbstractFloat,U<:Int}
    Gammas::Array{Vector{T},1}
    Xis::Array{Vector{U},1}
    us::Array{Array{T,2},1}
end

#endregion

#region custom sampling
"""
    sample_u!(ret, ξ, R, M)

Sample rows of the u matrix, either from MVN with mean 0 and covariance matrix M or a row of 0s

# Arguments
- `ret` : vector of size R, output
- `ξ` : ξ value sampled from Binomial distribution. Set to 0 to return a row of 0s, 1 to sample from MVN
- `R` : dimension of u vectors, length of return vector
- `M` : R×R covariance matrix for MVN samples
"""
function sample_u!(ret::AbstractArray{U,1},ξ, R, M::AbstractArray{T,2}) where {T,U}
    if (ξ == 1)
        ret[1:size(ret,1)] = rand(MultivariateNormal(zeros(R), M))
    else
        ret[1:size(ret,1)] = zeros(R)
    end
end

"""
    sample_rgig(a,b)

Sample from the GeneralizedInverseGaussian distribution with p=1/2, b=b, a=a

# Arguments
- `a` : shape and scale parameter a, sometimes also called ψ
- `b` : shape and scale parameter b, sometimes also called χ

# Returns
one sample from the GIG distribution with p=1/2, b=b, a=a
"""
function sample_rgig(a,b)::Float64
    return rand(GeneralizedInverseGaussian(a,b,1/2))
end

"""
    sample_Beta(a,b)

Sample from the Beta distribution, with handling for a=0 and/or b=0

#Arguments
- `a` : shape parameter a ≥ 0
- `b` : shape parameter b ≥ 0
"""
function sample_Beta(a,b)
    Δ = 0.0
    if a > 0.0 && b > 0.0
        Δ = rand(Beta(a, b))
    elseif a > 0.0
        Δ = 1.0
    elseif b > 0.0
        Δ = 0.0
    else
        Δ = sample([0.0,1.0])
    end
    return Δ
end
#endregion


#region helper functions

"""
    sample_π_dirichlet!(ret, r,η,λ)

Sample from the 3-variable doirichlet distribution with weights
[r^η,1,1] + [#{λ[r] == 0}, #{λ[r] == 1}, #{λ[r] = -1}]

# Arguments
- `ret` : return vector of length 3
- `r` : integer, base term for the first weight and index for λ vector
- `η` : real number, power term for the first weight
- `λ` : 1d array of -1,0,1s, used to determine which weight is added to

# Returns
A vector of length 3 drawn from the Dirichlet distribution
"""
function sample_π_dirichlet!(ret::AbstractVector{U},r,η,λ::AbstractVector{T}) where {T,U}
    wts = [r^η,1,1]
    if λ[r] == 1
        wts[2] = 2
    elseif λ[r] == 0
        wts[1] = r^η + 1
    else
        wts[3] = 2
    end
    ret[1:3] = rand(Dirichlet(wts))
    nothing
end
#endregion



"""
    initialize_variables!(state,X, η, ζ, ι, R, aΔ, bΔ, ν, V, x_transform)

    Initialize all variables using prior distributions. Note, if x_transform is true V will be ignored and overwritten with the implied value from X.
    All initializations done in place on the state argument.

    # Arguments
    - `state` : a row-table structure containing all past states, the current state, and space for the future states
    - `X` : vector of unweighted symmetric adjacency matrices to be used as predictors. each element of the array should be 1 matrix
    - `η` : hyperparameter used to sample from the Dirichlet distribution (r^η)
    - `ζ` : hyperparameter used as the shape parameter in the gamma distribution used to sample θ
    - `ι` : hyperparameter used as the scale parameter in the gamma distribution used to sample θ
    - `R` : the dimensionality of the latent variables u, a hyperparameter
    - `aΔ`: hyperparameter used as the a parameter in the beta distribution used to sample Δ.
    - `bΔ`: hyperparameter used as the b parameter in the beta distribution used to sample Δ. aΔ and bΔ values causing the Beta distribution to have mass concentrated closer to 0 will cause more zeros in ξ
    - `ν` : hyperparameter used as the degrees of freedom parameter in the InverseWishart distribution used to sample M.
    - `V`: Value of V, the number of nodes in the original X matrix. Only input when x_transform is false. Always output.
    - `x_transform`: boolean, set to false if X has been pre-transformed into one row per sample. True by default.

    # Returns
    new X matrix
"""
function initialize_variables!(state::Table, X::AbstractArray{T}, η, ζ, ι, R, aΔ, bΔ, ν, V=0, x_transform::Bool=true) where {T}
    # η must be greater than 1, if it's not set it to its default value of 1.01
    if (η <= 1)
        η = 1.01
        println("η value invalid, reset to default of 1.01")
    end

    if x_transform
        V = Int64(size(X[1],1))
    end
    q = floor(Int,V*(V-1)/2)

    X_new = Matrix{Float64}(undef, size(X,1), q)
    if x_transform
        for i in 1:size(X,1)
            X_new[i,:] = upper_triangle(X[i])
        end
    else
        X_new = X
    end

    state.θ[1,:,:] = rand(Gamma(ζ, 1/ι))

    state.S[1,:,:] = map(k -> rand(Exponential(state.θ[1]/2)), 1:q)
    D = Diagonal(state.S[1,:,1])
    
    state.πᵥ[1,:,:] = zeros(R,3)
    for r in 1:R
        state.πᵥ[1,r,:] = rand(Dirichlet([r^η,1,1]))
    end
    state.λ[1,:,:] = map(r -> sample([0,1,-1], StatsBase.weights(state.πᵥ[1,r,:]),1)[1], 1:R)
    Λ = Diagonal(state.λ[1,:,1])
    state.Δ[1,:,:] = sample_Beta(aΔ, bΔ)

    state.ξ[1,:,:] = map(k -> rand(Binomial(1,state.Δ[1])), 1:V)
    state.M[1,:,:] = rand(InverseWishart(ν,cholesky(Matrix(I,R,R))))
    for i in 1:V
        ret = zeros(R)
        sample_u!(ret,state.ξ[1,i,1],R,Matrix(state.M[1,:,:]))
        state.u[1,:,i] = ret
    end
    state.μ[1,:,:] = 1.0
    state.τ²[1,:,:] = rand(Uniform(0,1))^2
    uᵀΛu = transpose(state.u[1,:,:]) * Λ * state.u[1,:,:]
    uᵀΛu_upper = reshape(upper_triangle(uᵀΛu),(q,))

    state.γ[1,:,:] = rand(MultivariateNormal(uᵀΛu_upper, state.τ²[1]*D))
    X_new
end


#region update variables

"""
    update_τ²!(state::Table, l_state::Table, X::AbstractArray{T,2}, y::AbstractVector{U}, V)

Sample the next τ² value from the InverseGaussian distribution with mean n/2 + V(V-1)/4 and variance ((y - μ1 - Xγ)ᵀ(y - μ1 - Xγ) + (γ - W)ᵀD⁻¹(γ - W)

# Arguments
- `state` : current state, a tuple of all variables (current row of the state table)
- `X` : 2 dimensional array of predictor values, 1 row per sample (upper triangle of original X)
- `y` : vector of response values
- `V` : dimension of original symmetric adjacency matrices

# Returns
nothing - updates are done in place
"""
function update_τ²!(state::Table, i, X::AbstractArray{T,2}, y::AbstractVector{U}, V) where {T,U}
    uᵀΛu = transpose(state.u[i-1,:,:]) * Diagonal(state.λ[i-1,:,1]) * state.u[i-1,:,:]
    W = upper_triangle(uᵀΛu)
    n  = size(y,1)

    #TODO: better variable names, not so much reassignment
    #TODO: a.tau and b.tau?
    μₜ  = (n/2) + (V*(V-1)/4)
    yμ1Xγ = (y - state.μ[i-1].*ones(n,1) - X*state.γ[i-1,:,1])

    γW = (state.γ[i-1,:,:] - W)
    yμ1Xγᵀyμ1Xγ = transpose(yμ1Xγ) * yμ1Xγ
    γWᵀγW = transpose(γW) * inv(Diagonal(state.S[i-1,:,1])) * γW

    σₜ² = (yμ1Xγᵀyμ1Xγ[1] + γWᵀγW[1])/2
    state.τ²[i,:,:] = rand(InverseGamma(μₜ, σₜ²))
    nothing
end

"""
    update_u_ξ!(c_state, l_state, V)

Sample the next u and ξ values

# Arguments
- `state` : current state, a tuple of all variables (current row of the state table)
- `V` : dimension of original symmetric adjacency matrices

# Returns
nothing - updates are done in place
"""
function update_u_ξ!(state::Table, i, V)
    w_top = zeros(V)
    for k in 1:V
        U = transpose(state.u[i-1,:,Not(k)]) * Diagonal(state.λ[i-1,:,1])
        s = create_upper_tri(state.S[i-1,:,:],V)
        Γ = create_upper_tri(state.γ[i-1,:,:], V)
        if k == 1
            γk=vcat(Γ[1,2:V])
            H = Diagonal(vcat(s[1,2:V]))
        elseif k == V
            γk=vcat(Γ[1:V-1,V])
            H = Diagonal(vcat(s[1:V-1,V]))
        else
            γk= vcat(Γ[1:k-1,k],Γ[k,k+1:V])
            H = Diagonal(vcat(s[1:k-1,k],s[k,k+1:V]))
        end
        Σ = inv(((transpose(U)*inv(H)*U)/state.τ²[i]) + inv(state.M[i-1,:,:]))
        m = Σ*(transpose(U)*inv(H)*γk)/state.τ²[i]

        mvn_a = MultivariateNormal(zeros(size(H,1)),Symmetric(state.τ²[i]*H))
        mvn_b_Σ = Symmetric(state.τ²[i] * H + U * state.M[i-1,:,:] * transpose(U))
        mvn_b = MultivariateNormal(zeros(size(H,1)),mvn_b_Σ)
        w_top = (1-state.Δ[i-1]) * pdf(mvn_a,γk)
        w_bot = w_top + state.Δ[i-1] * pdf(mvn_b,γk)
        w = w_top / w_bot

        mvn_f = MultivariateNormal(m,Symmetric(Σ))

        state.ξ[i,k,:] = update_ξ(w)

        # the paper says the first term is (1-w) but their code uses ξ. Again i think this makes more sense
        # that this term would essentially be an indicator rather than a weight
        state.u[i,:,k] = state.ξ[i,k,1] .* rand(mvn_f)
    end
    nothing
end

"""
    update_ξ(w)

Sample the next ξ value from the Bernoulli distribution with parameter 1-w

# Arguments
- `w` : parameter to use for sampling, probability that 0 is drawn

# Returns
the new value of ξ
"""
function update_ξ(w)
    if w == 0
        return 1
    elseif w == 1
        return 0
    end
    Int64(rand(Bernoulli(1 - w)))
end

"""
    update_γ!(c_state, l_state, X, y, n)

Sample the next γ value from the normal distribution, decomposed as described in Guha & Rodriguez 2018

# Arguments
- `c_state` : current state, a tuple of all variables (current row of the state table)
- `l_state` : last state, a tuple of all variables (previous row of the state table)
- `X` : 2 dimensional array of predictor values, 1 row per sample (upper triangle of original X)
- `y` : response values
- `n` : number of samples

# Returns
nothing - all updates are done in place
"""
function update_γ!(state::Table, i, X::AbstractArray{T,2}, y::AbstractVector{U}, n) where {T,U}
    uᵀΛu = transpose(state.u[i,:,:]) * Diagonal(state.λ[i-1,:,1]) * state.u[i,:,:]
    W = upper_triangle(uᵀΛu)

    D = Diagonal(state.S[i-1,:,1])
    τ²D = state.τ²[i] * D
    q = size(D,1)

    Δᵧ₁ = rand(MultivariateNormal(zeros(q), (τ²D)))
    Δᵧ₂ = rand(MultivariateNormal(zeros(n), I(n)))
    Δᵧ₃ = (X / sqrt(state.τ²[i])) * Δᵧ₁ + Δᵧ₂
    one = τ²D * (transpose(X)/sqrt(state.τ²[i])) * inv(X * D * transpose(X) + I(n))
    two = (((y - state.μ[i-1] .* ones(n,1) - X * W) / sqrt(state.τ²[i])) - Δᵧ₃)
    γw = Δᵧ₁ + one * two
    γ = γw + W
    state.γ[i,:,:] = γ[:,1]
    nothing
end

"""
    update_D!(c_state, l_state V)

Sample the next D value from the GeneralizedInverseGaussian distribution with p = 1/2, a=((γ - uᵀΛu)^2)/τ², b=θ

# Arguments
- `c_state` : current state, a tuple of all variables (current row of the state table)
- `l_state` : last state, a tuple of all variables (previous row of the state table)
- `V` : dimension of original symmetric adjacency matrices

# Returns
nothing - all updates are done in place
"""
function update_D!(state::Table, i, V)
    q = floor(Int,V*(V-1)/2)
    uᵀΛu = transpose(state.u[i,:,:]) * Diagonal(state.λ[i-1,:,1]) * state.u[i,:,:]
    uᵀΛu_upper = upper_triangle(uᵀΛu)
    a_ = (state.γ[i,:,:] - uᵀΛu_upper).^2 / state.τ²[i]
    state.S[i,:,:] = map(k -> sample_rgig(state.θ[i-1],a_[k]), 1:q)
    nothing
end

"""
    update_θ!(c_state, ζ, ι, V)

Sample the next θ value from the Gamma distribution with a = ζ + V(V-1)/2 and b = ι + ∑(s[k,l]/2)

# Arguments
- `c_state` : current state, a tuple of all variables (current row of the state table)
- `ζ` : hyperparameter, used to construct `a` parameter
- `ι` : hyperparameter, used to construct `b` parameter
- `V` : dimension of original symmetric adjacency matrices

# Returns
nothing - all updates are done in place
"""
function update_θ!(state::Table, i, ζ, ι, V)
    a = ζ + (V*(V-1))/2
    b = ι + sum(state.S[i,:,:])/2
    state.θ[i,:,:] = rand(Gamma(a,1/b))
    nothing
end

"""
    update_Δ!(c_state, aΔ, bΔ)

Sample the next Δ value from the Beta distribution with parameters a = aΔ + ∑ξ and b = bΔ + V - ∑ξ

# Arguments
- `c_state` : current state, a tuple of all variables (current row of the state table)
- `aΔ`: hyperparameter used as part of the a parameter in the beta distribution used to sample Δ.
- `bΔ`: hyperparameter used as part of the b parameter in the beta distribution used to sample Δ.

# Returns
nothing - all updates are done in place
"""
function update_Δ!(state::Table, i, aΔ, bΔ)
    a = aΔ + sum(state.ξ[i,:,1])
    b = bΔ + sum(1 .- state.ξ[i,:,1])
    state.Δ[i,:,:] = sample_Beta(a,b)
    nothing
end

"""
    update_M!(c_state,ν,V)

Sample the next M value from the InverseWishart distribution with df = V + # of nonzero columns in u and Ψ = I + ∑ uΛuᵀ

# Arguments
- `c_state` : current state, a tuple of all variables (current row of the state table)
- `ν` : hyperparameter, base df for IW distribution (to be added to by sum of ξs)
- `V` : dimension of original symmetric adjacency matrices

# Returns
nothing - all updates are done in place
"""
function update_M!(state::Table, i, ν, V)
    R = size(state.u[i,:,:],1)
    uuᵀ = zeros(R,R)
    num_nonzero = 0
    for v in 1:V
        uuᵀ = uuᵀ + state.u[i,:,v] * transpose(state.u[i,:,v])
        if state.ξ[i,v,1] ≉ 0
            num_nonzero = num_nonzero + 1
        end
    end
    Ψ = I(R) + uuᵀ
    df = ν + num_nonzero
    state.M[i,:,:] = rand(InverseWishart(df,cholesky(Matrix(Ψ))))
    nothing
end

"""
    update_μ!(c_state, X, y, n)

Sample the next μ value from the normal distribution with mean 1ᵀ(y - Xγ)/n and variance τ²/n

# Arguments
- `c_state` : current state, a tuple of all variables (current row of the state table)
- `i` : 
- `X` : 2 dimensional array of predictor values, 1 row per sample (upper triangle of original X)
- `y` : response values
- `n` : number of samples (length of y)

# Returns
nothing - all updates are done in place
"""
function update_μ!(state::Table, i, X::AbstractArray{T,2}, y::AbstractVector{U}, n) where {T,U}
    μₘ = (ones(1,n) * (y .- X * state.γ[i,:,1])) / n
    σₘ = sqrt(state.τ²[i]/n)
    state.μ[i,:,:] = rand(Normal(μₘ[1],σₘ))
    nothing
end

"""
    update_Λ(c_state, l_state, R)

Sample the next values of λ from [1,0,-1] with probabilities determined from a normal mixture

# Arguments
- `c_state` : current state, a tuple of all variables (current row of the state table)
- `l_state` : last state, a tuple of all variables (previous row of the state table)
- `R` : the dimensionality of the latent variables u

# Returns
nothing - all updates are done in place
"""
function update_Λ!(state::Table, i, R)
    Λ = Diagonal(state.λ[i-1,:,1])
    q = size(state.γ[i,:,1],1)
    τ²D = state.τ²[i] * Diagonal(state.S[i,:,1])
    for r in 1:R
        Λ₋₁= deepcopy(Λ)
        Λ₋₁[r,r] = -1
        Λ₀ = deepcopy(Λ)
        Λ₀[r,r] = 0
        Λ₁ = deepcopy(Λ)
        Λ₁[r,r] = 1
        u_tr = transpose(state.u[i,:,:])
        W₋₁= upper_triangle(u_tr * Λ₋₁ * state.u[i,:,:])
        W₀ = upper_triangle(u_tr * Λ₀ * state.u[i,:,:])
        W₁ = upper_triangle(u_tr * Λ₁ * state.u[i,:,:])

        n₀ = prod(map(j -> pdf(Normal(W₀[j],sqrt(τ²D[j,j])),state.γ[i,j,1]),1:q))
        n₁ = prod(map(j -> pdf(Normal(W₁[j],sqrt(τ²D[j,j])),state.γ[i,j,1]),1:q))
        n₋₁ = prod(map(j -> pdf(Normal(W₋₁[j],sqrt(τ²D[j,j])),state.γ[i,j,1]),1:q))
        p_bot = state.πᵥ[i-1,r,1] * n₀ + state.πᵥ[i-1,r,2] * n₁ + state.πᵥ[i-1,r,3] * n₋₁
        p1 = state.πᵥ[i-1,r,1] * n₀ / p_bot
        p2 = state.πᵥ[i-1,r,2] * n₁ / p_bot
        p3 = state.πᵥ[i-1,r,3] * n₋₁ / p_bot
        state.λ[i,r,:] = sample([0,1,-1],StatsBase.weights([p1,p2,p3]))
    end
    nothing
end

"""
    update_π(c_state,η,R)

Sample the new values of πᵥ from the Dirichlet distribution with parameters [1 + #{r: λᵣ= 1}, #{r: λᵣ = 0} + r^η, 1 + #{r: λᵣ = -1 }]

# Arguments
- `c_state` : current state, a tuple of all variables (current row of the state table)
- `η` : hyperparameter used for sampling the 0 value (r^η)
- `R` : dimension of u vectors

# Returns
new value of πᵥ
"""
function update_π!(state::Table,i,η,R)
    for r in 1:R
        ret = zeros(3)
        sample_π_dirichlet!(ret,r,η,state.λ[i,:,1])
        state.πᵥ[i,r,:] = ret
    end
    nothing
end

#endregion

"""
    GibbsSample!(state, iteration, X, y, V, η, ζ, ι, R, aΔ, bΔ, ν)

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

# Returns:
nothing, all updating is done in place
"""
function GibbsSample!(state::Table, iteration, X::AbstractArray{U,2}, y::AbstractVector{S}, V, η, ζ, ι, R, aΔ, bΔ, ν) where {S,U}
    n = size(X,1)
    q = Int64(V*(V-1)/2)

    update_τ²!(state, iteration, X, y, V)
    update_u_ξ!(state, iteration, V)
    update_γ!(state, iteration, X, y, n)
    update_D!(state, iteration, V)
    update_θ!(state, iteration, ζ, ι, V)
    update_Δ!(state, iteration, aΔ, bΔ)
    update_M!(state, iteration, ν, V)
    update_μ!(state, iteration, X, y, n)
    update_Λ!(state, iteration, R)
    update_π!(state, iteration, η, R)
    nothing
end



function GenerateSamples!(X::AbstractArray{T,2}, y::AbstractVector{U}, R; η=1.01,ζ=1.0,ι=1.0,aΔ=1.0,bΔ=1.0, ν=12, nburn=30000, nsamples=20000, V=0, x_transform=true) where {T,U}
    if V == 0 && !x_transform
        ArgumentError("If x_transform is false a valid V value must be given")
    end
    
    q = Int64(V*(V-1)/2)
    total = nburn + nsamples + 1
    p = Progress(total-1,1)
   
    state = Table(τ² = MArray{Tuple{total,1,1},Float64}(undef), u = MArray{Tuple{total,R,V},Float64}(undef), 
                  ξ = MArray{Tuple{total,V,1},Float64}(undef), γ = MArray{Tuple{total,q,1},Float64}(undef),
                  S = MArray{Tuple{total,q,1},Float64}(undef), θ = MArray{Tuple{total,1,1},Float64}(undef),
                  Δ = MArray{Tuple{total,1,1},Float64}(undef), M = MArray{Tuple{total,R,R},Float64}(undef),
                  μ = MArray{Tuple{total,1,1},Float64}(undef), λ = MArray{Tuple{total,R,1},Float64}(undef),
                  πᵥ= MArray{Tuple{total,R,3},Float64}(undef))

    X = initialize_variables!(state, X, η, ζ, ι, R, aΔ, bΔ, ν, V, x_transform)
    for i in 2:total
        GibbsSample!(state, i, X, y, V, η, ζ, ι, R, aΔ, bΔ, ν)
        next!(p)
    end
    return state
end
