#region custom sampling
"""
    sample_u!(ret, ξ, R, M)

Sample rows of the u matrix, either from MVN with mean 0 and covariance matrix M or a row of 0s

# Arguments
- `state` : all states, a vector of tuples (row-table) of all variables
- `i` : index of current state, used to index state variable
- `j` : index of the row of the u matrix to sample for
- `R` : dimension of u vectors, length of u to set
"""
function sample_u!(state::Table, i, j, R)
    if (state.ξ[1,i] == 1)
        state.u[i,:,j] = rand(MultivariateNormal(zeros(R), Matrix(state.M[i,:,:])))
    else
        state.u[i,:,j] = zeros(R)
    end
    nothing
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
    return sample_gig(1/2,b,a)
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

"""
    sample_π_dirichlet!(ret::AbstractVector{U},r,η,λ::AbstractVector{T})

Sample from the 3-variable doirichlet distribution with weights
[r^η,1,1] + [#{λ[r] == 0}, #{λ[r] == 1}, #{λ[r] = -1}]

# Arguments
- `state` : all states, a vector of tuples (row-table) of all variables
- `i` : index of current state, used to index state variable
- `r` : integer, base term for the first weight and index for λ vector
- `η` : real number, power term for the first weight
- `λ` : 1d array of -1,0,1s, used to determine which weight is added to

# Returns
A vector of length 3 drawn from the Dirichlet distribution
"""
function sample_π_dirichlet!(state::Table,i,r,η,λ::AbstractVector{T}) where {T}
    wts = [r^η,1,1]
    if λ[r] == 1
        wts[2] = 2
    elseif λ[r] == 0
        wts[1] = r^η + 1
    else
        wts[3] = 2
    end
    state.πᵥ[i,r,1:3] = rand(Dirichlet(wts))
    nothing
end
#endregion



"""
    initialize_variables!(state::Table, X_new::AbstractArray{U}, X::AbstractArray{T}, η, ζ, ι, R, aΔ, bΔ, ν, V=0, x_transform::Bool=true)

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
    - `ν` : hyperparameter used as the degrees of freedom parameter in the InverseWishart distribution used to sample M.
    - `V`: Value of V, the number of nodes in the original X matrix. Only input when x_transform is false. Always output.
    - `x_transform`: boolean, set to false if X has been pre-transformed into one row per sample. True by default.

    # Returns
    nothing
"""
function initialize_variables!(state::Table, X_new::AbstractArray{U}, X::AbstractArray{T}, η, ζ, ι, R, aΔ, bΔ, ν, V=0, x_transform::Bool=true) where {T,U}
    # η must be greater than 1, if it's not set it to its default value of 1.01
    if (η <= 1)
        η = 1.01
        println("η value invalid, reset to default of 1.01")
    end

    if x_transform
        V = Int64(size(X[1],1))
    end
    q = floor(Int,V*(V-1)/2)

    n = size(X,1)
    if x_transform
        for i in 1:n
            X_new[i,:] = lower_triangle(X[i])
        end
    else
        X_new[:,:] = X
    end

    state.θ[1] = rand(Gamma(ζ, 1/ι))

    state.S[1,:] = map(k -> rand(Exponential(state.θ[1]/2)), 1:q)
    D = Diagonal(state.S[1,:,1])

    state.πᵥ[1,:,:] = zeros(R,3)
    for r in 1:R
        state.πᵥ[1,r,:] = rand(Dirichlet([r^η,1,1]))
    end
    state.λ[1,:] = map(r -> sample([0,1,-1], StatsBase.weights(state.πᵥ[1,r,:]),1)[1], 1:R)
    Λ = Diagonal(state.λ[1,:])
    state.Δ[1] = sample_Beta(aΔ, bΔ)

    state.ξ[1,:] = map(k -> rand(Binomial(1,state.Δ[1])), 1:V)
    state.M[1,:,:] = rand(InverseWishart(ν,cholesky(Matrix(I,R,R))))
    for i in 1:V
        sample_u!(state,1,i,R)
    end
    state.μ[1] = 1.0
    state.τ²[1] = rand(Uniform(0,1))^2
    uᵀΛu = transpose(state.u[1,:,:]) * Λ * state.u[1,:,:]
    uᵀΛu_upper = reshape(lower_triangle(uᵀΛu),(q,))

    state.γ[1,:] = rand(MultivariateNormal(uᵀΛu_upper, state.τ²[1]*D))
    X_new
end


#region update variables

"""
    update_τ²!(state::Table, i, X::AbstractArray{T,2}, y::AbstractVector{U}, V)

Sample the next τ² value from the InverseGaussian distribution with mean n/2 + V(V-1)/4 and variance ((y - μ1 - Xγ)ᵀ(y - μ1 - Xγ) + (γ - W)ᵀD⁻¹(γ - W)

# Arguments
- `state` : all states, a vector of tuples (row-table) of all variables
- `i` : index of current state, used to index state variable
- `X` : 2 dimensional array of predictor values, 1 row per sample (upper triangle of original X)
- `y` : vector of response values
- `V` : dimension of original symmetric adjacency matrices

# Returns
nothing - updates are done in place
"""
function update_τ²!(state::Table, i, X::AbstractArray{T,2}, y::AbstractVector{U}, V) where {T,U}
    uᵀΛu = transpose(state.u[i-1,:,:]) * Diagonal(state.λ[i-1,:]) * state.u[i-1,:,:]
    W = lower_triangle(uᵀΛu)
    n  = size(y,1)

    #TODO: better variable names, not so much reassignment
    #TODO: a.tau and b.tau?
    μₜ  = (n/2) + (V*(V-1)/4)
    yμ1Xγ = (y - state.μ[i-1].*ones(n) - X*state.γ[i-1,:])

    γW = (state.γ[i-1,:] - W)
    yμ1Xγᵀyμ1Xγ = transpose(yμ1Xγ) * yμ1Xγ
    γWᵀγW = transpose(γW) * inv(Diagonal(state.S[i-1,:])) * γW

    σₜ² = (yμ1Xγᵀyμ1Xγ[1] + γWᵀγW[1])/2
    state.τ²[i] = rand(InverseGamma(μₜ, σₜ²))
    nothing
end

"""
    update_u_ξ!(state::Table, i, V)

Sample the next u and ξ values

# Arguments
- `state` : all states, a vector of tuples (row-table) of all variables
- `i` : index of current state, used to index state variable
- `V` : dimension of original symmetric adjacency matrices

# Returns
nothing - updates are done in place
"""
function update_u_ξ!(state::Table, i, V)
    w_top = zeros(V)
    for k in 1:V
        U = transpose(state.u[i-1,:,Not(k)]) * Diagonal(state.λ[i-1,:])
        s = create_lower_tri(state.S[i-1,:], V)
        Γ = create_lower_tri(state.γ[i-1,:], V)
        if k == 1
            γk=vcat(Γ[2:V,1])
            H = Diagonal(vcat(s[2:V,1]))
        elseif k == V
            γk=vcat(Γ[V,1:V-1])
            H = Diagonal(vcat(s[V,1:V-1]))
        else
            γk= vcat(Γ[k,1:k-1],Γ[k+1:V,k])
            H = Diagonal(vcat(s[k,1:k-1],s[k+1:V,k]))
        end
        Σ = inv(((transpose(U)*inv(H)*U)/state.τ²[i]) + inv(state.M[i-1,:,:]))
        m = Σ*(transpose(U)*inv(H)*γk)/state.τ²[i]

        mvn_a = MultivariateNormal(zeros(size(H,1)),Symmetric(state.τ²[i]*H))
        mvn_b_Σ = Symmetric(state.τ²[i] * H + U * state.M[i-1,:,:] * transpose(U))
        mvn_b = MultivariateNormal(zeros(size(H,1)),mvn_b_Σ)
        w_top = (1-state.Δ[i-1]) * pdf(mvn_a,γk)
        w_bot = w_top + state.Δ[i-1] * pdf(mvn_b,γk)
        w = w_top / w_bot

        mvn_f = zeros(size(Σ))
        try
            mvn_f = MultivariateNormal(m,Symmetric(Σ))

        catch 
            show(stdout,"text/plain",Symmetric(Σ))
        end

        state.ξ[i,k] = update_ξ(w)

        # the paper says the first term is (1-w) but their code uses ξ. Again i think this makes more sense
        # that this term would essentially be an indicator rather than a weight
        state.u[i,:,k] = state.ξ[i,k] .* rand(mvn_f)
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
    update_γ!(state::Table, i, X::AbstractArray{T,2}, y::AbstractVector{U}, n)

Sample the next γ value from the normal distribution, decomposed as described in Guha & Rodriguez 2018

# Arguments
- `state` : all states, a vector of tuples (row-table) of all variables
- `i` : index of current state, used to index state variable
- `X` : 2 dimensional array of predictor values, 1 row per sample (upper triangle of original X)
- `y` : response values
- `n` : number of samples

# Returns
nothing - all updates are done in place
"""
function update_γ!(state::Table, i, X::AbstractArray{T,2}, y::AbstractVector{U}, n) where {T,U}
    uᵀΛu = transpose(state.u[i,:,:]) * Diagonal(state.λ[i-1,:]) * state.u[i,:,:]
    W = lower_triangle(uᵀΛu)

    D = Diagonal(state.S[i-1,:])
    τ²D = state.τ²[i] * D
    q = size(D,1)

    Δᵧ₁ = rand(MultivariateNormal(zeros(q), (τ²D)))
    Δᵧ₂ = rand(MultivariateNormal(zeros(n), I(n)))
    Δᵧ₃ = (X / sqrt(state.τ²[i])) * Δᵧ₁ + Δᵧ₂
    one = τ²D * (transpose(X)/sqrt(state.τ²[i])) * inv(X * D * transpose(X) + I(n))
    two = (((y - state.μ[i-1] .* ones(n) - X * W) / sqrt(state.τ²[i])) - Δᵧ₃)
    γw = Δᵧ₁ + one * two
    γ = γw + W
    state.γ[i,:] = γ[:,1]
    nothing
end

"""
    update_D!(state::Table, i, V)

Sample the next D value from the GeneralizedInverseGaussian distribution with p = 1/2, a=((γ - uᵀΛu)^2)/τ², b=θ

# Arguments
- `state` : all states, a vector of tuples (row-table) of all variables
- `i` : index of current state, used to index state variable
- `V` : dimension of original symmetric adjacency matrices

# Returns
nothing - all updates are done in place
"""
function update_D!(state::Table, i, V)
    q = floor(Int,V*(V-1)/2)
    uᵀΛu = transpose(state.u[i,:,:]) * Diagonal(state.λ[i-1,:]) * state.u[i,:,:]
    uᵀΛu_upper = lower_triangle(uᵀΛu)
    a_ = (state.γ[i,:] - uᵀΛu_upper).^2 / state.τ²[i]
    state.S[i,:] = map(k -> sample_rgig(state.θ[i-1],a_[k]), 1:q)
    nothing
end

"""
    update_θ!(state::Table, i, ζ, ι, V)

Sample the next θ value from the Gamma distribution with a = ζ + V(V-1)/2 and b = ι + ∑(s[k,l]/2)

# Arguments
- `state` : all states, a vector of tuples (row-table) of all variables
- `i` : index of current state, used to index state variable
- `ζ` : hyperparameter, used to construct `a` parameter
- `ι` : hyperparameter, used to construct `b` parameter
- `V` : dimension of original symmetric adjacency matrices

# Returns
nothing - all updates are done in place
"""
function update_θ!(state::Table, i, ζ, ι, V)
    a = ζ + (V*(V-1))/2
    b = ι + sum(state.S[i,:])/2
    state.θ[i] = rand(Gamma(a,1/b))
    nothing
end

"""
    update_Δ!(state::Table, i, aΔ, bΔ)

Sample the next Δ value from the Beta distribution with parameters a = aΔ + ∑ξ and b = bΔ + V - ∑ξ

# Arguments
- `state` : all states, a vector of tuples (row-table) of all variables
- `i` : index of current state, used to index state variable
- `aΔ`: hyperparameter used as part of the a parameter in the beta distribution used to sample Δ.
- `bΔ`: hyperparameter used as part of the b parameter in the beta distribution used to sample Δ.

# Returns
nothing - all updates are done in place
"""
function update_Δ!(state::Table, i, aΔ, bΔ)
    a = aΔ + sum(state.ξ[i,:])
    b = bΔ + sum(1 .- state.ξ[i,:])
    state.Δ[i] = sample_Beta(a,b)
    nothing
end

"""
    update_M!(state::Table, i, ν, V)

Sample the next M value from the InverseWishart distribution with df = V + # of nonzero columns in u and Ψ = I + ∑ uΛuᵀ

# Arguments
- `state` : all states, a vector of tuples (row-table) of all variables
- `i` : index of current state, used to index state variable
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
        if state.ξ[i,v] ≉ 0
            num_nonzero = num_nonzero + 1
        end
    end
    Ψ = I(R) + uuᵀ
    df = ν + num_nonzero
    state.M[i,:,:] = rand(InverseWishart(df,cholesky(Matrix(Ψ))))
    nothing
end

"""
    update_μ!(state::Table, i, X::AbstractArray{T,2}, y::AbstractVector{U}, n)

Sample the next μ value from the normal distribution with mean 1ᵀ(y - Xγ)/n and variance τ²/n

# Arguments
- `state` : all states, a vector of tuples (row-table) of all variables
- `i` : index of current state, used to index state variable
- `X` : 2 dimensional array of predictor values, 1 row per sample (upper triangle of original X)
- `y` : response values
- `n` : number of samples (length of y)

# Returns
nothing - all updates are done in place
"""
function update_μ!(state::Table, i, X::AbstractArray{T,2}, y::AbstractVector{U}, n) where {T,U}
    μₘ = (ones(1,n) * (y .- X * state.γ[i,:])) / n
    σₘ = sqrt(state.τ²[i]/n)
    state.μ[i] = rand(Normal(μₘ[1],σₘ))
    nothing
end

"""
    update_Λ!(state::Table, i, R)

Sample the next values of λ from [1,0,-1] with probabilities determined from a normal mixture

# Arguments
- `state` : all states, a vector of tuples (row-table) of all variables
- `i` : index of current state, used to index state variable
- `R` : the dimensionality of the latent variables u

# Returns
nothing - all updates are done in place
"""
function update_Λ!(state::Table, i, R)
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

        n₀ = prod(map(j -> pdf(Normal(W₀[j],sqrt(τ²D[j,j])),state.γ[i,j]),1:q))
        n₁ = prod(map(j -> pdf(Normal(W₁[j],sqrt(τ²D[j,j])),state.γ[i,j]),1:q))
        n₋₁ = prod(map(j -> pdf(Normal(W₋₁[j],sqrt(τ²D[j,j])),state.γ[i,j]),1:q))
        p_bot = state.πᵥ[i-1,r,1] * n₀ + state.πᵥ[i-1,r,2] * n₁ + state.πᵥ[i-1,r,3] * n₋₁
        p1 = state.πᵥ[i-1,r,1] * n₀ / p_bot
        p2 = state.πᵥ[i-1,r,2] * n₁ / p_bot
        p3 = state.πᵥ[i-1,r,3] * n₋₁ / p_bot
        state.λ[i,r] = sample([0,1,-1],StatsBase.weights([p1,p2,p3]))
    end
    nothing
end

"""
    update_π!(state::Table,i,η,R)

Sample the new values of πᵥ from the Dirichlet distribution with parameters [1 + #{r: λᵣ= 1}, #{r: λᵣ = 0} + r^η, 1 + #{r: λᵣ = -1 }]

# Arguments
- `state` : all states, a vector of tuples (row-table) of all variables
- `i` : index of current state, used to index state variable
- `η` : hyperparameter used for sampling the 0 value (r^η)
- `R` : dimension of u vectors

# Returns
new value of πᵥ
"""
function update_π!(state::Table,i,η,R)
    for r in 1:R
        sample_π_dirichlet!(state,i,r,η,state.λ[i,:])
    end
    nothing
end

#endregion

"""
    GibbsSample!(state::Table, iteration, X::AbstractArray{U,2}, y::AbstractVector{S}, V, η, ζ, ι, R, aΔ, bΔ, ν)

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

    if ν < R
        ArgumentError("ν value ($ν) must be greater than R value ($R)")
    elseif ν == R
        println("Warning: ν==R may give poor accuracy. Consider increaseing ν")
    end

    n = size(X,1)
    q = Int64(V*(V-1)/2)
    total = nburn + nsamples + 1
    p = Progress(total-1,1)

    state = Table(τ² = Array{Float64,3}(undef,(total,1,1)), u = Array{Float64,3}(undef,(total,R,V)),
                  ξ = Array{Float64,3}(undef,(total,V,1)), γ = Array{Float64,3}(undef,(total,q,1)),
                  S = Array{Float64,3}(undef,(total,q,1)), θ = Array{Float64,3}(undef,(total,1,1)),
                  Δ = Array{Float64,3}(undef,(total,1,1)), M = Array{Float64,3}(undef,(total,R,R)),
                  μ = Array{Float64,3}(undef,(total,1,1)), λ = Array{Float64,3}(undef,(total,R,1)),
                  πᵥ= Array{Float64,3}(undef,(total,R,3)))

    X_new = Matrix{eltype(T)}(undef, n, q)
    initialize_variables!(state, X_new, X, η, ζ, ι, R, aΔ, bΔ, ν, V, x_transform)
    for i in 2:total
        GibbsSample!(state, i, X_new, y, V, η, ζ, ι, R, aΔ, bΔ, ν)
        next!(p)
    end
    return state
end
