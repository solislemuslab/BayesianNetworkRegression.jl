#region BNRPosteriors
struct BNRPosteriors{T<:AbstractFloat,U<:Int}
    Gammas::Array{Array{T,1},1}
    Xis::Array{Array{U,1},1}
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
function sample_u!(ret::AbstractVector{U},ξ, R, M::AbstractArray{T,2}) where {T,U}
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
end
#endregion

"""
    init_vars!(X, η, ζ, ι, R, aΔ, bΔ, ν, V, x_transform)

    Initialize all variables using prior distributions. Note, if x_transform is true V will be ignored and overwritten with the implied value from X.

    # Arguments
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
    - `X` : matrix of re-ordered predictors. one row per sample, V*(V-1)/2 columns
    - `θ` : set to 1 draw from Gamma(ζ,ι)
    - `D` : set to a diagonal matrix of V(V-1)/2 draws from the Exponential(θ/2) distribution
    - `πᵥ`: set to a R × 3 matrix of draws from the Dirichlet distribution, where the second and third columns are draws from Dirichlet(1) and the first are draws from Dirichlet(r^η)
    - `Λ` : R × R diagonal matrix of λ values, which are sampled from [0,1,-1] with probabilities assigned from the rth row of πᵥ
    - `Δ` : set to 1 draw from Beta(aΔ, bΔ) (or 1 if aΔ or bΔ are 0).
    - `ξ` : set to V draws from Bernoulli(Δ)
    - `M` : set to R × R matrix drawn from InverseWishart(ν, I_R) (where I_R is the identity matrix with dimension R × R)
    - `u` : the latent variables u, set to a V × R matrix where each row is sampled from MultivariateNormal(0,M) if ξ[r] is 1 or set to a row of 1s otherwise.
    - `μ` : set to 1 (non-informative prior)
    - `τ²`: set to the square of 1 sample from Uniform(0,1) (non-informative prior)
    - `γ` : set to 1 draw from MultivariateNormal(uᵀΛu_upper, τ²*s), where uᵀΛu_upper is a vector of the terms in the upper triangle of uᵀΛu (does not include the diagonal)
    - `V` : dimension of original symmetric adjacency matrices
"""
function init_vars!(X::AbstractArray{T}, η, ζ, ι, R, aΔ, bΔ, ν, V=0, x_transform::Bool=true) where {T}
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

    θ = rand(Gamma(ζ, 1/ι))

    S = map(k -> rand(Exponential(θ/2)), 1:q)

    D = Diagonal(S)
    πᵥ = zeros(R,3)
    for r in 1:R
        πᵥ[r,:] = rand(Dirichlet([r^η,1,1]))
    end
    λ = map(r -> sample([0,1,-1], StatsBase.weights(πᵥ[r,:]),1)[1], 1:R)
    Λ = Diagonal(λ)
    Δ = sample_Beta(aΔ, bΔ)

    ξ = map(k -> rand(Binomial(1,Δ)), 1:V)
    M = rand(InverseWishart(ν,cholesky(Matrix(I,R,R))))
    u = zeros(R,V)
    u_ret = Vector{AbstractFloat}(undef,R)
    for i in 1:V
        sample_u!(u_ret,ξ[i],R,M)
        u[:,i] = u_ret
    end
    μ = 1.0
    τ²= rand(Uniform(0,1))^2
    uᵀΛu = transpose(u) * Λ * u
    uᵀΛu_upper = upper_triangle(uᵀΛu)

    γ = rand(MultivariateNormal(uᵀΛu_upper, τ²*D))
    return (X_new, θ, D, πᵥ, Λ, Δ, ξ, M, u, μ, τ², γ)
end

#region update variables
"""
    update_μ(y, X, γ, τ², n)

Sample the next μ value from the normal distribution with mean 1ᵀ(y - Xγ)/n and variance τ²/n

# Arguments
- `X` : 2 dimensional array of predictor values, 1 row per sample (upper triangle of original X)
- `y` : response values
- `γ` : vector of regression parameters
- `τ²`: overall variance parameter
- `n` : number of samples (length of y)

# Returns
new value of μ
"""
function update_μ(X::AbstractArray{T,2}, y::AbstractVector{U}, γ::AbstractVector{S}, τ², n::Int) where {S,T,U}
    μₘ = (ones(1,n) * (y .- X*γ)) / n
    σₘ = sqrt(τ²/n)
    rand(Normal(μₘ[1],σₘ))
end

"""
    update_γ!(ret,X, y, D, Λ, u, μ, τ²)

Sample the next γ value from the normal distribution, decomposed as described in Guha & Rodriguez 2018

# Arguments
- `ret` : return vector of length size(γ,1)
- `X` : 2 dimensional array of predictor values, 1 row per sample (upper triangle of original X)
- `y` : response values
- `D` : diagonal matrix of s values
- `Λ` : R × R diagonal matrix of λ values
- `u` : the latent variables u
- `μ` : overall mean value for the relationship
- `τ²`: overall variance parameter
- `n` : number of samples

# Returns
new value of γ
"""
function update_γ!(ret::AbstractVector{O},X::AbstractArray{T,2}, y::AbstractVector{U}, D::Diagonal{Q,Vector{Q}}, Λ::Diagonal{P,Vector{P}}, u::AbstractArray{S,2}, μ, τ², n) where {O,P,Q,S,T,U}
    uᵀΛu = transpose(u) * Λ * u
    W = upper_triangle(uᵀΛu)
    q = size(D,1)

    τ²D = τ²*D

    Δᵧ₁ = rand(MultivariateNormal(zeros(q), (τ²D)))
    Δᵧ₂ = rand(MultivariateNormal(zeros(n), I(n)))
    Δᵧ₃ = (X / sqrt(τ²)) * Δᵧ₁ + Δᵧ₂
    one = τ²D * (transpose(X)/sqrt(τ²)) * inv(X * D * transpose(X) + I(n))
    two = (((y - μ .* ones(n,1) - X * W) / sqrt(τ²)) - Δᵧ₃)
    γw = Δᵧ₁ + one * two
    γ = γw + W
    ret[1:size(ret,1)] = γ[:,1]
end

"""
    update_τ²(X, y, μ, γ, Λ, u, D, V)

Sample the next τ² value from the InverseGaussian distribution with mean n/2 + V(V-1)/4 and variance ((y - μ1 - Xγ)ᵀ(y - μ1 - Xγ) + (γ - W)ᵀD⁻¹(γ - W)

# Arguments
- `X` : 2 dimensional array of predictor values, 1 row per sample (upper triangle of original X)
- `y` : vector of response values
- `μ` : overall mean value for the relationship
- `γ` : vector of regression parameters
- `Λ` : R × R diagonal matrix of λ values
- `u` : the latent variables u
- `D` : diagonal matrix of s values
- `V` : dimension of original symmetric adjacency matrices

# Returns
new value of τ²
"""
function update_τ²(X::AbstractArray{T,2}, y::AbstractVector{U}, μ, γ::AbstractVector{Q}, Λ::Diagonal{P,Vector{P}}, u::AbstractArray{S,2}, D::Diagonal{O,Vector{O}}, V) where {O,P,Q,S,T,U}
    uᵀΛu = transpose(u) * Λ * u
    W = upper_triangle(uᵀΛu)
    n  = size(y,1)

    #TODO: better variable names, not so much reassignment
    μₜ  = (n/2) + (V*(V-1)/4)
    yμ1Xγ = (y - μ.*ones(n,1) - X*γ)

    γW = (γ - W)
    yμ1Xγᵀyμ1Xγ = transpose(yμ1Xγ) * yμ1Xγ
    γWᵀγW = transpose(γW) * inv(D) * γW

    σₜ² = (yμ1Xγᵀyμ1Xγ[1] + γWᵀγW[1])/2
    rand(InverseGamma(μₜ, σₜ²))
end

"""
    update_D!(D_out,γ, u, Λ, θ, τ², V)

Sample the next D value from the GeneralizedInverseGaussian distribution with p = 1/2, a=((γ - uᵀΛu)^2)/τ², b=θ

# Arguments
- `D_out`: Diagonal output variable
- `γ` : vector of regression parameters
- `u` : the latent variables u
- `Λ` : R × R diagonal matrix of λ values
- `θ` : b parameter for the GeneralizedInverseGaussian distribution
- `τ²`: overall variance parameter
- `V` : dimension of original symmetric adjacency matrices

# Returns
new value of D
"""
function update_D!(D_out::Diagonal{Q,Vector{Q}},γ::AbstractVector{T}, u::AbstractArray{U,2}, Λ::Diagonal{S,Vector{S}}, θ, τ², V) where {Q,S,T,U}
    q = floor(Int,V*(V-1)/2)
    uᵀΛu = transpose(u) * Λ * u
    uᵀΛu_upper = upper_triangle(uᵀΛu)
    a_ = (γ - uᵀΛu_upper).^2 / τ²
    D_out[:,:] = Diagonal(map(k -> sample_rgig(θ,a_[k]), 1:q))
end

"""
    update_θ(ζ, ι, V, D)

Sample the next θ value from the Gamma distribution with a = ζ + V(V-1)/2 and b = ι + ∑(s[k,l]/2)

# Arguments
- `ζ` : hyperparameter, used to construct `a` parameter
- `ι` : hyperparameter, used to construct `b` parameter
- `V` : dimension of original symmetric adjacency matrices
- `D` : diagonal matrix of s values

# Returns
new value of θ
"""
function update_θ(ζ, ι, V, D::Diagonal{T,Vector{T}}) where {T}
    a = ζ + (V*(V-1))/2
    b = ι + sum(diag(D))/2
    rand(Gamma(a,1/b))
end

"""
    update_u_ξ(u, γ, D, τ², Δ, M, Λ, V)

Sample the next u and ξ values

# Arguments
- `u` : the current latent variables u - replaced with the new values of u
- `γ` : vector of regression parameters
- `D` : diagonal matrix of s values
- `τ²`: overall variance parameter
- `Δ` : set to 1 draw from Beta(aΔ, bΔ)
- `M` : set to R × R matrix drawn from InverseWishart(V, I_R) (where I_R is the identity matrix with dimension R × R)
- `Λ` : R × R diagonal matrix of λ values
- `V` : dimension of original symmetric adjacency matrices

# Returns
The new value of ξ (the u passed in is updated)
"""
function update_u_ξ!(u::AbstractArray{T,2}, γ::AbstractVector{Q}, D::Diagonal{P,Vector{P}}, τ², Δ, M::AbstractArray{S,2}, Λ::Diagonal{O,Vector{O}}, V) where {O,P,Q,S,T}
    w_top = zeros(V)
    u_new = zeros(size(u)...)
    ξ = zeros(Int64,V)
    mus = zeros(V,5)
    cov = zeros(V,5,5)
    for k in 1:V
        U = transpose(u[:,Not(k)])*Λ
        s = create_upper_tri(diag(D),V)
        Γ = create_upper_tri(γ, V)
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
        Σ = inv(((transpose(U)*inv(H)*U)/τ²) + inv(M))
        m = Σ*(transpose(U)*inv(H)*γk)/τ²

        mvn_a = MultivariateNormal(zeros(size(H,1)),Symmetric(τ²*H))
        mvn_b_Σ = Symmetric(τ²*H + U*M*transpose(U))
        mvn_b = MultivariateNormal(zeros(size(H,1)),mvn_b_Σ)
        w_top = (1-Δ) * pdf(mvn_a,γk)
        w_bot = w_top + Δ*pdf(mvn_b,γk)
        w = w_top / w_bot

        mvn_f = MultivariateNormal(m,Symmetric(Σ))
        mus[k,:] = m
        cov[k,:,:] = Σ

        ξ[k] = update_ξ(w)

        # the paper says the first term is (1-w) but their code uses ξ. Again i think this makes more sense
        # that this term would essentially be an indicator rather than a weight
        u_new[:,k] = ξ[k].*rand(mvn_f)
    end
    u[:,:] = u_new
    return ξ
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
    update_Δ(aΔ, bΔ, ξ, V)

Sample the next Δ value from the Beta distribution with parameters a = aΔ + ∑ξ and b = bΔ + V - ∑ξ

# Arguments
- `aΔ`: hyperparameter used as part of the a parameter in the beta distribution used to sample Δ.
- `bΔ`: hyperparameter used as part of the b parameter in the beta distribution used to sample Δ.
- `ξ` : vector of ξ values, 0 or 1

# Returns
the new value of Δ
"""
function update_Δ(aΔ, bΔ, ξ::AbstractVector{T}) where {T}
    a = aΔ + sum(ξ)
    b = bΔ + sum(1 .- ξ)
    sample_Beta(a,b)
end


"""
    update_M(u,ν,V,ξ)

Sample the next M value from the InverseWishart distribution with df = V + # of nonzero columns in u and Ψ = I + ∑ uΛuᵀ

# Arguments
- `u` : R × V matrix of latent variables
- `ν` : hyperparameter, base df for IW distribution (to be added to by sum of ξs)
- `V` : dimension of original symmetric adjacency matrices
- `ξ` : vector of ξ values, 0 or 1 (sum is added to df for IW distribution)

# Returns
the new value of M
"""
function update_M(u::AbstractArray{T,2},ν,V,ξ::AbstractVector{U}) where {T,U}
    R = size(u,1)
    uΛu = zeros(R,R)
    num_nonzero = 0
    for i in 1:V
        uΛu = uΛu + u[:,i]*transpose(u[:,i])
        if ξ[i] ≉ 0
            num_nonzero = num_nonzero + 1
        end
    end
    Ψ = I(R) + uΛu
    df = ν + num_nonzero
    rand(InverseWishart(df,Ψ))
end

"""
    update_Λ(πᵥ, R, Λ, u, D, τ², γ)

Sample the next values of λ from [1,0,-1] with probabilities determined from a normal mixture

# Arguments
- `Λ_new` : return array, populated with the new values of λ (a diagonal array of 1,0,-1s )
- `πᵥ`: 3 column vectors of dimension R used to weight normal mixture for probability values
- `R` : the dimensionality of the latent variables u
- `Λ` : R × R diagonal matrix of λ values
- `u` : R × V matrix of latent variables
- `D` : diagonal matrix of s values
- `τ²`: overall variance parameter
- `γ` : vector of regression parameters

# Returns
new value of Λ
"""
function update_Λ!(Λ_new::Diagonal{T,Vector{T}}, πᵥ::AbstractArray{U,2}, R, Λ::Diagonal{T,Vector{T}}, u::Array{S,2}, D::Diagonal{P,Vector{P}}, τ², γ::AbstractVector{Q}) where {P,Q,S,T,U}
    λ_new = zeros(Int64,size(Λ,1))
    for r in 1:R
        Λ₋₁= deepcopy(Λ)
        Λ₋₁[r,r] = -1
        Λ₀ = deepcopy(Λ)
        Λ₀[r,r] = 0
        Λ₁ = deepcopy(Λ)
        Λ₁[r,r] = 1
        u_tr = transpose(u)
        W₋₁= upper_triangle(u_tr * Λ₋₁ * u)
        W₀ = upper_triangle(u_tr * Λ₀ * u)
        W₁ = upper_triangle(u_tr * Λ₁ * u)

        τ²D = τ²*D
        n₀ = prod(map(i -> pdf(Normal(W₀[i],sqrt(τ²D[i,i])),γ[i]),1:size(γ,1)))
        n₁ = prod(map(i -> pdf(Normal(W₁[i],sqrt(τ²D[i,i])),γ[i]),1:size(γ,1)))
        n₋₁ = prod(map(i -> pdf(Normal(W₋₁[i],sqrt(τ²D[i,i])),γ[i]),1:size(γ,1)))
        p_bot = πᵥ[r,1] * n₀ + πᵥ[r,2] * n₁ + πᵥ[r,3] * n₋₁
        p1 = πᵥ[r,1] * n₀ / p_bot
        p2 = πᵥ[r,2] * n₁ / p_bot
        p3 = πᵥ[r,3] * n₋₁ / p_bot
        λ_new[r] = sample([0,1,-1],StatsBase.weights([p1,p2,p3]))
    end
    Λ_new[:,:] = Diagonal(λ_new)
end

"""
    update_π(λ,η,R)

Sample the new values of πᵥ from the Dirichlet distribution with parameters [1 + #{r: λᵣ= 1}, #{r: λᵣ = 0} + r^η, 1 + #{r: λᵣ = -1 }]

# Arguments
- `π_out`: array of size R × 3 of ouput variables
- `Λ` : R × R diagonal matrix of λ values
- `η` : hyperparameter used for sampling the 0 value (r^η)
- `R` : dimension of u vectors

# Returns
new value of πᵥ
"""
function update_π!(π_out::AbstractArray{U}, Λ::Diagonal{T,Vector{T}},η,R) where {T,U}
    λ = diag(Λ)
    for r in 1:R
        π_ret = Vector{AbstractFloat}(undef,3)
        sample_π_dirichlet!(π_ret,r,η,λ)
        π_out[r,:] = π_ret 
    end
end
#endregion

"""
    GibbsSample!(result,X, y, θ, D, πᵥ, Λ, Δ, ξ, M, u, μ, γ, V, η, ζ, ι, R, aΔ, bΔ, ν)

Take one GibbsSample

# Arguments
- `result`: Array of outputs (see returns section for more detail)
- `X` : matrix of unweighted symmetric adjacency matrices to be used as predictors. each row should be the upper triangle of the adjacency matrix associated with one sample.
- `y` : vector of response variables
- `θ` : θ parameter, drawn from the Gamma Distribution
- `D` : Diagonal matrix of s values
- `πᵥ`: 3 column vectors of dimension R used to weight normal mixture for probability values
- `Λ` : R × R diagonal matrix of λ values
- `Δ` : Δ parameter, drawn from Beta distribution
- `M` : R × R matrix, drawn from InverseWishart
- `u` : R × V matrix of latent variables
- `γ` : (V*V-1)-vector of regression parameters, influence of each edge
- `V` : number of nodes in adjacency matrices
- `η` : hyperparameter used for sampling the 0 value of the πᵥ parameter
- `ζ` : hyperparameter used for sampling θ
- `ι` : hyperparameter used for sampling θ
- `R` : hyperparameter - depth of u vectors (and others)
- `aΔ`: hyperparameter used for sampling Δ
- `bΔ`: hyperparameter used for sampling Δ
- `ν` : hyperparameter used for sampling M

# Returns
An array of the new values, [τ²_n, u_n, ξ_n, γ_n, D_n, θ_n, Δ_n, M_n, μ_n, Λ_n, πᵥ_n]
"""
#function GibbsSample!(result::AbstractVector{T},X::AbstractArray{U,2}, y::AbstractVector{S}, θ, D::Diagonal{L,Vector{L}}, πᵥ::AbstractArray{Q,2}, Λ::Diagonal{K,Vector{K}}, Δ, M::AbstractArray{P,2}, u::AbstractArray{O,2}, μ, γ::AbstractVector, V, η, ζ, ι, R, aΔ, bΔ, ν) where {K,L,O,P,Q,S,T,U}
function GibbsSample!(result::AbstractVector{T},X::AbstractArray{U,2}, y::AbstractVector{S}, V, η, ζ, ι, R, aΔ, bΔ, ν) where {S,T,U}
    u = result[2]
    γ, D, θ, Δ, M, μ, Λ, πᵥ = result[4:11]
    n = size(X,1)

    τ²_n = update_τ²(X, y, μ, γ, Λ, u, D, V)

    # I could just pass u into update_u_ξ!, but I think that would be confusing to read 
    # since all of the other new variables are _n, so I'm going to be slightly inefficient and copy u to
    # the new u_n variable
    u_n = u
    ξ_n = update_u_ξ!(u_n, γ, D, τ²_n, Δ, M, Λ, V)

    γ_n = Vector{AbstractFloat}(undef,size(γ,1))
    update_γ!(γ_n, X, y, D, Λ, u_n, μ, τ²_n, n)

    q = floor(Int,V*(V-1)/2)
    D_n = Diagonal(Vector{AbstractFloat}(undef,q))
    update_D!(D_n, γ_n, u_n, Λ, θ, τ²_n, V)
    
    θ_n = update_θ(ζ, ι, V, D_n)
    Δ_n = update_Δ(aΔ, bΔ, ξ_n)
    M_n = update_M(u_n, ν, V,ξ_n)
    μ_n = update_μ(X, y, γ_n, τ²_n, n)

    Λ_n = Diagonal(Vector{Int}(undef,R))
    update_Λ!(Λ_n, πᵥ, R, Λ, u_n, D_n, τ²_n, γ_n)

    πᵥ_n = Array{AbstractFloat,2}(undef,(R,3))
    update_π!(πᵥ_n, Λ_n, η,R)
    result[1:11] = [τ²_n, u_n, ξ_n, γ_n, D_n, θ_n, Δ_n, M_n, μ_n, Λ_n, πᵥ_n]
    nothing
end


function GenerateSamples!(X::AbstractArray{T,2}, y::AbstractVector{U}, R; η=1.01,ζ=1.0,ι=1.0,aΔ=1.0,bΔ=1.0, ν=12, nburn=30000, nsamples=20000, V=0, x_transform=true) where {T,U}
    if V == 0 && !x_transform
        ArgumentError("If x_transform is false a valid V value must be given")
    end
    
    X, θ, D, πᵥ, Λ, Δ, ξ, M, u, μ, τ², γ = init_vars!(X, η, ζ, ι, R, aΔ, bΔ, ν, V, x_transform)

    total = nburn + nsamples
    p = Progress(total,1)
    result = [τ²,u,ξ,γ,D,θ,Δ,M,μ,Λ,πᵥ]
    γₐ = SVector{nsamples}
    γₐ = [γ]
    ξₐ = [ξ]
    uₐ = [u]
    # burn-in
    for i in 1:nburn
        #τ²[1], u[1], ξ[1], γ[1], D[1], θ[1], Δ[1], M[1], μ[1], Λ[1], πᵥ[1] = GibbsSample(...
        GibbsSample!(result, X, y, V, η, ζ, ι, R, aΔ, bΔ,ν)
        next!(p)
    end
    for i in 1:nsamples
        #τ²[1], u[1], ξ[1], γ[1], D[1], θ[1], Δ[1], M[1], μ[1], Λ[1], πᵥ[1] = GibbsSample(...
        GibbsSample!(result, X, y, V, η, ζ, ι, R, aΔ, bΔ,ν)
        push!(uₐ,result[2])
        push!(ξₐ,result[3])
        push!(γₐ,result[4])
        next!(p)
    end
    return BNRPosteriors(γₐ[2:size(γₐ,1)],ξₐ[2:size(γₐ,1)],uₐ[2:size(γₐ,1)])
end
