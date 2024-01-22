#region output
"""
Results struct
    state: Table with arguments (n=num generations):
    τ² = Array{Float64,3} dimension (n,1,1) variance of error term, 
    u = Array{Float64,3} dimension (n,R,V) R-dim latent variables for each node,
    ξ = Array{Float64,3} dimension (n,V,1) binary vector to denote if a node is influential, 
    γ = Array{Float64,3} dimension (n,q,1) regression coefficients for edge effects,
    S = Array{Float64,3} dimension (n,q,1) scale parameters for the variance of γ, 
    θ = Array{Float64,3} dimension (n,1,1) exponential parameter for the scale S,
    Δ = Array{Float64,3} dimension (n,1,1) Bernoulli parameter for ξ, 
    M = Array{Float64,3} dimension (n,R,R) covariance matrix of latent variables u,
    μ = Array{Float64,3} dimension (n,1,1) overall mean, 
    λ = Array{Float64,3} dimension (n,R,1) {0,1,-1} variables that govern which entries in the latent variables u are informative,
    πᵥ= Array{Float64,3} dimension (n,R,3) Dirichlet prior probabilities of λ being 0,1,-1.

    rhatξ: Table with one column, rows=number of nodes
    ξ = Array{Float64,1} dimension (num nodes,1)

    rhatγ: Table with one column, rows=number of nodes
    γ = Array{Float64,1} dimension (num edges,1)
"""
struct Results
    state::Table
    rhatξ::Table
    rhatγ::Table
    burn_in::Int
    sampled::Int
end

"""
BNRSummary struct
    edge_coef: DataFrame with edge coefficient point estimates and endpoints of credible intervals
        dimension (q,5)
    prob_nodes: DataFrame with probabilities of being influential for each node
        dimension (V,1)
    ci_level: Int - level used for the credible intervals (default=95)
"""
struct BNRSummary
    edge_coef::DataFrame
    prob_nodes::DataFrame
    ci_level::Int
end

"""
show(io::IO,b::BNRSummary)

Show the summary output, node probabilities of influence and edge coefficient estimates

# Arguments
- `io::IO``: The I/O stream to which the summary will be printed.
- `b::BNRSummary`: The summary object to print
"""
function show(io::IO,b::BNRSummary)
    print(io,"\n",
        "Edge Coefficient Estimates ($(b.ci_level)% credible intervals)\n",
        b.edge_coef ,"\n",
        "Node Probabilities\n",
        b.prob_nodes
    )
end
Base.show(io::IO,b::BNRSummary) = show(io,b)

"""
show(io::IO,b::Results)

Show the summary output as a result of fitting the BNR model, node probabilities of influence and edge coefficient estimates.

# Arguments
- `io::IO``: The I/O stream to which the summary will be printed.
- `r::Results`: The results object to summarize and show,
"""
function show(io::IO,r::Results)
    show(io,Summary(r))
end
Base.show(io::IO,r::Results) = show(io,r)

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
    initialize_variables!(state::Table, η, R, ν, rng, V=0)

    Initialize all variables using prior distributions. 
    All initializations done in place on the state argument.

    # Arguments
    - `state` : a row-table structure containing all past states, the current state, and space for the future states
    - `η` : hyperparameter used to sample from the Dirichlet distribution (r^η)
    - `R` : the dimensionality of the latent variables u, a hyperparameter
    - `rng` : random number generator to be used for sampling
    - `ν` : hyperparameter used as the degrees of freedom parameter in the InverseWishart distribution used to sample M.
    - `V`: Value of V, the number of nodes in the original X matrix.

    # Returns
    nothing
"""
function initialize_variables!(state::Table, η, R, ν, rng , V=0)
    # η must be greater than 1, if it's not set it to its default value of 1.01
    if (η <= 1)
        η = 1.01
        println("η value invalid, reset to default of 1.01")
    end
    q = floor(Int,V*(V+1)/2)

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
    
    utlamu = transpose(state.u[1,:,:]) * Diagonal(state.λ[1,:]) * state.u[1,:,:]

    low_tri = lower_triangle(utlamu)

    state.γ[1,:] = rand(rng,MultivariateNormal(reshape(low_tri,(q,)), state.τ²[1]*Diagonal(state.S[1,:,1])))
end

"""
    setup_X!(X_new::AbstractArray{U}, X::AbstractArray{T})

Set up the X matrix so it's in the expanded model matrix form (one row per sample)

# Arguments
- `X_new` : 2-dimensional n × V(V+1)/2 matrix - will hold reshaped X
- `X` : vector of unweighted symmetric adjacency matrices to be used as predictors. each element of the array should be 1 matrix
- `x_transform`: boolean, set to false if X has been pre-transformed into one row per sample. True by default.

# Returns
nothing
"""
function setup_X!(X_new::AbstractArray{U}, X::AbstractArray{T}, x_transform::Bool=true) where {T,U}
    if x_transform
        for i in axes(X,1)
            X_new[i,:] = lower_triangle(X[i])
        end
    else
        X_new[:,:] = X
    end
end

#region update variables

"""
    update_τ²!(state::Table, i, X::AbstractArray{T,2}, y::AbstractVector{U}, V, rng)

Sample the next τ² value from the InverseGaussian distribution with mean n/2 + V(V+1)/4 and variance ((y - μ1 - Xγ)ᵀ(y - μ1 - Xγ) + (γ - W)ᵀD⁻¹(γ - W)

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

    state.τ²[i] = rand(rng,InverseGamma((n/2) + (V*(V+1)/4), σₜ²))
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

        Σ⁻¹ = nothing
        try
            Σ⁻¹ = ((Uᵀ)*(H\U))/state.τ²[i] + inv(state.M[i-1,:,:])
        catch
            Σ⁻¹ = ((Uᵀ)*(H\U))/state.τ²[i] + inv(Symmetric(state.M[i-1,:,:]))
        end

        C = 0
        try
            C = cholesky(Hermitian(Σ⁻¹,:L))
        catch 
            if !isposdef(Hermitian(Σ⁻¹,:L))
                Σ⁻¹ = Σ⁻¹ + I(size(Σ⁻¹,1))*1e-5
            end
            try 
                C = cholesky(Hermitian(Σ⁻¹,:L))
            catch 
                println(stderr,"eigvals(Hermitian(Σ⁻¹,:L))",eigvals(Hermitian(Σ⁻¹,:L)))
                println(stderr,"eigvals(Σ⁻¹)",eigvals(Σ⁻¹))
                if !isposdef(Hermitian(Σ⁻¹,:L))
                    Σ⁻¹ = Σ⁻¹ + I(size(Σ⁻¹,1))*4e-5
                end
                try 
                    C = cholesky(Hermitian(Σ⁻¹,:L))
                catch
                    println(stderr,"eigvals(Hermitian(Σ⁻¹,:L))",eigvals(Hermitian(Σ⁻¹,:L)))
                println(stderr,"eigvals(Σ⁻¹)",eigvals(Σ⁻¹))
                    throw()
                end
            end
            println(stderr,"eigvals(Hermitian(Σ⁻¹,:L))",eigvals(Hermitian(Σ⁻¹,:L)))
            println(stderr,"eigvals(Σ⁻¹)",eigvals(Σ⁻¹))
        end

        #w_top = log(1-state.Δ[i-1]) + logpdf(MultivariateNormal(zeros(size(H,1)),Symmetric(state.τ²[i]*H)),γk)
        #w_bot = log(state.Δ[i-1]) + logpdf( MultivariateNormal(zeros(size(H,1)), Symmetric(state.τ²[i] * H + U * state.M[i-1,:,:] * Uᵀ)),γk)
        #w = exp(w_top - log(exp(w_bot) + exp(w_top)))

        w_top = (1-state.Δ[i-1]) * pdf(MultivariateNormal(zeros(size(H,1)),Symmetric(state.τ²[i]*H)),γk)
        w_bot = state.Δ[i-1] * pdf( MultivariateNormal(zeros(size(H,1)), Symmetric(state.τ²[i] * H + U * state.M[i-1,:,:] * Uᵀ)),γk)
        w = w_top / (w_bot + w_top)

        if(isnan(w))
            println(stderr,"w_top=")
            println(stderr,w_top)
            println(stderr,"")
            println(stderr,"w_bot=")
            println(stderr,w_top)
            println(stderr,"")
        end


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
    if w <= 0
        return 1
    elseif w >= 1
        return 0
    end
    out = -1
    try 
        out = Int64(rand(rng,Bernoulli(1 - w)))
    catch
        println(stderr, "w = ", w)
    end
    if out==-1
        println(stderr,"bad out")
        out = rand(rng,Bernoulli(0.5))
    end
    return out
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
    state.S[i,:] = map(k -> sample_rgig(state.θ[i-1],a_[k], rng), axes(state.S,2))
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
    state.θ[i] = rand(rng,Gamma(ζ + (V*(V+1))/2,2/(2*ι + sum_kbn(state.S[i,:]))))
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

    Ψ0 = 0
    Ψ = 0
    try
        Ψ = cholesky(Matrix(I(R) + uuᵀ))
    catch 
        
        try
            Ψ = cholesky(Hermitian(Matrix(I(R) + uuᵀ)))
        catch
            Ψ0 = Hermitian(Matrix(I(R) + uuᵀ))
            if !isposdef(Ψ0)
                Ψ0 = Ψ0 + I(size(Ψ0,1))*1e-5
            end
            @show Ψ0
            Ψ = cholesky(Ψ0)
        end
    end

    state.M[i,:,:] = rand(rng,InverseWishart(ν + num_nonzero,Ψ))
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
         ν=10, nburn=30000, nsamp=20000, x_transform=true, suppress_timer=false, num_chains=2, seed=nothing, purge_burn=nothing) where {T,U}

Fit the Bayesian Network Regression model, generating `nsamp` Gibbs samples after `nburn` burn-in are discarded. If the Gelman-Rubin PSRF cutoff
is not achieved, nsamp samples will be added to the burn-in until either convergence is achieved or `maxburn` burn-in are generated

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
- `minburn`: integer, default=30000, minimum number of burn-in samples to generate and discard
- `nsamp`: integer, default=20000, number of Gibbs samples to generate after burn-in
- `maxburn`: integer, default minburn+nsamp, maximum number of burn-in samples to generate and discard
- `psrf_cutoff`: float, defalut=1.2, value at which convergence is determined to have been achieved
- `x_transform`: boolean, default=true, set to false if X has been pre-transformed into one row per sample. Otherwise the X will be transformed automatically.
- `suppress_timer`: boolean, default=false, set to true to suppress "progress meter" output
- `num_chains`: integer, default=2, number of separate sampling chains to run (for checking convergence)
- `seed`: integer, default=nothing, random seed used for repeatability
- `purge_burn`: integer, default=nothing, if set must be less than the number of burn-in samples (and ideally burn-in is a multiple of this value). After how many burn-in samples to delete previous burn-in samples.
- `filename`: logfile with the parameters used for the fit, default="parameters.log". The file will be overwritten if a new name is not specified.

# Returns

`Results` object with the state table from the first chain and PSRF r-hat values for  γ and ξ 

"""
function Fit0!(X::AbstractArray{T}, y::AbstractVector{U}, R; η=1.01,V=30,ζ=1.0,ι=1.0,aΔ=1.0,bΔ=1.0, 
    ν=10, minburn=30000, nsamp=20000, maxburn=50000, psrf_cutoff=1.01, x_transform=true, suppress_timer=false, 
    num_chains=2, seed=nothing, purge_burn=nothing, filename="parameters.log") where {T,U}
    ## Saving parameters to file:
    logfile = open(filename,"w")
    write(logfile, "BayesianNetworkRegression.jl Fit! function\n")
    write(logfile, Dates.format(Dates.now(), "yyyy-mm-dd H:M:S.s") * "\n")
    write(logfile, citation(returnstring=true))
    write(logfile, "\n\nParameters:\n")
    str = "R=$R, η=$η, ζ=$ζ, ι=$ι, aΔ=$aΔ, bΔ=$bΔ, ν=$ν, minburn=$minburn, nsamp=$nsamp, maxburn=$maxburn, psrf_cutoff=$psrf_cutoff, \n"
    str *= "x_transform=$x_transform, suppress_timer=$suppress_timer, num_chains=$num_chains, purge_burn=$purge_burn \n"

    ## setting a seed to print to logfile
    seed = isnothing(seed) ? sample(1:55555,1)[1] : seed
    str *= "seed=$seed"
    write(logfile, str)
    close(logfile)

    generate_samples!(X, y, R; η=η,ζ=ζ,ι=ι,aΔ=aΔ,bΔ=bΔ,ν=ν,minburn=minburn,nsamp=nsamp,maxburn=maxburn,psrf_cutoff=psrf_cutoff,
    x_transform=x_transform,suppress_timer=suppress_timer,num_chains=num_chains,seed=seed,purge_burn=purge_burn)
end


function Fit!(X::AbstractArray{T}, y::AbstractVector{U}, R; η=1.01,V=30,ζ=1.0,ι=1.0,aΔ=1.0,bΔ=1.0, 
    ν=10, mingen=30000, maxgen=50000, psrf_cutoff=1.01, x_transform=true, suppress_timer=false, 
    num_chains=2, seed=nothing, purge_burn=nothing, filename="parameters.log") where {T,U}
    ## Saving parameters to file:
    logfile = open(filename,"w")
    write(logfile, "BayesianNetworkRegression.jl Fit! function\n")
    write(logfile, Dates.format(Dates.now(), "yyyy-mm-dd H:M:S.s") * "\n")
    write(logfile, citation(returnstring=true))
    write(logfile, "\n\nParameters:\n")
    str = "R=$R, η=$η, ζ=$ζ, ι=$ι, aΔ=$aΔ, bΔ=$bΔ, ν=$ν, mingen=$mingen, maxgen=$maxgen, psrf_cutoff=$psrf_cutoff, \n"
    str *= "x_transform=$x_transform, suppress_timer=$suppress_timer, num_chains=$num_chains, purge_burn=$purge_burn \n"

    ## setting a seed to print to logfile
    seed = isnothing(seed) ? sample(1:55555,1)[1] : seed
    str *= "seed=$seed"
    write(logfile, str)
    close(logfile)

    generate_samples_dbl!(X, y, R; η=η,ζ=ζ,ι=ι,aΔ=aΔ,bΔ=bΔ,ν=ν,mingen=mingen,maxgen=maxgen,psrf_cutoff=psrf_cutoff,
    x_transform=x_transform,suppress_timer=suppress_timer,num_chains=num_chains,seed=seed,purge_burn=purge_burn)
end


"""
    return_psrf_VOI(states,num_chains,nburn,nsamp,V,q)

organize and return ALL (burn-in and post-burn-in) samples for the first state, calculate and return PSRF r-hat for ξ and γ

# Arguments 
- `states`: a vector of states, each of which is a row-table structure containing all past states (or only post-burn-in)
- `num_chains`: the number of chains run (length of states vector)
- `nburn`: the number of burn-in samples. If the states in `states` only contain post-burn, set this to 0
- `nsamp`: the number of post-burn-in samples to use
- `V`: the dimension of the original adjacency matrix, used to size the return table
- `q`: the dimension of the model matrix, used to size the return table

# Returns
A `Results` object with the state table from the first chain and PSRF r-hat tables for ξ and γ 

"""
function return_psrf_VOI(states,num_chains,nburn,nsamp,R,V,q)
    all_ξs = Array{Float64,3}(undef,(nsamp,V,num_chains))
    all_γs = Array{Float64,3}(undef,(nsamp,q,num_chains))

    total = nsamp + nburn

    for c=1:num_chains
        all_ξs[:,:,c] = states[c].ξ[nburn+1:total,:,1]
        all_γs[:,:,c] = states[c].γ[nburn+1:total,:,1]
    end

    psrfξ = Table(ξ = Vector{Float64}(undef,V))
    psrfγ = Table(γ = Vector{Float64}(undef,q))

    psrfγ.γ[1:q] = rhat(all_γs)
    psrfξ.ξ[1:V] = rhat(all_ξs)

    #states_out = combine_states!(states,nburn,nsamp,num_chains,R,V,q)
    return Results(states[1],psrfξ,psrfγ,nburn,nsamp)

    #return Results(states_out,psrfξ,psrfγ,nburn,nsamp*num_chains)
end

function combine_states!(states,nburn,nsamp,num_chains,R,V,q)
    tot_save = nsamp * num_chains + nburn
    total = nburn+nsamp
    
    states_out = Table(τ² = Array{Float64,3}(undef,(tot_save,1,1)), u = Array{Float64,3}(undef,(tot_save,R,V)),
    ξ = Array{Float64,3}(undef,(tot_save,V,1)), γ = Array{Float64,3}(undef,(tot_save,q,1)),
    S = Array{Float64,3}(undef,(tot_save,q,1)), θ = Array{Float64,3}(undef,(tot_save,1,1)),
    Δ = Array{Float64,3}(undef,(tot_save,1,1)), M = Array{Float64,3}(undef,(tot_save,R,R)),
    μ = Array{Float64,3}(undef,(tot_save,1,1)), λ = Array{Float64,3}(undef,(tot_save,R,1)),
    πᵥ= Array{Float64,3}(undef,(tot_save,R,3)), Σ⁻¹= Array{Float64,3}(undef,(tot_save,R,R)),
    invC = Array{Float64,3}(undef,(tot_save,R,R)), μₜ = Array{Float64,3}(undef,(tot_save,R,1)))

    copy_table!(states_out,states[1],1:nburn,1:nburn)

    for c in 1:num_chains
        copy_table!(states_out,states[c],((c-1)*nsamp+1+nburn):(c*nsamp+nburn),(nburn+1):total)
    end
    return states_out
end

"""
    initialize_and_run!(X::AbstractArray{T},y::AbstractVector{U},c,total,V,R,η,ζ,ι,aΔ,bΔ,ν,rng,x_transform,suppress_timer,prog_freq,purge_burn,nsamp,channel) where {T,U}

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
- `nsamp`: integer, the number of post burn-in samples to retain. only necessary to provide when purge_burn is not nothing
- `channel`: channel between worker and manager, used to update the progress meter when running parallel chains

# Returns

The complete `state` table with all samples of all variables.

"""
function initialize_and_run!(X::AbstractArray{T},y::AbstractVector{U},c,total,V,R,η,ζ,ι,aΔ,bΔ, 
                             ν,rng,prog_freq,purge_burn,nsamp,channel) where {T,U}

    q = Int64(V*(V+1)/2)

    tot_save = total
    if (!isnothing(purge_burn))
        tot_save = nsamp + purge_burn;
    end

    nburn = total - nsamp


    state = Table(τ² = Array{Float64,3}(undef,(tot_save,1,1)), u = Array{Float64,3}(undef,(tot_save,R,V)),
            ξ = Array{Float64,3}(undef,(tot_save,V,1)), γ = Array{Float64,3}(undef,(tot_save,q,1)),
            S = Array{Float64,3}(undef,(tot_save,q,1)), θ = Array{Float64,3}(undef,(tot_save,1,1)),
            Δ = Array{Float64,3}(undef,(tot_save,1,1)), M = Array{Float64,3}(undef,(tot_save,R,R)),
            μ = Array{Float64,3}(undef,(tot_save,1,1)), λ = Array{Float64,3}(undef,(tot_save,R,1)),
            πᵥ= Array{Float64,3}(undef,(tot_save,R,3)), Σ⁻¹= Array{Float64,3}(undef,(tot_save,R,R)),
            invC = Array{Float64,3}(undef,(tot_save,R,R)), μₜ = Array{Float64,3}(undef,(tot_save,R,1)))


    initialize_variables!(state, η, R, ν, rng, V)
    return run!(X,y,state,c,2,nburn,total,V,R,η,ζ,ι,aΔ,bΔ,ν,rng,prog_freq,purge_burn,channel)
end


function run!(X::AbstractArray{T},y::AbstractVector{U},state::Table,c,first_index,nburn,total,V,R,η,ζ,ι,aΔ,bΔ, 
    ν,rng,prog_freq,purge_burn,channel) where {T,U}
    j = first_index
    for i in first_index:total
        gibbs_sample!(state, j, X, y, V, η, ζ, ι, R, aΔ, bΔ, ν, rng)
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
        ν=10, nburn=30000, nsamp=20000, x_transform=true, suppress_timer=false, num_chains=2, seed=nothing, purge_burn=nothing) where {T,U}

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
- `minburn`: integer, default=30000, minimum number of burn-in samples to generate and discard
- `nsamp`: integer, default=20000, number of Gibbs samples to generate after burn-in
- `maxburn`: integer, default minburn+nsamp, maximum number of burn-in samples to generate and discard
- `psrf_cutoff`: float, defalut=1.2, value at which convergence is determined to have been achieved
- `x_transform`: boolean, default=true, set to false if X has been pre-transformed into one row per sample. Otherwise the X will be transformed automatically.
- `suppress_timer`: boolean, default=false, set to true to suppress "progress meter" output
- `num_chains`: integer, default=2, number of separate sampling chains to run (for checking convergence)
- `seed`: integer, default=nothing, random seed used for repeatability
- `purge_burn`: integer, default=nothing, if set must be less than the number of burn-in samples (and ideally burn-in is a multiple of this value). After how many burn-in samples to delete previous burn-in samples.

# Returns

`Results` object with the state table from the first chain and PSRF r-hat values for  γ and ξ 
"""
function generate_samples!(X::AbstractArray{T}, y::AbstractVector{U}, R; η=1.01,ζ=1.0,ι=1.0,aΔ=1.0,bΔ=1.0,
    ν=10, minburn=30000, nsamp=20000, maxburn=50000, psrf_cutoff=1.2, x_transform=true, suppress_timer=false, 
    num_chains=2, seed=nothing,purge_burn=nothing) where {T,U}

    if ν < R
        ArgumentError("ν value ($ν) must be greater than R value ($R)")
    elseif ν == R
        println("Warning: ν==R may give poor accuracy. Consider increasing ν")
    end

    if x_transform
        V = size(X[1],1)
        q = floor(Int,V*(V+1)/2)
    else
        q = size(X,2)
        V = Int64((-1 + sqrt( 1 + 8 * q))/2)
    end

    n = size(X,1)

    X_new = Matrix{eltype(T)}(undef, n, q)
    setup_X!(X_new,X,x_transform)

    nburn = minburn

    states = Vector{Table}(undef,num_chains)
    total = nburn + nsamp

    prog_freq = 1000
    if prog_freq >= nburn
        prog_freq = 10
    end

    rngs = [ (isnothing(seed) ? Xoshiro() : Xoshiro(seed+c)) for c = 1:num_chains ]

    if !isnothing(purge_burn) && (purge_burn < nburn) && purge_burn != 0
        if nburn % purge_burn != 0 
            purge_burn = purge_burn - (nburn % purge_burn)
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
        t = @async begin
            states = pmap(1:num_chains) do c
                return initialize_and_run!(X_new,y,c,total,V,R,η,ζ,ι,aΔ,bΔ,ν,rngs[c],prog_freq,purge_burn,nsamp,channel)
            end
            put!(channel, false)
        end
        fetch(t)
    end


    tot_generated = nburn+nsamp

    tmp_res = return_psrf_VOI(states,num_chains,!isnothing(purge_burn) ? purge_burn : nburn,nsamp,R,V,q)

    println(stderr, tot_generated, " samples generated. Max PSRF XI: ", round(max(tmp_res.rhatξ.ξ...),digits=2), ". Max PSRF Gamma: ", round(max(tmp_res.rhatγ.γ...),digits=2))


    while ( (max(tmp_res.rhatξ.ξ...) > psrf_cutoff || max(tmp_res.rhatγ.γ...) > psrf_cutoff) && tot_generated < (maxburn+nsamp) )
        num2move = 0

        # we want to generate nburn more samples
        if !isnothing(purge_burn)
            if nsamp + purge_burn <= nburn
                num2move = 1
            else
                num2move = nsamp + purge_burn - nburn
            end
        else 
            num2move = total - nburn
        end

        tot_sze = size(states[1],1)#!isnothing(purge_burn) ? purge_burn + nsamp : nburn + nsamp

        #num2move = num2move > total ? 0 : total - num2move
        println(stderr,"num2move: ", num2move, " nburn: ", nburn, " nsamp: ", nsamp, " purge_burn: ", purge_burn)

        p = Progress(Int(floor((tot_generated + nburn-1)/prog_freq));dt=1,showspeed=true, enabled = !suppress_timer,start=Int(floor(tot_generated/prog_freq)))
        channel = RemoteChannel(()->Channel{Bool}(), 1)

        @sync begin 
            @async while take!(channel)
                next!(p)
            end
            t = @async begin
                states = pmap(1:num_chains) do c
                    ## this is super inefficient, should we do this differently?
                    for i in 1:num2move
                        copy_table!(states[c],i,tot_sze - num2move+i)
                    end
                    #      run!(X    ,y,state    ,c,first_index,nburn,
                    #           total,                                V,R,η,ζ,ι,aΔ,bΔ, ν,rng,prog_freq,
                    #           purge_burn,channel)
                    return run!(X_new,y,states[c],c,num2move+1,nburn > nsamp ? nburn - nsamp + num2move : 0,
                                num2move > 1 ? num2move+nburn : nburn,V,R,η,ζ,ι,aΔ,bΔ,ν,rngs[c],prog_freq,
                                purge_burn,channel)
                end
                put!(channel, false)
            end

            fetch(t)
        end
        A = num2move > 1 ? num2move+nburn : nburn
        B = num2move
        tot_generated = tot_generated + A - B
        tmp_res = return_psrf_VOI(states,num_chains,!isnothing(purge_burn) ? purge_burn : nburn,nsamp,R,V,q)
        println(stderr, tot_generated, " samples generated. Max PSRF XI: ", round(max(tmp_res.rhatξ.ξ...),digits=3), ". Max PSRF Gamma: ", round(max(tmp_res.rhatγ.γ...),digits=3))
        stt = isnothing(purge_burn) ? nburn : purge_burn
        @show mean(states[1].ξ[stt+1:(stt+nsamp),:,1],dims=1)[1,:,1]
    end
    stt = isnothing(purge_burn) ? nburn : purge_burn
    println("R = ",R, " nu=",ν, " nburn= ",nburn, " nsamp = ", nsamp)
    println(tot_generated, " samples generated. Max PSRF XI: ", round(max(tmp_res.rhatξ.ξ...),digits=3), ". Max PSRF Gamma: ", round(max(tmp_res.rhatγ.γ...),digits=3))
    println()
    @show mean(states[1].ξ[stt+1:(stt+nsamp),:,1],dims=1)[1,:,1]
    return_psrf_VOI(states,num_chains,!isnothing(purge_burn) ? purge_burn : nburn,nsamp,R,V,q)
end

"""
    generate_samples_dbl!(X::AbstractArray{T}, y::AbstractVector{U}, R; η=1.01,ζ=1.0,ι=1.0,aΔ=1.0,bΔ=1.0, V=0,
        ν=10, min_gen=10000, max_gen=100000, x_transform=true, suppress_timer=false, num_chains=2, seed=nothing, purge_burn=nothing) where {T,U}

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
- `mingen`: integer, default=10000, minimum number of samples to generate - 1/2 will be discarded as burn-in
- `maxgen`: integer, default=100000, maximum number of samples to generate - 1/2 will be discarded as burn-in
- `psrf_cutoff`: float, defalut=1.01, value at which convergence is determined to have been achieved
- `x_transform`: boolean, default=true, set to false if X has been pre-transformed into one row per sample. Otherwise the X will be transformed automatically.
- `suppress_timer`: boolean, default=false, set to true to suppress "progress meter" output
- `num_chains`: integer, default=2, number of separate sampling chains to run (for checking convergence)
- `seed`: integer, default=nothing, random seed used for repeatability
- `purge_burn`: integer, default=nothing, if set must be less than the number of burn-in samples (and ideally burn-in is a multiple of this value). After how many burn-in samples to delete previous burn-in samples. (Ignored)

# Returns

`Results` object with the state table from the first chain and PSRF r-hat values for  γ and ξ 
"""
function generate_samples_dbl!(X::AbstractArray{T}, y::AbstractVector{U}, R; η=1.01,ζ=1.0,ι=1.0,aΔ=1.0,bΔ=1.0,
    ν=10, mingen=10000, maxgen=100000, psrf_cutoff=1.01, x_transform=true, suppress_timer=false, 
    num_chains=2, seed=nothing,purge_burn=nothing) where {T,U}

    if ν < R
        ArgumentError("ν value ($ν) must be greater than R value ($R)")
    elseif ν == R
        println("Warning: ν==R may give poor accuracy. Consider increasing ν")
    end

    nburn = convert(Int64,round(mingen/2))
    nsamp = mingen-nburn

    if x_transform
        V = size(X[1],1)
        q = floor(Int,V*(V+1)/2)
    else
        q = size(X,2)
        V = Int64((-1 + sqrt( 1 + 8 * q))/2)
    end

    n = size(X,1)

    X_new = Matrix{eltype(T)}(undef, n, q)
    setup_X!(X_new,X,x_transform)

    states = Vector{Table}(undef,num_chains)
    total = nburn + nsamp

    prog_freq = 1000
    if prog_freq >= nburn
        prog_freq = 10
    end
    rngs = [ (isnothing(seed) ? Xoshiro() : Xoshiro(seed+c)) for c = 1:num_chains ]

    if !isnothing(purge_burn) && (purge_burn < nburn) && purge_burn != 0
        if nburn % purge_burn != 0 
            purge_burn = purge_burn - (nburn % purge_burn)
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
        t = @async begin
            states = pmap(1:num_chains) do c
                return initialize_and_run!(X_new,y,c,total,V,R,η,ζ,ι,aΔ,bΔ,ν,rngs[c],prog_freq,purge_burn,nsamp,channel)
            end
            put!(channel, false)
        end
        fetch(t)
    end


    tot_generated = nburn+nsamp
    tot_samples = nsamp

    tmp_res = return_psrf_VOI(states,num_chains,!isnothing(purge_burn) ? purge_burn : nburn,nsamp,R,V,q)

    println(stderr, tot_generated, " samples generated. Max PSRF XI: ", round(max(tmp_res.rhatξ.ξ...),digits=3), ". Max PSRF Gamma: ", round(max(tmp_res.rhatγ.γ...),digits=3))


    while ( (max(tmp_res.rhatξ.ξ...) > psrf_cutoff || max(tmp_res.rhatγ.γ...) > psrf_cutoff || isnan(max(tmp_res.rhatξ.ξ...)) || isnan(max(tmp_res.rhatγ.γ...))) && tot_generated < maxgen )

        if isnan(max(tmp_res.rhatγ.γ...))
            println(stderr,"-----------------------------------------------------")
            println(stderr, "rhat: ", tmp_res.rhatγ.γ)
            println(stderr, mean(states[1].ξ[nburn+1:total,:,1],dims=1)[1,:,1])
            println(stderr,"-----------------------------------------------------")
        end
        if isnan(max(tmp_res.rhatξ.ξ...))
            println(stderr,"-----------------------------------------------------")
            println(stderr, "rhat: ", tmp_res.rhatξ.ξ)
            println(stderr,"-----------------------------------------------------")
        end

        num2move = 0

        halfburn = convert(Int64, round(mingen/2))

        # we want to generate mingen more samples
        # the first mingen/2 that were previously samples are now burn-in
        # we'll save num_samp + mingen/2 samples
        
        num2move = convert(Int64,round((tot_generated + mingen)/2))

        num2move = tot_samples
        tot_samples = tot_samples + halfburn
        nsamp = tot_samples

        tot_sze = size(states[1],1) #!isnothing(purge_burn) ? purge_burn + nsamp : nburn + nsamp
        tot_save = tot_samples + halfburn

        #num2move = num2move > total ? 0 : total - num2move
        println(stderr,"num2move: ", num2move, " nburn: ", nburn, " nsamp: ", nsamp, " tot_save: ", tot_save, " first_index: ",num2move+1 )

        p = Progress(Int(floor((tot_save-1)/prog_freq));dt=1,showspeed=true, enabled = !suppress_timer,
                     start=Int(round((num2move+1)/prog_freq)))
        channel = RemoteChannel(()->Channel{Bool}(), 1)

        @sync begin 
            @async while take!(channel)
                next!(p)
            end
            t = @async begin
                states = pmap(1:num_chains) do c
                    ## this is super inefficient, should we do this differently?
                    state = Table(τ² = Array{Float64,3}(undef,(tot_save,1,1)), u = Array{Float64,3}(undef,(tot_save,R,V)),
                                  ξ = Array{Float64,3}(undef,(tot_save,V,1)), γ = Array{Float64,3}(undef,(tot_save,q,1)),
                                  S = Array{Float64,3}(undef,(tot_save,q,1)), θ = Array{Float64,3}(undef,(tot_save,1,1)),
                                  Δ = Array{Float64,3}(undef,(tot_save,1,1)), M = Array{Float64,3}(undef,(tot_save,R,R)),
                                  μ = Array{Float64,3}(undef,(tot_save,1,1)), λ = Array{Float64,3}(undef,(tot_save,R,1)),
                                  πᵥ= Array{Float64,3}(undef,(tot_save,R,3)), Σ⁻¹= Array{Float64,3}(undef,(tot_save,R,R)),
                                  invC = Array{Float64,3}(undef,(tot_save,R,R)), μₜ = Array{Float64,3}(undef,(tot_save,R,1)))
                    #for i in 1:num2move
                    #    copy_table!(state,states[c],i,tot_sze - num2move+i)
                    #end
                    copy_table!(state,states[c],1:num2move,(tot_sze - num2move + 1):tot_sze)
                    #      run!(X    ,y,state    ,c,first_index,nburn,total,
                    #           V,R,η,ζ,ι,aΔ,bΔ, ν,rng,prog_freq,
                    #           purge_burn,channel)
                    return run!(X_new,y,state,c,num2move+1,0,tot_save,
                                V,R,η,ζ,ι,aΔ,bΔ,ν,rngs[c],prog_freq,
                                purge_burn,channel)
                end
                put!(channel, false)
            end

            fetch(t)
        end
        #A = num2move > 1 ? num2move+nburn : nburn
        #B = num2move
        tot_generated = tot_generated + mingen
        tmp_res = return_psrf_VOI(states,num_chains,!isnothing(purge_burn) ? purge_burn : nburn,nsamp,R,V,q)
        println(stderr, tot_generated, " samples generated. Max PSRF XI: ", round(max(tmp_res.rhatξ.ξ...),digits=3), ". Max PSRF Gamma: ", round(max(tmp_res.rhatγ.γ...),digits=3))
        stt = isnothing(purge_burn) ? nburn : purge_burn
        @show mean(states[1].ξ[stt+1:(stt+nsamp),:,1],dims=1)[1,:,1]
    end
    stt = isnothing(purge_burn) ? nburn : purge_burn
    println("\n")
    println("R = ",R, " nu=",ν, " nburn= ",nburn, " nsamp = ", nsamp)
    println("\n")
    println(tot_generated, " samples generated. Max PSRF XI: ", round(max(tmp_res.rhatξ.ξ...),digits=4), ". Max PSRF Gamma: ", round(max(tmp_res.rhatγ.γ...),digits=4))
    println("-------------------------------------------------")
    return_psrf_VOI(states,num_chains,!isnothing(purge_burn) ? purge_burn : nburn,nsamp,R,V,q)
end

"""
    Summary(results::Results;interval::Int=95,digits::Int=3)

Generate summary statistics for results: point estimates and credible intervals for edge coefficients, probabilities of influence for individual nodes

# Arguments
- `results`: a Results object, returned from running [`Fit!`](@ref)
- `interval`: (optional) Integer, level for credible intervals. Default is 95%.
- `digits`: (optional) Integer, number of digits (after the decimal) to round results to. Default is 3.

# Returns
A BNRSummary object containing a matrix of edge coefficient point estimates (`coef_matrix`), a matrix of edge coefficient credible intervals (`ci_matrix`), and a DataFrame 
containing the probability of influence of each node (`pi_nodes`).
"""
function Summary(results::Results;interval::Int=95,digits::Int=3)
    nburn = results.burn_in
    nsamp = results.sampled
    total = nburn+nsamp

    lower_bound = (100-interval)/200
    upper_bound = 1-lower_bound
    γ_sorted = sort(results.state.γ[nburn+1:total,:,:],dims=1)
    lw = convert(Int64, round(nsamp * lower_bound))
    hi = convert(Int64, round(nsamp * upper_bound))
    n = size(results.state.γ,2)
    
    ci_df = DataFrame(
             node1        = zeros(Int64,n), 
             node2        = zeros(Int64,n)
            )
    ci_df[:,:estimate]    = round.(mean(results.state.γ[nburn+1:total,:,:],dims=1)[1,:];digits)
    ci_df[:,:lower_bound] = round.(γ_sorted[lw,:,1];digits)
    ci_df[:,:upper_bound] = round.(γ_sorted[hi,:,1];digits)

    q = size(ci_df,1)
    V = Int64((-1 + sqrt( 1 + 8 * q))/2)

    i = 1
    for k = 1:V
        for l = k+1:V
            ci_df[i,:node1] = convert(Int64,k)
            ci_df[i,:node2] = convert(Int64,l)
            i += 1
        end
    end

    xi_df = DataFrame(probability=round.(mean(results.state.ξ[nburn+1:total,:,:],dims=1)[1,:];digits))

    return BNRSummary(ci_df,xi_df,interval)

end

