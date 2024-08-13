"""
    create_lower_tri(vec,V)

Create a lower triangluar matrix from a vector of the form [12, ... 1V,23,...(V-1)V]
in the form [11  0............]
            [12 22  0.........]
            [13 23 33 0.......]
            [.................]
            [1V .....(V-1)V VV]
# Arguments
- `vector`: vector containing values to put into the upper triangluar matrix
- `V`  : dimension of output matrix

# Returns
Lower triangluar matrix containing values of the vector
"""
function create_lower_tri(vector::AbstractVector{T},V) where {T}
    mat = zeros(T,V,V)
    i = 1
    for k = 1:V
        for l = k:V
            mat[l,k] = vector[i,1] 
            i += 1
        end
    end
    return mat
end

"""
    lower_triangle(matrix)

Return the lower triangle (without the diagonal) of the matrix as a vector

# Arguments
- `matrix`: matrix of which to capture the upper triangle

# Returns
Vector of upper triangluar section of `matrix`
"""
function lower_triangle(matrix::AbstractArray{T,2}) where {T}
    if size(matrix,1) != size(matrix,2)
        print("error: matrix must be square")
        return 0
    end

    V = size(matrix,1)

    k = 1
    ret = zeros(T,convert(Int64, round(V*(V + 1)/2)))
    for i in 1:size(matrix,1)
        for j in i:size(matrix,2)
            ret[k,1] = matrix[j,i]
            k = k + 1
        end
    end
    return ret
end

"""
    copy_table!(table,to,from)

Copy contents of one index from a table to another. 

# Arguments
- `table`: table object to copy contents from/to
- `to`: index to copy to
- `from`: index to copy from

# Returns
Nothing, all updates are done in place
"""
function copy_table!(table,to,from)
    table.τ²[to,:,:] = deepcopy(table.τ²[from,:,:])
    table.u[to,:,:] = deepcopy(table.u[from,:,:])
    table.ξ[to,:,:] = deepcopy(table.ξ[from,:,:])
    table.γ[to,:,:] = deepcopy(table.γ[from,:,:])
    table.S[to,:,:] = deepcopy(table.S[from,:,:])
    table.θ[to,:,:] = deepcopy(table.θ[from,:,:])
    table.Δ[to,:,:] = deepcopy(table.Δ[from,:,:])
    table.M[to,:,:] = deepcopy(table.M[from,:,:])
    table.μ[to,:,:] = deepcopy(table.μ[from,:,:])
    table.λ[to,:,:] = deepcopy(table.λ[from,:,:])
    table.πᵥ[to,:,:] = deepcopy(table.πᵥ[from,:,:])
end


"""
    copy_table!(table,to,from)

Copy contents of one index from one table to another index in a second. 

# Arguments
- `table_t`: table object to copy contents to
- `table_f`: table object to copy contents from
- `to`: index to copy to
- `from`: index to copy from

# Returns
Nothing, all updates are done in place
"""
function copy_table!(table_t,table_f,to,from)
    table_t.τ²[to,:,:] = deepcopy(table_f.τ²[from,:,:])
    table_t.u[to,:,:] = deepcopy(table_f.u[from,:,:])
    table_t.ξ[to,:,:] = deepcopy(table_f.ξ[from,:,:])
    table_t.γ[to,:,:] = deepcopy(table_f.γ[from,:,:])
    table_t.S[to,:,:] = deepcopy(table_f.S[from,:,:])
    table_t.θ[to,:,:] = deepcopy(table_f.θ[from,:,:])
    table_t.Δ[to,:,:] = deepcopy(table_f.Δ[from,:,:])
    table_t.M[to,:,:] = deepcopy(table_f.M[from,:,:])
    table_t.μ[to,:,:] = deepcopy(table_f.μ[from,:,:])
    table_t.λ[to,:,:] = deepcopy(table_f.λ[from,:,:])
    table_t.πᵥ[to,:,:] = deepcopy(table_f.πᵥ[from,:,:])
end



"""
Function to print the citation of the paper for users of BayesianNetworkRegression.jl
- returnstring = false (Default); it only prints the citation to screen.
"""
function citation(;returnstring=false)
    str = "If you use BayesianNetworkRegression.jl, please cite:\n"
    str *= "@article{Ozminkowski2022,\n"
    str *= "author = {Ozminkowski, S. and Sol\'{i}s-Lemus, C.},\n"
    str *= "year = {2022},\n"
    str *= "title = {{Identifying microbial drivers in biological phenotypes with a Bayesian Network Regression model}},\n"
    str *= "journal = {In preparation}\n"
##    println("volume = {34},")
##    println("number = {12},")
##    println("pages = {3292--3298},")
##    println("pmid = {28961984}")
    str *= "}"
    if !returnstring
        println(str)
        return nothing
    else
        return str
    end
end


"""
Auxiliary functions needed in the tests folder
"""
function symmetrize_matrices(X)
    X_new = Array{Array{Int8,2},1}(undef,0)
    for i in 1:size(X,1)
        B = convert(Matrix, reshape(X[i], 4, 4))
        push!(X_new,Symmetric(B))
    end
    X = X_new
end