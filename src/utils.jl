"""
    create_lower_tri(vec,V)

Create a lower triangluar matrix from a vector of the form [12, ... 1V,23,...(V-1)V]
in the form [0  0..........]
            [12 0 .........]
            [13 23 0.......]
            [..............]
            [1V ...(V-1)V 0]
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
        for l = k+1:V
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
    k = 1
    ret = zeros(T,convert(Int64, round(size(matrix,1)*(size(matrix,2) - 1)/2)))
    for i in 1:size(matrix,1)
        for j in (i+1):size(matrix,2)
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