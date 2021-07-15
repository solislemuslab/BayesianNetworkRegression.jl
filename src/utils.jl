"""
    create_lower_tri(vec,V)

Create a lower triangluar matrix from a vector of the form [12, ... 1V,23,...(V-1)V]
to the form [0  0..........]
            [12 0 .........]
            [13 23 0.......]
            [..............]
            [1V ...(V-1)V 0]
# Arguments
- `vec`: vector containing values to put into the upper triangluar matrix
- `V`  : dimension of output matrix

# Returns
Upper triangluar matrix containing values of `vec`
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

Return the upper triangle (without the diagonal) of the matrix as a vector

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
