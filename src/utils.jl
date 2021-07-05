"""
    create_upper_tri(vec,V)

Create an upper triangluar matrix from a vector of the form [12, ... 1V,23,...(V-1)V]
to the form [0 12 13  ... 1V]
            [0 0  23  ... 2V]
            [...............]
            [0 0  0...(V-1)V]
# Arguments
- `vec`: vector containing values to put into the upper triangluar matrix
- `V`  : dimension of output matrix

# Returns
Upper triangluar matrix containing values of `vec`
"""
function create_upper_tri(vec::Array{T,1},V::Int64) where T <: Union{Int64,Float64}
    mat = zeros(T,V,V)
    vec2 = deepcopy(vec)
    for k = 1:V
        for l = k+1:V
            mat[k,l] = popfirst!(vec2)
        end
    end
    return mat
end

"""
    upper_triangle(matrix)

Return the upper triangle (without the diagonal) of the matrix as a vector

# Arguments
- `matrix`: matrix of which to capture the upper triangle

# Returns
Vector of upper triangluar section of `matrix`
"""
function upper_triangle(matrix::Array{T,2}) where T <: Union{Int64,Float64}
    k = 1
    ret = zeros(T,convert(Int64, round(size(matrix,1)*(size(matrix,2) - 1)/2)))
    for i in 1:size(matrix,1)
        for j in (i+1):size(matrix,2)
            ret[k] = matrix[i,j]
            k = k + 1
        end
    end
    return ret
end
