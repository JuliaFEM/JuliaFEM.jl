# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md


""" Find all nonzero rows from sparse matrix.

Returns
-------

Ordered list of row indices.
"""
function get_nonzero_rows(A::SparseMatrixCSC)
    return sort(unique(rowvals(A)))
end

function get_nonzero_columns(A::SparseMatrixCSC)
    return get_nonzero_rows(transpose(A))
end

""" Resize sparse matrix A to (higher) dimension n x m. """
function resize_sparse(A::SparseMatrixCSC, n::Int, m::Int)
    I, J, V = findnz(A)
    @assert length(I) <= n
    @assert length(J) <= m
    return sparse(I, J, V, n, m)
end

""" Resize sparse vector b to (higher) dimension n. """
function resize_sparsevec(b::SparseVector, n::Int)
    return sparsevec(findnz(b)..., n)
end

#=
function resize!(A::SparseMatrixCSC, m::Int64, n::Int64)
    if (n == A.n) && (m == A.m)
        return
    end
    @assert n >= A.n
    @assert m >= A.m
    append!(A.colptr, A.colptr[end]*ones(Int, m-A.m))
    A.n = n
    A.m = m
end
=#
