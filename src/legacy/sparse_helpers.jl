# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md
#
# Sparse-matrix convenience helpers used by the legacy linear-system /
# constraint code (`legacy/solvers.jl`, `legacy/linear_system.jl`).
# These are CSC-only; the modern element-/DOF-based assembly pipeline
# does not need them and they are not part of the default export
# surface. They live inside `JuliaFEM.Legacy` so that they are loaded
# only when `JULIAFEM_ENABLE_LEGACY=1`.

"""
    get_nonzero_rows(A::SparseMatrixCSC) -> Vector{Int}

Return the sorted, deduplicated row indices of all stored entries of `A`.
CSC-only: walks `rowvals(A)` directly.
"""
function get_nonzero_rows(A::SparseMatrixCSC)
    return sort(unique(rowvals(A)))
end

"""
    get_nonzero_columns(A::SparseMatrixCSC) -> Vector{Int}

Return the sorted, deduplicated column indices of all stored entries of
`A`. Implemented as `get_nonzero_rows(transpose(A))`.
"""
function get_nonzero_columns(A::SparseMatrixCSC)
    return get_nonzero_rows(copy(transpose(A)))
end

"""
    resize_sparse(A::AbstractMatrix, n::Int, m::Int) -> SparseMatrixCSC

Return a `n × m` CSC matrix containing the nonzero entries of `A`.
Generic over `AbstractMatrix` (walks `findall(!iszero, A)`).
"""
function resize_sparse(A::AbstractMatrix, n::Int, m::Int)
    idx = findall(!iszero, A)
    I = getindex.(idx, 1)
    J = getindex.(idx, 2)
    V = [A[i] for i in idx]
    return sparse(I, J, V, n, m)
end

"""
    resize_sparsevec(b::SparseVector, n::Int) -> SparseVector

Return a length-`n` sparse vector containing the nonzero entries of `b`.
"""
function resize_sparsevec(b::SparseVector, n::Int)
    return sparsevec(b.nzind, b.nzval, n)
end
