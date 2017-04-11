# This file is a part of JuliaFEM/FEMAssemble
# License is MIT: see https://github.com/JuliaFEM/FEMAssemble.jl/blob/master/LICENSE.md

module FEMSparse

include("SparseMatrixCOO.jl")
include("SparseVectorCOO.jl")
include("SparseVectorDOK.jl")

export SparseMatrixCOO, SparseVectorCOO, SparseVectorDOK, add!

#=
""" Resize sparse matrix A to (higher) dimension n x m. """
function resize_sparse(A, n, m)
    return sparse(findnz(A)..., n, m)
end

""" Resize sparse vector b to (higher) dimension n. """
function resize_sparsevec(b, n)
    return sparsevec(findnz(b)..., n)
end
=#

end
