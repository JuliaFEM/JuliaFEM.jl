# This file is a part of JuliaFEM/FEMAssemble
# License is MIT: see https://github.com/JuliaFEM/FEMAssemble.jl/blob/master/LICENSE.md

module FEMSparse

include("SparseMatrixCOO.jl")
include("SparseVectorCOO.jl")
include("SparseVectorDOK.jl")

export SparseMatrixCOO,
       SparseVectorCOO,
       SparseVectorDOK,
       add!,
       get_nonzero_rows,
       get_nonzero_columns,
       remove_row!,
       remove_column!

end
