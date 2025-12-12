# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
Cache structures for zero-allocation assembly.

All assemblers use pre-allocated caches containing:
- Global system matrices and vectors (K, f)
- Element/node-level workspace
- Sparse matrix structures

Caches are created once per problem and reused across multiple assembly
calls (e.g., in nonlinear iterations or time stepping).
"""

using SparseArrays
using Tensors

include("coo_cache.jl")
include("csc_cache.jl")
# include("nodal_cache.jl")  # TODO: File doesn't exist yet
