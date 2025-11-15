# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
Topology module - concrete implementations.

Abstract type and interface are defined in topology/api.jl.
This file is kept for backward compatibility and to provide any
additional helper functions beyond the core API.
"""

# NOTE: AbstractTopology{N} and interface functions (nnodes, dim, reference_coordinates, edges, faces)
# are now defined in topology/api.jl, which is included before this file in JuliaFEM.jl
