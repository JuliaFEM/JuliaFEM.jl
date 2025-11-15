# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
Mesh API definitions.

This file defines mesh-specific abstract types and interfaces.
Must be included after core api.jl.
"""

# ============================================================================
# MESH ABSTRACTIONS
# ============================================================================

"""
    AbstractMesh

Abstract type for all mesh structures.

# Interface Requirements

Concrete mesh types must implement:
- `nnodes_total(mesh)` - Total number of nodes
- `nelements(mesh)` - Total number of elements
- `get_node(mesh, node_id)` - Get node coordinates
- `connectivity_matrix(mesh)` - Element connectivity
- `get_elements_for_node(mesh, node_id)` - Elements containing node
- `get_element_set(mesh, set_name)` - Get named element set
- `get_node_set(mesh, set_name)` - Get named node set

# Concrete Types
- `Mesh{T<:AbstractTopology}` - Parametric mesh with single topology type
- `MixedMesh` - Mesh with multiple topology types (future)

# Design Philosophy

Meshes own topology (node coordinates, connectivity).
Physics references meshes (does not own them).
Multiple Physics can share one Mesh (multiphysics coupling).

# Element Sets and Node Sets

Meshes support named sets for:
- Applying boundary conditions
- Defining material regions
- Post-processing specific regions

# See Also
- Concrete implementation: `Mesh{T}` in src/mesh/mesh.jl
"""
abstract type AbstractMesh end

# ============================================================================
# MESH OPERATIONS
# ============================================================================

"""
    nnodes_total(mesh::AbstractMesh) -> Int

Total number of nodes in the mesh.

# Examples
```julia
mesh = Mesh(Tet4(), nodes, connectivity)
n = nnodes_total(mesh)  # e.g., 1000 nodes
```
"""
function nnodes_total end

"""
    nelements(mesh::AbstractMesh) -> Int

Total number of elements in the mesh.

# Examples
```julia
mesh = Mesh(Hex8(), nodes, connectivity)
ne = nelements(mesh)  # e.g., 500 elements
```
"""
function nelements end

"""
    get_node(mesh::AbstractMesh, node_id::Int) -> Vec{Dim}

Get coordinates of a node by its ID.

# Arguments
- `mesh`: Mesh structure
- `node_id`: Node identifier (1-based)

# Returns
- Node coordinates as `Vec{Dim}` (Dim = 1, 2, or 3)

# Examples
```julia
X = get_node(mesh, 42)  # Vec{3}(1.0, 2.0, 3.0)
```
"""
function get_node end

"""
    connectivity_matrix(mesh::AbstractMesh) -> Matrix{Int}

Get element connectivity matrix.

# Returns
- Matrix where each row is an element's node IDs
- Size: `(nelements, nnodes_per_element)`

# Examples
```julia
conn = connectivity_matrix(mesh)
elem1_nodes = conn[1, :]  # [1, 2, 3, 4] for Tet4
```
"""
function connectivity_matrix end

"""
    get_elements_for_node(mesh::AbstractMesh, node_id::Int) -> Vector{Int}

Get all elements containing a given node.

Critical for nodal assembly (node-to-elements map).

# Arguments
- `mesh`: Mesh structure
- `node_id`: Node identifier

# Returns
- Vector of element IDs containing this node

# Examples
```julia
elems = get_elements_for_node(mesh, 10)  # [5, 6, 7, 8, 12, 15]
```

# See Also
- Nodal assembly: docs/book/multigpu_nodal_assembly.md
"""
function get_elements_for_node end

"""
    get_element_set(mesh::AbstractMesh, set_name::String) -> Vector{Int}

Get element IDs in a named element set.

# Arguments
- `mesh`: Mesh structure
- `set_name`: Name of element set (e.g., "body", "surface1")

# Returns
- Vector of element IDs in the set

# Examples
```julia
body_elements = get_element_set(mesh, "body")
surface_elements = get_element_set(mesh, "traction_surface")
```

# See Also
- [`get_node_set`](@ref) for node sets
"""
function get_element_set end

"""
    get_node_set(mesh::AbstractMesh, set_name::String) -> Vector{Int}

Get node IDs in a named node set.

# Arguments
- `mesh`: Mesh structure
- `set_name`: Name of node set (e.g., "fixed_nodes", "boundary")

# Returns
- Vector of node IDs in the set

# Examples
```julia
fixed_nodes = get_node_set(mesh, "fixed_support")
loaded_nodes = get_node_set(mesh, "load_application")
```

# See Also
- [`get_element_set`](@ref) for element sets
"""
function get_node_set end

# ============================================================================
# MESH REFINEMENT
# ============================================================================

"""
    AbstractRefineStrategy

Abstract type for mesh refinement strategies.

# Concrete Strategies
- `LongestEdgeBisection`: Recursive longest edge bisection
- `RedGreenRefinement`: Red-green triangulation
- `Uniform`: Uniform refinement (split all elements)

# See Also
- [`refine`](@ref) for refinement function
- Implementation: src/mesh/refine.jl
"""
abstract type AbstractRefineStrategy end

"""
    refine(mesh::AbstractMesh, strategy::AbstractRefineStrategy) -> AbstractMesh

Refine a mesh using a refinement strategy.

# Arguments
- `mesh`: Original mesh
- `strategy`: Refinement strategy

# Returns
- Refined mesh (new instance)

# Examples
```julia
# Longest edge bisection
refined = refine(mesh, LongestEdgeBisection())

# Uniform refinement
refined = refine(mesh, Uniform())
```

# See Also
- [`AbstractRefineStrategy`](@ref) for available strategies
"""
function refine end
