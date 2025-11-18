# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
Abstract assembler type hierarchy.

Assemblers implement the strategy for HOW to assemble finite element systems,
independent of WHAT is being assembled (handled by domain kernels).
"""

"""
    AbstractAssembler

Base type for all assembly strategies.

Assembly strategies define:
- Traversal pattern (element-based vs nodal-based)
- Sparse matrix format (COO, CSC)
- Memory access patterns
- Backend (CPU, GPU)

All assemblers use pre-allocated cache structures for zero-allocation assembly.
"""
abstract type AbstractAssembler end

"""
    AbstractAssemblerCache

Base type for pre-allocated assembly workspace.

Caches contain all memory needed for assembly:
- Global matrices and vectors (K, f)
- Element/node-level workspace
- DOF mapping buffers
- Integration point data

Caches are created once and reused across multiple assembly calls
(e.g., in nonlinear iterations).
"""
abstract type AbstractAssemblerCache end

"""
    ElementBasedAssembler <: AbstractAssembler

Assembly strategy that traverses elements.

Element-based assemblers loop over all elements, compute local stiffness
matrices, and scatter to global system. This is the classical FEM approach.

Concrete types:
- `COOAssembler`: Accumulate triplets, build sparse matrix at end
- `CSCAssembler`: Pre-built CSC structure, in-place assembly
"""
abstract type ElementBasedAssembler <: AbstractAssembler end

"""
    NodalBasedAssembler <: AbstractAssembler

Assembly strategy that traverses nodes.

Nodal-based assemblers loop over nodes, then gather contributions from
all elements touching that node. This pattern is GPU-friendly (one thread
per node) and has better cache locality for nodal DOFs.

Concrete types:
- `NodalAssembler`: Node-by-node assembly with node-to-elements map
"""
abstract type NodalBasedAssembler <: AbstractAssembler end

# Concrete assembler types

"""
    COOAssembler <: ElementBasedAssembler

Classical element-by-element assembly using COO (coordinate) format.

**Strategy**: Accumulate triplets `(i, j, value)` in vectors, build sparse
matrix at end using `sparse(I, J, V, m, n)`.

**Performance**: Baseline (1.0x), moderate memory usage.

**Best for**: Prototyping, debugging, simple problems.

**Limitations**: Slower than CSC for repeated assembly (nonlinear problems).

# Usage

```julia
assembler = COOAssembler()
cache = create_cache(assembler, mesh, kernel)
assemble!(cache, assembler, kernel, mesh)
K, f = extract_system(cache)
```
"""
struct COOAssembler <: ElementBasedAssembler end

"""
    CSCAssembler <: ElementBasedAssembler

Optimized assembly using pre-built CSC (compressed sparse column) structure.

**Strategy**: Build sparsity pattern once, reuse structure across assembly
calls. Use two-pointer merge algorithm to insert element contributions
directly into CSC arrays.

**Performance**: 4.1x faster than COO, 16.6x less memory.

**Best for**: Production code, nonlinear problems (repeated assembly).

**Algorithm**: Inspired by Ferrite.jl but adapted for JuliaFEM architecture.

# Usage

```julia
assembler = CSCAssembler()
cache = create_cache(assembler, mesh, kernel)  # Pre-builds sparsity pattern

# Nonlinear loop
for iteration in 1:max_iter
    assemble!(cache, assembler, kernel, mesh)  # Zero allocations!
    K, f = extract_system(cache)
    # ... solve, update ...
end
```
"""
struct CSCAssembler <: ElementBasedAssembler end

"""
    NodalAssembler <: NodalBasedAssembler

Node-by-node assembly using inverse connectivity (node-to-elements map).

**Strategy**: For each node, gather contributions from all touching elements.
Natural for GPU parallelization (one thread per node).

**Performance**: Expected 2-10x speedup on GPU for large problems (> 100k nodes).

**Best for**: GPU acceleration, very large problems.

**Status**: Planned for future implementation.

# Usage

```julia
assembler = NodalAssembler()
cache = create_cache(assembler, mesh, kernel)
assemble!(cache, assembler, kernel, mesh)
K, f = extract_system(cache)
```
"""
struct NodalAssembler <: NodalBasedAssembler end

# Kernel interface (domain-specific)

"""
    AbstractKernel

Base type for domain-specific assembly kernels.

Kernels define WHAT to assemble (element stiffness, force vector) for
a specific physics domain (continuum, plate, beam, etc.).

Required interface:
- `compute_element_stiffness!(cache, kernel, element_id, ...)`: Compute Ke, fe
- `dofs_per_node(kernel)`: Number of DOFs per node
- `get_dof_mapping!(dofs, kernel, element_id, mesh)`: Fill DOF indices

See `src/assemblers/kernel_interface.jl` for detailed interface specification.
"""
abstract type AbstractKernel end

# Cache types for element/node-level workspace

"""
    ElementCache

Workspace for element-level computations.

Contains pre-allocated arrays for:
- Local stiffness matrix `Ke` [ndofs_elem × ndofs_elem]
- Local force vector `fe` [ndofs_elem]
- Node coordinates `coords` [nnodes_elem × ndim]
- Global DOF indices `dofs` [ndofs_elem]
- Integration point data

Zero allocations during assembly - all arrays reused.
"""
struct ElementCache
    Ke::Matrix{Float64}       # Local stiffness matrix
    fe::Vector{Float64}       # Local force vector
    coords::Matrix{Float64}   # Element node coordinates
    dofs::Vector{Int}         # Global DOF indices
end

"""
    NodeCache

Workspace for node-level computations (nodal assembly).

Contains pre-allocated arrays for:
- Node DOF contributions
- Element indices touching this node
- Local-to-global mapping buffers

Used by `NodalAssembler`.
"""
struct NodeCache
    node_dofs::Vector{Int}           # Global DOF indices for this node
    touching_elements::Vector{Int}   # Elements touching this node
    local_indices::Vector{Int}       # Local node indices in elements
end

"""
    create_element_cache(mesh::AbstractMesh, kernel::AbstractKernel) -> ElementCache

Create pre-allocated workspace for element assembly.

# Arguments
- `mesh`: Finite element mesh
- `kernel`: Domain kernel defining DOF structure

# Returns
- `ElementCache` with arrays sized for largest element in mesh
"""
function create_element_cache(mesh::AbstractMesh, kernel::AbstractKernel)
    # Get maximum element size from Mesh{N,T} type parameters
    MeshType = typeof(mesh)
    max_nnodes_elem = MeshType.parameters[1]::Int
    TopologyType = MeshType.parameters[2]  # T from Mesh{N,T}
    ndofs_per_node = dofs_per_node(kernel)
    max_ndofs_elem = max_nnodes_elem * ndofs_per_node
    ndim = dim(TopologyType())

    return ElementCache(
        zeros(max_ndofs_elem, max_ndofs_elem),  # Ke
        zeros(max_ndofs_elem),                   # fe
        zeros(max_nnodes_elem, ndim),            # coords
        zeros(Int, max_ndofs_elem)               # dofs
    )
end

"""
    create_node_cache(mesh::AbstractMesh, kernel::AbstractKernel) -> NodeCache

Create pre-allocated workspace for nodal assembly.

# Arguments
- `mesh`: Finite element mesh
- `kernel`: Domain kernel defining DOF structure

# Returns
- `NodeCache` with arrays sized for node with most touching elements
"""
function create_node_cache(mesh::AbstractMesh, kernel::AbstractKernel)
    # Get maximum number of elements touching any node
    node_to_elements = NodeToElementsMap(mesh)
    max_touching = maximum(length(get_node_spider(node_to_elements, i))
                          for i in 1:nnodes_total(mesh))
    ndofs_per_node = dofs_per_node(kernel)

    return NodeCache(
        zeros(Int, ndofs_per_node),      # node_dofs
        zeros(Int, max_touching),        # touching_elements
        zeros(Int, max_touching)         # local_indices
    )
end
