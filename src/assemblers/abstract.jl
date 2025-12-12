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
    NodeBasedCOOAssembler <: NodalBasedAssembler

Node-by-node assembly using COO format with block integration.

**Strategy**: Loop over nodes, compute 3×3 stiffness blocks for touching elements,
scatter to COO triplets. Uses `PreparedElement` and `compute_block!` from
continuum kernel for efficient block-based integration.

**Performance**: 
- CPU single-thread: ~1.5-2× slower than element-based (more kernel calls)
- CPU multi-thread: ~1.5-2× faster (better parallelization)
- GPU: ~10-50× faster (massive parallelism, no atomics)

**Best for**: GPU acceleration, contact mechanics, matrix-free methods.

**Status**: ✅ Implemented

# Usage

```julia
assembler = NodeBasedCOOAssembler()
cache = create_cache(assembler, mesh, kernel)
assemble!(cache, assembler, kernel, mesh)  # Nodal traversal!
K, f = extract_system(cache)
```
"""
struct NodeBasedCOOAssembler <: NodalBasedAssembler end

"""
    DOFBasedCOOAssembler <: NodalBasedAssembler

DOF-by-DOF (row-by-row) assembly using COO format with scalar entry computation.

**Strategy**: Loop over DOFs (matrix rows), compute scalar entries for touching elements,
scatter to COO triplets. Finest granularity possible - computes ONE matrix entry at a time.

**Performance**: 
- CPU single-thread: ~2-3× slower than element-based (finest granularity)
- CPU multi-thread: ~Same as node-based (parallelization over rows)
- GPU: ~Same as node-based (one thread per DOF)
- Matrix-free K*v: ~5-10× faster (no matrix storage)

**Best for**: Matrix-free Krylov solvers, very large problems (> 1M DOFs).

**Status**: ✅ Implemented

# Usage

```julia
assembler = DOFBasedCOOAssembler()
cache = create_cache(assembler, mesh, kernel)
assemble!(cache, assembler, kernel, mesh)  # DOF-wise traversal!
K, f = extract_system(cache)
```
"""
struct DOFBasedCOOAssembler <: NodalBasedAssembler end

"""
    NodalAssembler <: NodalBasedAssembler

Generic node-by-node assembly (future: CSC, GPU variants).

**Strategy**: For each node, gather contributions from all touching elements.
Natural for GPU parallelization (one thread per node).

**Performance**: Expected 2-10x speedup on GPU for large problems (> 100k nodes).

**Best for**: GPU acceleration, very large problems.

**Status**: Placeholder for future GPU-optimized implementations.

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

# ============================================================================
# ELEMENT-LEVEL CACHE ABSTRACTIONS
# ============================================================================

"""
    AbstractGeometryCache

Abstract base type for geometry caches.

All geometry cache implementations must:
- Store node coordinates, shape function gradients, and integration weights
- Support indexing: `cache.X[i]`, `cache.∇N_data[q, k]`, `cache.detJ_w[q]`

Concrete implementations:
- `GeometryCache`: Mutable, Vector-based (convenient for updates)
- `ImmutableGeometryCache{N,NIP}`: Immutable, NTuple-based (zero allocations, still parametric)

Note: Type parameters removed from GeometryCache to avoid 80 bytes allocation in hot loops.
Sizes (N, NIP) can be inferred from field dimensions at runtime.
"""
abstract type AbstractGeometryCache end

"""
    AbstractMaterialStateCache{FieldType, StateType}

Abstract base type for assembly material workspaces.

All assembly material workspace implementations must:
- Store material fields and temporary state at all integration points
- Support field access via `workspace.fields[q]` and state via `workspace.states[q]`

**Purpose:** Per-element temporary workspace during assembly (not persistent storage).

# Type Parameters
- `FieldType`: NamedTuple type for material fields (e.g., `(σ=..., 𝔻=...)` for mechanics)
- `StateType`: NamedTuple type for state (e.g., `(ε_p=..., α=..., κ=...)` for plasticity)

Concrete implementations:
- `AssemblyMaterialWorkspace{FieldType, StateType}`: Mutable, Vector-based (compositional, per-element workspace)

# See Also
- `GlobalMaterialCache`: Persistent state storage for time-stepping (all elements)
"""
abstract type AbstractMaterialStateCache{FieldType, StateType} end
