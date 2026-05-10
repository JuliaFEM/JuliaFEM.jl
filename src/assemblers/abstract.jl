# SPDX-FileCopyrightText: 2015-2026 Jukka Aho
# SPDX-License-Identifier: MIT

"""
Abstract assembler type hierarchy.

Assemblers implement the strategy for HOW to assemble finite element systems,
independent of WHAT is being assembled (handled by domain kernels).
"""

"""
    AbstractAssembler

Base type for all assembly strategies.

Assembly strategies define:
- Traversal pattern (element-based vs DOF/nodal-based)
- Sparse output format (currently COO triplets converted to CSC at the end)
- Memory access patterns
- Backend (single-threaded CPU today; KernelAbstractions GPU mirror for the
  DOF-based assembler)

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
"""
abstract type ElementBasedAssembler <: AbstractAssembler end

"""
    NodalBasedAssembler <: AbstractAssembler

Assembly strategy that traverses nodes / DOFs rather than elements.
The only concrete subtype today is `DOFBasedCOOAssembler`, which walks
one DOF row at a time. Future GPU-friendly node-major variants would
sit here as well.
"""
abstract type NodalBasedAssembler <: AbstractAssembler end

# Concrete assembler types

"""
    COOAssembler <: ElementBasedAssembler

Classical element-by-element assembly using COO (coordinate) format.

Strategy: accumulate triplets `(i, j, value)` in vectors, build the sparse
matrix at the end with `sparse(I, J, V, m, n)`.

Performance: straightforward and debuggable, with moderate memory usage.

Best for: prototyping, debugging, simple problems.

Limitations: specialized to the element-based continuum COO path; for
large matrix-free solves prefer `DOFBasedCOOAssembler` and the
matrix-free operators.

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
    DOFBasedCOOAssembler <: NodalBasedAssembler

DOF-by-DOF (row-by-row) assembly using COO format with scalar entry computation.

Strategy: loop over DOFs (matrix rows), compute scalar entries for
touching elements, and scatter to COO triplets. The finest granularity
possible: one matrix entry at a time.

Performance:

- CPU single-thread: roughly 2 - 3x slower than element-based.
- CPU multi-thread: planned row-parallel extension; the current CPU
  implementation is single-threaded.
- GPU: use `dof_based_coo_ka.jl`, which mirrors the same row-wise data
  layout through KernelAbstractions.
- Matrix-free `K*v`: roughly 5 - 10x faster (no matrix storage).

Best for: matrix-free Krylov solvers and very large problems (> ~1M DOFs).

Status: serial CPU and KernelAbstractions GPU mirror implemented;
multi-thread row-parallel CPU is planned.

# Usage

```julia
elements, handler = create_elements!(mesh, ET)
assembler   = DOFBasedCOOAssembler()
cache       = create_cache(assembler, elements, handler, mesh, kernel)
assemble!(cache, assembler, mesh)  # DOF-wise traversal (kernel lives in cache)
K, f        = extract_system(cache)
```
"""
struct DOFBasedCOOAssembler <: NodalBasedAssembler end

# Kernel interface (domain-specific)

"""
    AbstractKernel

Base type for domain-specific assembly kernels.

Kernels define WHAT to assemble (element stiffness, force vector) for
a specific physics domain (continuum, plate, beam, etc.).

Two contracts coexist, depending on the assembler:

- Element-based path (`ElementBasedAssembler`): kernels populate per-element
  block scratch (`element_cache.K_blocks`, `element_cache.f_blocks`) via
  `update_geometry_cache!`, `update_material_cache!`, and per-block
  helpers such as `compute_block!`. The assembler then scatters those
  blocks into the global COO triplet store.
- DOF-based / matrix-free path (`NodalBasedAssembler`): kernels implement
  the microkernel contract in `src/assemblers/microkernel.jl`
  (`qpoint_buffer_eltype`, `update_qpoint_buffer!`, `evaluate_entry`,
  optionally `evaluate_mass_entry`, plus `reference_fields`).

In addition every kernel must answer `dofs_per_node(kernel)` (typically
inherited from `get_field(kernel)`) and provide `get_dof_mapping!` (the
default in this file works for any kernel that exposes a `get_field`).
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
- `GeometryCache`: parametric on storage backing; same struct serves the
  heap-owned legacy form and the SoA `view`-backed form used by the
  DOF-based assembler.
"""
abstract type AbstractGeometryCache end

"""
    AbstractAssemblyMaterialWorkspace{FieldType, StateType}

Abstract base type for per-element assembly material workspaces.

All assembly material workspace implementations must:
- Store material fields and temporary state at all integration points
- Support field access via `workspace.fields[q]` and state via `workspace.states[q]`

Purpose: per-element temporary workspace during assembly (not persistent storage).
The persistent state used for time-stepping lives in `GlobalMaterialCache`.

# Type Parameters
- `FieldType`: NamedTuple type for material fields (e.g., `(σ=..., 𝔻=...)` for mechanics)
- `StateType`: NamedTuple type for state (e.g., `(ε_p=..., α=..., κ=...)` for plasticity)

Both type parameters are concrete `NamedTuple` types derived from
`required_material_fields(material)` and `required_state_variables(material)`.

Concrete implementations:
- `AssemblyMaterialWorkspace{FieldType, StateType}`: Mutable, Vector-based (per-element workspace)

# See Also
- `GlobalMaterialCache`: Persistent state storage for time-stepping (all elements)
"""
abstract type AbstractAssemblyMaterialWorkspace{FieldType, StateType} end

# Backwards-compatible alias for the previous name. The old name was
# misleading because the parameter `StateType` is the per-IP `NamedTuple`
# from the trait system, not anything related to a (now-removed)
# `AbstractMaterialState` type.
const AbstractMaterialStateCache = AbstractAssemblyMaterialWorkspace

# ============================================================================
# KERNEL DEFAULTS — `get_field`, `dofs_per_node`, `get_dof_mapping!`
# ============================================================================
#
# Every concrete `AbstractKernel` participates in DOF distribution through
# three calls. Most single-field kernels only need to expose the underlying
# `AbstractField` they wrap; the defaults below derive everything else.
# Multi-field kernels (e.g. `MixedUPKernel`, thermo-elastic) override
# `dofs_per_node` and `get_dof_mapping!` directly and override `get_field`
# with an error to make accidental single-field assumptions fail loudly.

"""
    get_field(kernel::AbstractKernel) -> AbstractField

Extract the (single) field from a kernel.

Single-field kernels implement this so the defaults for `dofs_per_node`
and `get_dof_mapping!` work without further intervention. Multi-field
kernels override `dofs_per_node` / `get_dof_mapping!` directly and
typically override `get_field` to throw.
"""
function get_field(kernel::AbstractKernel)
    error("get_field not implemented for $(typeof(kernel)). " *
          "Either implement get_field(::$(typeof(kernel))) or override " *
          "dofs_per_node / get_dof_mapping! directly.")
end

"""
    dofs_per_node(kernel::AbstractKernel) -> Int

Default kernel-level DOFs per node: delegate to `get_field(kernel)`.
Multi-field kernels override directly.
"""
@inline function dofs_per_node(kernel::AbstractKernel)
    return dofs_per_node(get_field(kernel))
end

"""
    operator_is_posdef(kernel::AbstractKernel) -> Bool

Trait answering whether the matrix-free stiffness operator built around
`kernel` is symmetric positive-definite. The default is `true` (single-field
elliptic problems such as continuum displacement, heat). Mixed / saddle-point
kernels (`MixedUPKernel`, `StokesMixedKernel`, `HellingerReissnerKernel`,
`HuWashizuKernel`, `BiotPoroelasticKernel`, `ThermoPoroelasticKernel`) override this to `false` so Krylov stacks (CG, etc.) do
not pick the SPD branch.
"""
@inline operator_is_posdef(::AbstractKernel) = true

"""
    get_dof_mapping!(dofs, kernel::AbstractKernel, element_id, mesh) -> Nothing

Fill `dofs` with the global DOF indices for `element_id`, in node-major
order, using `dofs_per_node(kernel)` and the kernel's underlying field.

Zero-allocation. Multi-field kernels override.
"""
@inline function get_dof_mapping!(
    dofs::AbstractVector{Int},
    kernel::AbstractKernel,
    element_id::Int,
    mesh::AbstractMesh,
)
    nodes = mesh.connectivity[element_id]
    ndofs_per_node = dofs_per_node(get_field(kernel))

    nnodes = length(nodes)
    idx = 1
    @inbounds for i in 1:nnodes
        node_id = Int(nodes[i])
        for component in 1:ndofs_per_node
            dofs[idx] = ndofs_per_node * (node_id - 1) + component
            idx += 1
        end
    end
    return nothing
end
