---
title: "ADR 002: Topology Types With Node Count Parameters"
description: "Architecture decision: topology types include node count parameter from mesh"
date: "November 13, 2025"
author: "Jukka Aho"
categories: ["architecture", "design-decision", "adr"]
keywords: ["topology", "basis", "lagrange", "node-count", "type-parameters"]
audience: "researchers"
level: "advanced"
type: "adr"
series: "The JuliaFEM Book"
chapter: 3
status: "accepted"
supersedes: "ADR 002 (November 9, 2025)"
---

## Context

In the original November 9, 2025 decision, we separated topology (geometry) from node count, arguing that node count should come from the basis function choice. However, practical implementation revealed that **node count comes from the mesh**, not from the basis choice.

Commercial FEM codes (Code Aster, Abaqus, Nastran) use element type names like `TRIA3`, `TRIA6`, `QUAD4`, `QUAD8`, `HEXA8`, `HEXA20`, `HEXA27`. These names **hardcode the node count** into the type identifier.

**Key insight:** When reading a mesh file (Abaqus `.inp`, Code Aster `.med`, GMSH `.msh`), the **mesh explicitly specifies node count**:
- "This is a hex element with connectivity (1,2,3,4,5,6,7,8)" → 8 nodes
- "This is a hex element with connectivity (1,2,...,20)" → 20 nodes
- "This is a hex element with connectivity (1,2,...,27)" → 27 nodes

The mesh reader knows the node count **before** any basis functions are chosen. The basis must match the node count, not determine it.

## Decision

**Topology types include a node count type parameter.** The node count comes from the mesh at runtime, and we extract it into the type system for compile-time performance.

### New Design

```julia
# Topology = Geometry + Node Count (from mesh)
abstract type AbstractTopology end

struct Hexahedron{N} <: AbstractTopology end
struct Tetrahedron{N} <: AbstractTopology end
struct Triangle{N} <: AbstractTopology end
struct Quadrilateral{N} <: AbstractTopology end

# Common aliases
const Hex8 = Hexahedron{8}
const Hex20 = Hexahedron{20}
const Hex27 = Hexahedron{27}
const Tet4 = Tetrahedron{4}
const Tet10 = Tetrahedron{10}

# Basis functions must MATCH the topology node count
struct Lagrange{T<:AbstractTopology, P} <: AbstractBasis end

# Validation at construction
function Element(topology::Hexahedron{N}, basis::Lagrange{Hexahedron{N}, P}, ...) where {N, P}
    # N from topology must match N expected by basis
    @assert N == nnodes(basis) "Node count mismatch"
    ...
end
```

## Rationale

### 1. Node Count Comes From Mesh

When reading mesh files, the connectivity explicitly specifies node count:

```python
# Abaqus .inp file
*ELEMENT, TYPE=C3D8
1, 1, 2, 3, 4, 5, 6, 7, 8      # 8 nodes → Hex8
*ELEMENT, TYPE=C3D20
2, 1, 2, 3, ..., 20             # 20 nodes → Hex20
```

The mesh reader creates elements **knowing the node count**. We must capture this in the type system.

### 2. Compile-Time Performance

Using `Val(N)` for node count enables compile-time loop unrolling:

```julia
# WITHOUT type parameter - runtime variable
function compute_element_stiffness!(K_blocks, X, material, ::Type{Hexahedron})
    N = length(X)  # Runtime variable!
    dN_dx = ntuple(i -> f(i), N)  # ALLOCATES! (runtime size unknown)
end

# WITH type parameter - compile-time constant
function compute_element_stiffness!(K_blocks, X, material, ::Type{Hexahedron{N}}) where {N}
    dN_dx = ntuple(i -> f(i), Val(N))  # Zero allocation! (compile-time size)
end
```

**Measured impact:** Without type parameter, 3.8 MB allocations. With `Val(N)`, 1.4 KB (99.96% reduction).

### 3. Separation Still Maintained

We still separate concerns, but node count is part of **topology from mesh**:

| Concern | What it defines | Example |
|---------|----------------|---------|
| **Topology + Node Count** | Geometric shape from mesh | `Hexahedron{8}` (from mesh connectivity) |
| **Basis** | Polynomial space matching topology | `Lagrange{Hexahedron{8}, 1}` (linear) |
| **Integration** | Numerical quadrature | `Gauss{2}` |

The basis must **match** the topology node count (validated at construction).

### 4. Mathematical Correctness Preserved

Node count still comes from **topology + polynomial degree**, but the mesh specifies this combination:

- **Mesh says:** "hex element, 8 nodes" → `Hexahedron{8}`
- **Basis infers:** 8 nodes = P1 Lagrange → `Lagrange{Hexahedron{8}, 1}`
- **Validation:** `nnodes(Lagrange{Hexahedron{8}, 1}) == 8` ✓

The relationship is still correct: 8 nodes uniquely determines P1, 20 nodes uniquely determines P2 serendipity, 27 nodes uniquely determines P2 full.

### 5. Eliminates Runtime Overhead

**Old approach (November 9 decision):**

```julia
struct Hexahedron <: AbstractTopology end  # No parameter

function assemble_element(X, topology::Hexahedron)
    N = length(X)  # Runtime computation
    # All loops use N (runtime variable)
end
```

**New approach (November 13 decision):**

```julia
struct Hexahedron{N} <: AbstractTopology end

function assemble_element(X, ::Type{Hexahedron{N}}) where {N}
    # N known at compile time
    # Loops unroll, ntuple is zero-allocation
end
```

## Consequences

### Positive

✅ **Zero-allocation assembly** via compile-time node count (`Val(N)`)  
✅ **Direct mesh compatibility** (node count from connectivity)  
✅ **Type-level dispatch** for performance-critical code  
✅ **Clear semantics** (`Hex8` = "hexahedron from mesh with 8 nodes")  
✅ **Simplifies implementation** (no wrappers, no runtime queries)  
✅ **Enables advanced element types** (edge, face DOFs still work)

### Negative

⚠️ **Type parameters propagate** through code (`AbstractTopology{N}` everywhere)  
⚠️ **More verbose types** (`Hexahedron{8}` instead of `Hexahedron`)  
⚠️ **Need type aliases** (`const Hex8 = Hexahedron{8}` for convenience)

### Neutral

⚡ **Still separates concerns** (topology from basis, node count from DOF count)  
⚡ **Mathematical correctness** maintained (node count = topology + polynomial degree)  
⚡ **Basis validation** required at construction (`nnodes(basis) == N`)

## Implementation

### Current Status (November 13, 2025)

Already implemented in codebase:

```julia
# src/topology/hexahedron.jl
struct Hexahedron{N} <: AbstractTopology end
const Hex8 = Hexahedron{8}
const Hex20 = Hexahedron{20}
const Hex27 = Hexahedron{27}

# src/topology/tetrahedron.jl
struct Tetrahedron{N} <: AbstractTopology end
const Tet4 = Tetrahedron{4}
const Tet10 = Tetrahedron{10}

# Usage in assembly
function compute_element_stiffness!(
    K_blocks::Matrix{Tensor{2,3}},
    X::Vector{Vec{3}},
    material::AbstractMaterial,
    ::Type{Hexahedron{N}}) where {N}
    
    # Extract N for compile-time operations
    dN_dx = ntuple(i -> compute_gradient(i, ...), Val(N))  # Zero allocation!
    
    for k in 1:N, l in 1:N  # Loops unroll for small N
        # ... assembly logic
    end
end
```

### Extending to Other Topologies

To add node count parameter to remaining topologies:

```julia
# Triangle
struct Triangle{N} <: AbstractTopology end
const Tri3 = Triangle{3}    # P1 linear
const Tri6 = Triangle{6}    # P2 quadratic
const Tri10 = Triangle{10}  # P3 cubic

# Quadrilateral
struct Quadrilateral{N} <: AbstractTopology end
const Quad4 = Quadrilateral{4}    # Q1 bilinear
const Quad8 = Quadrilateral{8}    # Q2 serendipity
const Quad9 = Quadrilateral{9}    # Q2 full tensor product
```

## References

- **Performance benchmark:** November 13, 2025 - `ntuple(f, N)` allocates 3.8 MB vs `ntuple(f, Val(N))` allocates 1.4 KB
- **Mesh file formats:** Abaqus .inp, Code Aster .med - explicitly specify node count in connectivity
- **Julia performance:** Type parameters enable loop unrolling and zero-allocation code generation
- **Previous decision:** November 9, 2025 - "Topology without node count" (superseded)

## Related ADRs

- **ADR 001:** Separation of Concerns (Topology/Interpolation/Integration) - Node count is part of topology from mesh
- **ADR 003** (future): Parametric Basis Types (`Lagrange{T, P}` implementation details)
- **ADR 004** (future): Element Composition Type System

## Approval

**Original Proposal:** November 9, 2025 (topology without node count)  
**Revised:** November 13, 2025 (topology WITH node count parameter)  
**Rationale for Change:** Node count comes from mesh, not basis. Type parameter enables zero-allocation performance.  
**Status:** Accepted - Reflects reality of mesh-driven FEM and enables critical performance optimizations  
**Implementation:** Already implemented for `Hexahedron{N}` and `Tetrahedron{N}`, extend to other topologies as needed

---

**Note:** This ADR supersedes the November 9 decision. The key insight is that **mesh connectivity determines node count**, and capturing this in the type system enables zero-allocation assembly via `Val(N)` for compile-time loop unrolling.
