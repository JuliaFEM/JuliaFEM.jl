---
title: "ADR 002: Topology Types Without Hardcoded Node Counts"
description: "Architecture decision to separate topology (geometry) from node count (determined by basis)"
date: "November 9, 2025"
author: "Jukka Aho"
categories: ["architecture", "design-decision", "adr"]
keywords: ["topology", "basis", "lagrange", "separation-of-concerns", "node-count"]
audience: "researchers"
level: "advanced"
type: "adr"
series: "The JuliaFEM Book"
chapter: 3
status: "accepted"
supersedes: "ADR 001 (partial)"
---

## Context

Commercial FEM codes (Code Aster, Abaqus, Nastran) use element type names like `TRIA3`, `TRIA6`, `QUAD4`, `QUAD8`, `HEXA8`, `HEXA20`, `HEXA27`. These names **hardcode the node count** into the type identifier.

This was JuliaFEM's original approach too (inherited from Code Aster heritage):

```julia
struct Tri3 <: AbstractTopology end   # 3-node triangle
struct Tri6 <: AbstractTopology end   # 6-node triangle
struct Quad4 <: AbstractTopology end  # 4-node quadrilateral
struct Quad8 <: AbstractTopology end  # 8-node quad (serendipity)
struct Quad9 <: AbstractTopology end  # 9-node quad (full tensor product)
```

**Problem:** This conflates three orthogonal concepts:

1. **Geometric shape** (triangle, quadrilateral, hexahedron)
2. **Node count** (3, 6, 4, 8, 9, ...)
3. **Implied polynomial degree** (linear, quadratic, cubic)

## Decision

**We remove node counts from topology types.** Topology defines ONLY the geometric shape of the reference element.

### New Design

```julia
# Topology = Pure geometry (NO node count)
abstract type AbstractTopology end

struct Point <: AbstractTopology end
struct Segment <: AbstractTopology end
struct Triangle <: AbstractTopology end
struct Quadrilateral <: AbstractTopology end
struct Tetrahedron <: AbstractTopology end
struct Hexahedron <: AbstractTopology end
struct Pyramid <: AbstractTopology end
struct Wedge <: AbstractTopology end  # Prism

# Node count is DERIVED from basis + topology
abstract type AbstractBasis end
struct Lagrange{T<:AbstractTopology, P} <: AbstractBasis end  # P = polynomial degree

nnodes(::Lagrange{Triangle, 1}) = 3   # P1 → vertices only
nnodes(::Lagrange{Triangle, 2}) = 6   # P2 → vertices + edge midpoints
nnodes(::Lagrange{Triangle, 3}) = 10  # P3 → vertices + edges + interior

nnodes(::Lagrange{Quadrilateral, 1}) = 4   # Q1 → corners
nnodes(::Lagrange{Quadrilateral, 2}) = 9   # Q2 → full tensor product

struct Serendipity{T<:AbstractTopology, P} <: AbstractBasis end
nnodes(::Serendipity{Quadrilateral, 2}) = 8  # Q2 without center node

# Element composition
Element(Triangle(), Lagrange{Triangle, 1}(), Gauss{2}(), (1,2,3))        # 3 nodes
Element(Triangle(), Lagrange{Triangle, 2}(), Gauss{3}(), (1,2,3,4,5,6))  # 6 nodes
Element(Triangle(), Lagrange{Triangle, 3}(), Gauss{4}(), (1,...,10))     # 10 nodes
```

## Rationale

### 1. Mathematical Correctness

In mathematics, there is no "3-node triangle" vs "6-node triangle". There is:

- **Triangle** (the geometric shape)
- **P1 Lagrange interpolation** (implies 3 nodes at vertices)
- **P2 Lagrange interpolation** (implies 6 nodes)

The node count is a **consequence** of choosing a polynomial approximation space over a given topology.

### 2. Separation of Concerns

| Concern | What it defines | Example |
|---------|----------------|---------|
| **Topology** | Geometric shape, parametric domain | `Triangle`, `Hexahedron` |
| **Basis** | Polynomial space + DOF placement | `Lagrange{Triangle, 2}` |
| **Integration** | Numerical quadrature | `Gauss{3}` |

**Old way (conflated):**

- `Tri3` conflates: Triangle + P1 Lagrange + 3 nodes
- `Tri6` conflates: Triangle + P2 Lagrange + 6 nodes
- Cannot use Triangle with Nédélec basis (edge DOFs)
- Cannot use Triangle with hierarchical basis
- Cannot add interior DOFs for pressure

**New way (separated):**

- `Triangle()` = just geometry
- `Lagrange{Triangle, 1}` = P1 interpolation → implies 3 nodes
- `Nedelec{Triangle, 1}` = edge DOFs → still 3 nodes, DOFs on edges
- `Hierarchical{Triangle, P}` = different basis, same topology

### 3. Eliminates Combinatorial Explosion

**Old Code Aster approach:**

```text
TRIA3, TRIA6, TRIA7, TRIA10 (cubic)
QUAD4, QUAD8, QUAD9
TETRA4, TETRA10
HEXA8, HEXA20, HEXA27
PENTA6, PENTA15 (wedge/prism)
PYRA5, PYRA13
```

Each is a **separate type** with duplicated code. Want reduced integration? Add `HEXA8R`. Want hybrid formulation? Add `HEXA8H`. Result: **hundreds of element types**.

**New JuliaFEM approach:**

```julia
# 8 topology types
topologies = [Point, Segment, Triangle, Quadrilateral, 
              Tetrahedron, Hexahedron, Pyramid, Wedge]

# × N basis families
bases = [Lagrange{T, P}, Serendipity{T, P}, Nedelec{T, P}, 
         RaviartThomas{T, P}, Hermite{T, P}, Hierarchical{T, P}, ...]

# × M integration schemes
integrations = [Gauss{N}, Lobatto{N}, Reduced, ...]

# All combinations work automatically via composition!
```

**No code duplication.** One `assemble_element()` function works for all.

### 4. Enables Advanced Element Types

**Edge elements (Nédélec) for electromagnetics:**

```julia
# DOFs are on EDGES, not at nodes!
element = Element(Triangle(), Nedelec{Triangle, 1}(), Gauss{2}(), (1,2,3))
nnodes(element)  # → 3 (geometric connectivity)
nedges(element)  # → 3
ndofs(element)   # → 3 (DOFs on edges, not nodes!)
```

Cannot represent this with `TRIA3` (assumes nodal DOFs).

**Face elements (Raviart-Thomas) for fluids:**

```julia
# DOFs are on FACES
element = Element(Tetrahedron(), RaviartThomas{Tetrahedron, 1}(), Gauss{2}(), 
                  (1,2,3,4))
nnodes(element)  # → 4
nfaces(element)  # → 4
ndofs(element)   # → 4 (DOFs on faces!)
```

**Mixed formulations (Taylor-Hood):**

```julia
# Velocity: Q2 (9 nodes)
# Pressure: Q1 (4 nodes) but with DOFs at subset of velocity nodes
# Or: pressure DOF at element center (not at any node!)
```

### 5. Correctness: Node Count ≠ DOF Count

**Critical insight:** Nodes are for **connectivity** (graph structure). DOFs are for **unknowns** (linear system).

```julia
# Standard nodal element: nodes = DOFs
el = Element(Triangle(), Lagrange{Triangle, 1}(), Gauss{1}(), (1,2,3))
nnodes(el)  # 3
ndofs(el)   # 3 (1 DOF per node for scalar field)

# Edge element: DOFs ≠ nodes
el = Element(Triangle(), Nedelec{Triangle, 1}(), Gauss{2}(), (1,2,3))
nnodes(el)  # 3 (geometric nodes for connectivity)
ndofs(el)   # 3 (but DOFs are on edges, not nodes!)

# Mixed element: Multiple fields
el = Element(Quadrilateral(), TaylorHood{Quadrilateral}(), Gauss{3}(), (...))
nnodes(el)       # Depends on formulation
ndofs(el, :velocity)  # Q2 → 9 DOFs
ndofs(el, :pressure)  # Q1 → 4 DOFs (or 1 at center)
```

## Consequences

### Positive

✅ **One topology type per geometric shape** (8 types total, not hundreds)  
✅ **Mathematically correct** (topology = geometry, not interpolation)  
✅ **Extensible** (add Nédélec, Raviart-Thomas, Hermite, ... without new topologies)  
✅ **No code duplication** (one assembly function for all)  
✅ **Type system enforces correctness** (basis must match topology)  
✅ **Educational** (code teaches FEM mathematics properly)

### Negative

⚠️ **Breaking change** from JuliaFEM v0.5.1 (but necessary for correctness)  
⚠️ **More complex type signatures** (`Element{Triangle, Lagrange{Triangle,1}, Gauss{2}, 3}`)  
⚠️ **Requires understanding separation of concerns** (topology ≠ basis)  
⚠️ **Migration needed for old code** (provide adapters and deprecation warnings)

### Neutral

⚡ **Type parameter becomes longer** but compile-time specialization still works  
⚡ **Need convenience constructors** for common cases  
⚡ **Documentation must be excellent** (this ADR is part of that!)

## Implementation Notes

### Migration Strategy

1. **Phase 1** (Current): Keep old `Tri3`, `Quad4`, etc. as topology types for compatibility
2. **Phase 1B** (Next):
   - Rename topology files: `tri3.jl` → `triangle.jl`
   - Create `struct Triangle <: AbstractTopology end`
   - Keep `Tri3 = Triangle` as alias
3. **Phase 2**:
   - Implement `Lagrange{T, P}` parametric basis
   - Map old constructors: `Element(Tri3, ...)` → `Element(Triangle(), Lagrange{Triangle,1}(), ...)`
4. **Phase 3**: Deprecate old names, migration guide

### Backwards Compatibility Shims

```julia
# Type aliases for transition
const Tri3 = Triangle
const Tri6 = Triangle  # Wait, this doesn't make sense anymore!

# Better: Basis aliases
const TRIA3 = Lagrange{Triangle, 1}
const TRIA6 = Lagrange{Triangle, 2}
const QUAD4 = Lagrange{Quadrilateral, 1}
const QUAD8 = Serendipity{Quadrilateral, 2}
const QUAD9 = Lagrange{Quadrilateral, 2}

# Constructor adapter
function Element(::Type{Tri3}, connectivity::NTuple{3, Int})
    @warn "Element(Tri3, ...) is deprecated, use Element(Triangle(), Lagrange{Triangle,1}(), ...)"
    Element(Triangle(), Lagrange{Triangle, 1}(), Gauss{2}(), connectivity)
end
```

### File Organization

```text
src/topology/
  point.jl           # struct Point <: AbstractTopology end
  segment.jl         # struct Segment <: AbstractTopology end
  triangle.jl        # struct Triangle <: AbstractTopology end (not tri3.jl!)
  quadrilateral.jl   # struct Quadrilateral <: AbstractTopology end
  tetrahedron.jl     # struct Tetrahedron <: AbstractTopology end
  hexahedron.jl      # struct Hexahedron <: AbstractTopology end
  pyramid.jl         # struct Pyramid <: AbstractTopology end
  wedge.jl           # struct Wedge <: AbstractTopology end

src/basis/
  lagrange.jl        # Lagrange{T, P} implementation
  serendipity.jl     # Serendipity{T, P}
  nedelec.jl         # Nedelec{T, P}
  raviart_thomas.jl  # RaviartThomas{T, P}
  # ... other basis families
```

## References

- **Mathematics:** Ciarlet, P.G. (1978). *The Finite Element Method for Elliptic Problems*. Chapter on finite element spaces.
- **Edge elements:** Nédélec, J.C. (1980). "Mixed finite elements in R³". *Numerische Mathematik*.
- **Code Aster documentation:** Examples of `TRIA3`, `TRIA6` naming (the anti-pattern we're fixing).
- **JuliaFEM v0.5.1:** Previous implementation with hardcoded node counts.

## Related ADRs

- **ADR 001:** Separation of Concerns (Topology/Interpolation/Integration) - This ADR refines the topology part
- **ADR 003** (future): Parametric Basis Types (`Lagrange{T, P}` implementation details)
- **ADR 004** (future): Element Composition Type System

## Approval

**Proposed by:** Jukka Aho  
**Discussed:** November 9, 2025 (AI-assisted design review session)  
**Status:** Accepted - This is the correct mathematical and architectural approach  
**Implementation:** Phase 1B (immediate next step after current topology/integration work)

---

**Note:** This ADR represents a fundamental insight that corrects a decades-old industry anti-pattern inherited from early FEM codes. The mathematical correctness and extensibility benefits far outweigh the migration costs.
