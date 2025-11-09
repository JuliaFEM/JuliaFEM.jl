---
title: "Element Architecture: Separation of Concerns"
description: "Understanding finite elements as composition of orthogonal concerns: topology, interpolation, integration, and fields"
date: "November 9, 2025"
author: "Jukka Aho"
categories: ["architecture", "theory", "design"]
keywords: ["element", "topology", "interpolation", "integration", "basis functions", "separation of concerns", "composition"]
audience: "researchers"
level: "intermediate"
type: "theory"
series: "The JuliaFEM Book"
chapter: 2
status: "draft"
---

## Introduction

What is a finite element? This seemingly simple question has profound implications for software architecture, performance, and maintainability. Most FEM codes conflate multiple concerns into monolithic "element types," leading to combinatorial explosion and code duplication. This chapter presents JuliaFEM's approach: **elements as composition of orthogonal concerns**.

## The Four Orthogonal Concerns

A finite element is fundamentally composed of **four independent concerns**:

### 1. Topology (Connectivity/Graph Theory)

**What it is:** The combinatorial structure of how nodes connect to form an element.

- **Examples:** `Tri3` (3-node triangle), `Quad4` (4-node quadrilateral), `Tet10` (10-node tetrahedron)
- **Properties:** Number of nodes, edges, faces; reference element geometry
- **Mathematics:** Graph theory, combinatorics
- **Rarely changes:** Topology is a mathematical object, not implementation-dependent

**Reference element:** The element in parametric coordinates $\xi \in [-1, 1]^d$

```text
Tri3 reference element:
     η
     ^
     |
  (0,1)
     |  \
     |    \
     |      \
     +---------> ξ
  (0,0)    (1,0)
```

### 2. Interpolation (Basis Functions)

**What it is:** How to interpolate field values between nodes.

- **Examples:** Lagrange polynomials, hierarchical polynomials, NURBS
- **Properties:** Polynomial order, continuity, partition of unity
- **Mathematics:** Approximation theory, functional analysis
- **Can vary:** Same topology with different interpolation schemes

**Interpolation formula:** $u(\xi) = \sum_{i=1}^n N_i(\xi) u_i$

where $N_i(\xi)$ are basis functions and $u_i$ are nodal values.

**Key property:** Basis functions are **independent of topology** (mostly).

- Linear Lagrange on `Tri3`: $N_1 = 1 - \xi - \eta$, $N_2 = \xi$, $N_3 = \eta$
- Linear Lagrange on `Quad4`: $N_1 = (1-\xi)(1-\eta)/4$, ...
- Hierarchical on `Tri3`: $N_1 = 1 - \xi - \eta$, $N_2 = \xi(1-\xi-\eta)$, ...

**Important:** Interpolation scheme determines polynomial order, NOT topology.

### 3. Integration (Quadrature Rules)

**What it is:** How to numerically integrate over the element.

- **Examples:** Gauss-Legendre, Gauss-Lobatto, reduced integration
- **Properties:** Number of points, weights, accuracy order
- **Mathematics:** Numerical integration theory
- **Can vary:** Full vs. reduced integration, different orders

**Integration formula:** $\int_\Omega f \, dV \approx \sum_{i=1}^{n_q} w_i f(\xi_i) |J(\xi_i)|$

where $w_i$ are quadrature weights, $\xi_i$ are integration points, and $|J|$ is the Jacobian determinant.

**Key property:** Integration scheme is **independent of topology and interpolation** (mostly).

- Full integration: Enough points to integrate exactly
- Reduced integration: Fewer points (e.g., for locking prevention)
- Selective integration: Different rules for different terms

### 4. Fields (Data)

**What it is:** The variables/data stored on the element.

- **Examples:** Displacement, temperature, pressure, velocity
- **Properties:** Scalar/vector/tensor, time-dependent or not
- **Mathematics:** Depends on the PDE being solved
- **Problem-dependent:** Elasticity has displacement, heat has temperature

**Field storage:** Each element stores values at nodes or integration points.

```julia
fields = Dict(
    :displacement => [u1, u2, u3],  # Nodal values
    :temperature => [T1, T2, T3],
    :stress => [σ1, σ2, σ3, σ4]     # Integration point values
)
```

**Key property:** Fields are **completely independent** of topology, interpolation, and integration.

## The Anti-Pattern: Abaqus's Mistake

Abaqus (and many commercial codes) conflate these concerns, leading to **combinatorial explosion**:

### Hexahedral Element Examples

| Element Type | Topology | Interpolation | Integration | Modes |
|--------------|----------|---------------|-------------|-------|
| `C3D8`       | Hex8     | Linear        | Full (2×2×2)| None  |
| `C3D8R`      | Hex8     | Linear        | Reduced (1) | None  |
| `C3D8I`      | Hex8     | Linear        | Full (2×2×2)| Incompatible |
| `C3D20`      | Hex20    | Quadratic     | Full (3×3×3)| None  |
| `C3D20R`     | Hex20    | Quadratic     | Reduced (2×2×2) | None |
| `C3D20RH`    | Hex20    | Quadratic     | Reduced     | Hybrid |
| `C3D27`      | Hex27    | Quadratic     | Full (3×3×3)| None  |
| `C3D27R`     | Hex27    | Quadratic     | Reduced     | None  |

**Result:** 8 different "element types" for what is fundamentally **one topology** with different choices for interpolation and integration!

### The Problem with Conflation

```c
// Abaqus-style (pseudo-code)
class C3D8 {
    // Everything mixed together
    Node nodes[8];
    void stiffness_matrix() {
        // Hardcoded: 8 nodes, linear shape functions, 2×2×2 Gauss
    }
};

class C3D8R {
    // Almost identical code, but different integration
    Node nodes[8];
    void stiffness_matrix() {
        // Hardcoded: 8 nodes, linear shape functions, 1 point
    }
};

// Now need C3D8I, C3D20, C3D20R, ... → code duplication nightmare
```

**Issues:**

- ❌ Code duplication (each element type reimplements similar logic)
- ❌ Combinatorial explosion (n topologies × m interpolations × k integrations)
- ❌ Maintenance nightmare (bug fix must be repeated in all variants)
- ❌ Cannot mix-and-match (user stuck with pre-defined combinations)
- ❌ No compile-time optimization (runtime dispatch on element type)

## JuliaFEM's Approach: Composition Over Conflation

### Separation of Concerns

```julia
# 1. Define topology (reference element)
abstract type AbstractTopology end
struct Tri3 <: AbstractTopology 
    nnodes::Int = 3
    dim::Int = 2
end
struct Quad4 <: AbstractTopology 
    nnodes::Int = 4
    dim::Int = 2
end
struct Hex8 <: AbstractTopology 
    nnodes::Int = 8
    dim::Int = 3
end

# 2. Define interpolation schemes
abstract type AbstractBasis end
struct Lagrange{P} <: AbstractBasis end  # P = polynomial order
struct Hierarchical{P} <: AbstractBasis end
struct NURBS{P} <: AbstractBasis end

# 3. Define integration rules
abstract type AbstractIntegration end
struct Gauss{N} <: AbstractIntegration end  # N = number of points
struct Lobatto{N} <: AbstractIntegration end
struct Reduced <: AbstractIntegration end

# 4. Element composes all three
struct Element{T <: AbstractTopology, B <: AbstractBasis, I <: AbstractIntegration, N}
    topology::T
    basis::B
    integration::I
    connectivity::NTuple{N, UInt}
    fields::Dict{Symbol, Any}  # TODO: Type-stable structure
end
```

### User-Facing API

```julia
# Create element by composing concerns
topology = Tri3()
basis = Lagrange{1}()       # Linear interpolation
integration = Gauss{3}()    # 3-point Gauss quadrature

element = Element(topology, basis, integration, 
                  connectivity=(1, 2, 3))

# Type-stable construction (preferred)
element = Element{Tri3, Lagrange{1}, Gauss{3}}(...)
```

**Benefits:**

- ✅ Mix-and-match any combination
- ✅ Type system enforces compatibility
- ✅ Compiler generates specialized code for each combination
- ✅ Zero runtime overhead (types disappear after compilation)

### Directory Structure

```text
src/
  topology/
    tri3.jl         # Reference triangle
    quad4.jl        # Reference quadrilateral
    tet10.jl        # Reference tetrahedron
    hex8.jl         # Reference hexahedron
    ...
  
  basis/
    lagrange.jl              # Lagrange polynomial bases
    lagrange_generated.jl    # Pre-generated for compile-time
    hierarchical.jl          # Hierarchical/p-refinement
    nurbs.jl                 # NURBS for isogeometric
    ...
  
  integration/
    gauss.jl        # Gauss-Legendre quadrature
    lobatto.jl      # Gauss-Lobatto quadrature
    reduced.jl      # Reduced integration
    ...
  
  elements/
    element.jl      # Element type definition
    integrate.jl    # Integration loop
    assemble.jl     # Global assembly
    ...
```

**Rationale:**

- Each concern in its own directory
- Clear separation of mathematical concepts
- Easy to find and modify code
- Natural place for new additions (new topology? → `topology/`)

## Mathematical Formulation

### Element Stiffness Matrix

The element stiffness matrix is computed by integrating over the element domain:

$$K^e_{ij} = \int_{\Omega_e} B_i^T D B_j \, dV$$

where:

- $B_i$ = strain-displacement matrix for node $i$ (depends on **basis derivatives**)
- $D$ = material constitutive matrix
- $\Omega_e$ = element domain

### Separation in Implementation

```julia
function element_stiffness(element::Element{T, B, I}) where {T, B, I}
    K = zeros(nnodes(T) * ndofs, nnodes(T) * ndofs)
    
    # Get integration points from integration scheme
    ips = integration_points(element.integration, element.topology)
    
    for ip in ips
        # Evaluate basis functions (depends on basis scheme)
        N = evaluate_basis(element.basis, ip.ξ)
        dN = evaluate_basis_derivatives(element.basis, ip.ξ)
        
        # Jacobian (depends on topology + node coordinates)
        J = jacobian(element.topology, element.connectivity, dN)
        
        # Strain-displacement matrix (depends on basis derivatives)
        B = strain_displacement_matrix(dN, J)
        
        # Integrate using quadrature weight
        K += ip.weight * B' * D * B * det(J)
    end
    
    return K
end
```

**Notice:** Each concern is accessed through clean interfaces:

- `integration_points()` → integration scheme
- `evaluate_basis()` → interpolation scheme
- `jacobian()` → topology + connectivity

### Type-Stability for Performance

With concrete types, the compiler can specialize:

```julia
# This becomes a specialized function with no runtime overhead
function element_stiffness(
    element::Element{Tri3, Lagrange{1}, Gauss{3}, 3}
)
    # Compiler knows at compile time:
    # - 3 nodes (Tri3)
    # - 3 basis functions (Lagrange{1})
    # - 3 integration points (Gauss{3})
    # - connectivity is NTuple{3, UInt}
    
    # Generated code has:
    # - No branches
    # - No allocations
    # - Vectorized loops
    # - Inlined function calls
end
```

**Performance benefit:** 100× speedup compared to runtime dispatch!

## Extending the System

### Adding a New Topology

```julia
# File: src/topology/hex27.jl
struct Hex27 <: AbstractTopology
    nnodes::Int = 27
    dim::Int = 3
end

# Reference element coordinates
reference_coordinates(::Hex27) = [
    # 8 corner nodes
    (-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
    (-1, -1,  1), (1, -1,  1), (1, 1,  1), (-1, 1,  1),
    # 12 mid-edge nodes
    (0, -1, -1), (1, 0, -1), (0, 1, -1), (-1, 0, -1),
    # ... (continue for all 27 nodes)
]

# Topology is a pure mathematical object
# No need to implement assembly, integration, etc.
```

**Usage:**

```julia
element = Element{Hex27, Lagrange{2}, Gauss{3}}(...)
# Automatically works with existing assembly code!
```

### Adding a New Interpolation Scheme

```julia
# File: src/basis/hierarchical.jl
struct Hierarchical{P} <: AbstractBasis end

# Evaluate basis functions
function evaluate_basis(basis::Hierarchical{P}, ξ::Vec) where P
    # Implement hierarchical polynomial evaluation
    # Return NTuple{N, Float64}
end

# Evaluate basis derivatives
function evaluate_basis_derivatives(basis::Hierarchical{P}, ξ::Vec) where P
    # Implement derivatives
    # Return NTuple{N, Vec}
end
```

**Usage:**

```julia
element = Element{Tri3, Hierarchical{3}, Gauss{4}}(...)
# Use same Tri3 topology with hierarchical basis!
```

### Adding a New Integration Rule

```julia
# File: src/integration/lobatto.jl
struct Lobatto{N} <: AbstractIntegration end

function integration_points(::Lobatto{N}, topology::T) where {N, T <: AbstractTopology}
    # Return integration points and weights for Lobatto quadrature
    # Specific to topology dimension
end
```

**Usage:**

```julia
element = Element{Quad4, Lagrange{1}, Lobatto{3}}(...)
# Use Lobatto instead of Gauss for same element!
```

## Compile-Time Guarantees

### Type System Enforcement

The type system prevents invalid combinations:

```julia
# ✅ Valid: Tri3 with 2D basis
element = Element{Tri3, Lagrange{1}, Gauss{3}}(...)

# ❌ Compile error: Cannot use 3D topology with 2D basis (if we enforce)
element = Element{Hex8, TriangularBasis, Gauss{3}}(...)

# ✅ Valid: Mix different integration rules
element1 = Element{Quad4, Lagrange{1}, Gauss{4}}(...)     # Full integration
element2 = Element{Quad4, Lagrange{1}, Reduced}(...)       # Reduced integration
element3 = Element{Quad4, Lagrange{2}, Gauss{9}}(...)     # Quadratic + more points
```

### Number of Nodes Known at Compile Time

```julia
# Connectivity is NTuple{N, UInt} where N is known at compile time
struct Element{T, B, I, N}
    topology::T
    basis::B
    integration::I
    connectivity::NTuple{N, UInt}  # N from topology
end

# Compiler can unroll loops over connectivity
for i in 1:length(element.connectivity)
    # Loop is unrolled at compile time!
end
```

**Result:** Zero-overhead abstractions, same performance as hand-written code.

## Backward Compatibility

### Type Aliases for Old Code

```julia
# Old code used to write:
# element = Element("Tri3", ...)

# Provide type aliases:
const Tri3Element = Element{Tri3, Lagrange{1}, Gauss{3}}
const Quad4Element = Element{Quad4, Lagrange{1}, Gauss{4}}

# Old code still works:
element = Tri3Element(connectivity=(1,2,3))
```

### Constructor Convenience

```julia
# Convenience constructors for common cases
function Element(::Type{Tri3}, connectivity::NTuple{3, UInt})
    Element{Tri3, Lagrange{1}, Gauss{3}}(
        Tri3(), Lagrange{1}(), Gauss{3}(), connectivity, Dict()
    )
end

# User can still write simple code:
element = Element(Tri3, (1, 2, 3))
```

## Performance Implications

### From Roadmap to HPC

This architectural decision directly supports the five performance principles:

1. **Type Stability** ✅
   - All element types are concrete
   - No runtime dispatch in hot paths
   - Compiler can optimize aggressively

2. **Zero Allocations** ✅
   - `NTuple{N}` for connectivity → stack allocated
   - Integration points known at compile time → no allocation
   - Basis evaluation can return tuples → no Vector allocation

3. **Specialization** ✅
   - Compiler generates optimized code for each `Element{T, B, I}`
   - No generic "one size fits all" slow path
   - Each combination gets its own fast implementation

4. **Parallelism** ✅
   - Element independence enables parallel assembly
   - Topology separation enables graph-based partitioning
   - No shared state between elements

5. **GPU Portability** ✅
   - Each concern can be ported to GPU independently
   - Small, focused kernels (evaluate basis, integrate, assemble)
   - Type-stable code → CUDA.jl can compile it

### Measured Impact

From benchmarks (see `docs/book/benchmarks/`):

- **Before (Dict-based, runtime dispatch):** 15 μs per element
- **After (type-stable composition):** 150 ns per element
- **Speedup:** 100× faster!

## Comparison with Other Libraries

### Gridap.jl

Gridap uses a similar separation but with different emphasis:

- Focus on general PDEs, not specifically FEM
- More abstract (CellField, FESpace concepts)
- Great for research, steeper learning curve

**JuliaFEM approach:** More explicit, educational focus.

### Ferrite.jl

Ferrite keeps element types somewhat mixed:

- Element types include both topology and interpolation
- Less flexible mixing-and-matching
- But simpler mental model for beginners

**JuliaFEM approach:** More flexible, better for advanced users.

### Deal.II (C++)

Deal.II has sophisticated separation:

- Template-based (C++ templates)
- Very fast, but complex compilation
- Steep learning curve

**JuliaFEM approach:** Julia's type system gives similar power without template complexity.

## Lessons Learned

### What Works

✅ **Separation of concerns is worth it**

- Initial overhead pays off in maintainability
- Performance benefits are real (100× speedup)
- Users appreciate flexibility

✅ **Type system enforcement is powerful**

- Catch errors at compile time, not runtime
- Compiler optimizations are dramatic
- Zero-cost abstractions are achievable

✅ **Documentation must explain WHY**

- Show the Abaqus anti-pattern
- Explain the mathematics
- Provide migration path for old code

### What's Hard

⚠️ **Forward declarations in Julia**

- No forward declarations → careful include order
- See `llm/INCLUDE_ORDER_EXAMPLES.md` for solutions

⚠️ **Balance between flexibility and simplicity**

- Too flexible → confusing for beginners
- Too simple → limiting for advanced users
- Solution: Convenience constructors + type aliases

⚠️ **Backward compatibility**

- Old code expects different API
- Need adapters and deprecation warnings
- Migration guide essential

## Conclusion

**Element** = Topology + Interpolation + Integration + Fields

This simple equation guides JuliaFEM's architecture:

1. **Topology** defines the reference element (mathematical object)
2. **Interpolation** defines how to interpolate (approximation theory)
3. **Integration** defines how to integrate (numerical analysis)
4. **Fields** define what data lives on the element (problem-specific)

**Benefits:**

- ✅ Clear separation of concerns
- ✅ Mix-and-match flexibility
- ✅ Type system enforcement
- ✅ 100× performance improvement
- ✅ Maintainable, extensible codebase

**Trade-off:**

- ⚠️ More complex initial setup
- ⚠️ Requires understanding of Julia's type system
- ⚠️ Documentation must be excellent

**Result:** A modern, high-performance, extensible FEM library that teaches good software engineering alongside finite element methods.

---

## Further Reading

- `llm/ARCHITECTURE.md` - Full architecture document
- `docs/book/roadmap_to_hpc.md` - Performance philosophy
- `docs/book/lagrange_basis_functions.md` - Lagrange interpolation theory
- `docs/contributor/testing_philosophy.md` - How we test this design

## References

1. Hughes, T.J.R. (2000). *The Finite Element Method: Linear Static and Dynamic Finite Element Analysis*. Dover. (Classic FEM reference)
2. Wriggers, P. (2006). *Computational Contact Mechanics*. Springer. (Contact mechanics focus)
3. Abaqus Documentation. (Example of element type proliferation)
4. Gridap.jl Documentation. (Alternative approach to FEM in Julia)
5. Ferrite.jl Documentation. (Another Julia FEM library)
6. Deal.II Documentation. (C++ FEM library with similar separation)
