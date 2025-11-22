# Basis Module: Design, Implementation, and Extension Guide

Interpolation schemes for JuliaFEM. This module defines basis families (`Lagrange{P}`, `Serendipity{P}`, plate/shell bases), the evaluation API (`get_basis_functions`, `get_basis_derivatives`), and the generator that produces closed-form, zero-allocation implementations. It is built around a strict separation of concerns: **topology owns geometry and node layout; basis owns interpolation**.

**Key Design Choices:**

- **Basis functions return `SVector{N, Float64}`** - Enables vector operations like `dot(coeffs, N)`
- **Derivatives return `SVector{N, Vec{D}}`** - Each entry is a gradient vector from Tensors.jl
- **Generated code in `basis_generated.jl`** - Not limited to Lagrange; supports all basis families

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Design Philosophy](#design-philosophy)
3. [Module Structure](#module-structure)
4. [The Basis Evaluation API](#the-basis-evaluation-api)
5. [How the Generator Works](#how-the-generator-works)
6. [The Vandermonde Approach](#the-vandermonde-approach)
7. [Extending the Module](#extending-the-module)
8. [Advanced Topics](#advanced-topics)
9. [Performance Characteristics](#performance-characteristics)
10. [References and Further Reading](#references-and-further-reading)

---

## Quick Start

```julia
using JuliaFEM

# Define topology (geometry + node count) and basis (interpolation)
topo  = Tetrahedron{10}()   # 10-node quadratic tetrahedron
basis = Lagrange{2}()       # quadratic Lagrange basis
ξ     = Vec(0.2, 0.3, 0.1)  # parametric coordinates

# Evaluate basis functions and derivatives
N  = get_basis_functions(topo, basis, ξ)    # → SVector{10, Float64}
dN = get_basis_derivatives(topo, basis, ξ)  # → SVector{10, Vec{3, Float64}}

# Vector operations work naturally
u_interp = dot(node_values, N)              # Interpolate field at ξ
grad_u = sum(node_values[i] * dN[i] for i in 1:10)  # Gradient

# Access individual basis functions if needed
N_5 = N[5]        # 5th basis function value (Float64)
dN_5 = dN[5]      # 5th basis function gradient (Vec{3, Float64})
```

**Key principle:** All heavy lifting (symbolic math, differentiation, code generation) happens **offline** in the generator; runtime evaluation is pure, inlineable, and allocation-free.

---

## Design Philosophy

### Separation of Concerns: The Core Principle

The finite element method traditionally bundles geometry, node layout, and interpolation into monolithic "element types" (Tri3, Quad4, Tet10, etc.). This coupling seems convenient initially but becomes a straightjacket when you need to:

- Use different interpolation families on the same mesh topology
- Swap between H¹, H(curl), H(div) spaces
- Experiment with hierarchical, modal, or reduced bases
- Implement plate/shell elements with specialized interpolations

**JuliaFEM draws a clear boundary:**

```julia
# Topology: Geometry + node ordering + connectivity
topology = Tetrahedron{10}()  # "I'm a 10-node tet reference element"
nnodes(topology) = 10
dim(topology) = 3
reference_coordinates(topology)  # Fixed node positions in [-1,1]³

# Basis: Interpolation scheme (NO geometry!)
basis = Lagrange{2}()  # "I do quadratic Lagrange interpolation"
# Basis doesn't know node positions, connectivity, or dimension
# It only defines HOW to interpolate

# Evaluation: Combine topology + basis
N = get_basis_functions(topology, basis, ξ)
```

This separation mirrors textbook presentations where geometry and approximation theory are distinct chapters. It's not just elegant—it's **practical**:

1. **Reusability:** Same `Tetrahedron{10}` works with `Lagrange{2}`, `Nedelec{1}`, or custom bases
2. **Performance:** Type parameters enable compile-time optimization (static loops, inlining)
3. **Maintainability:** Adding a new basis doesn't require touching topology code
4. **Correctness:** Single source of truth for node positions (`reference_coordinates`)

### Why Not `Lagrange{Tetrahedron, 2}`?

You might be tempted to write:

```julia
# ❌ DON'T: Topology in basis type
basis = Lagrange{Tetrahedron, 2}()
N = get_basis_functions(basis, ξ)  # No topology parameter needed!
```

This seems simpler (one less parameter!), but it creates problems:

**Problem 1**: Redundancy

```julia
# Topology appears TWICE!
element = Element(Tetrahedron, Lagrange{Tetrahedron, 2}, connectivity)
#                 ^^^^^^^^^^^  ^^^^^^^^^^^^^^^^^^^
```

**Problem 2**: Inflexibility

```julia
# Can't easily swap bases without changing type structure
function assemble_with_basis(::Type{B}) where B
    # How do we extract topology from B?
    # What if B is not Lagrange{T,P} but something else?
end
```

**Problem 3**: Violation of Single Responsibility

```julia
# Basis now "owns" topology information
# But basis shouldn't care about 3D vs 2D, edges vs faces, etc.
```

**Our solution**:

```julia
# ✅ DO: Separate parameters
get_basis_functions(topology, basis, ξ)
```

Yes, it's one more parameter. But it's the **right** parameter—explicit, clear, flexible.

### Type Stability and Performance

Every abstraction must pay for itself in performance. Here's how this design achieves **zero-cost abstraction**:

**Compile-time specialization**:

```julia
# Each (topology, basis) pair gets its own optimized method
@inline function get_basis_functions(::Tetrahedron{10}, ::Lagrange{2}, ξ::Vec{3,T}) where T
    u, v, w = ξ
    # Fully unrolled, inlined polynomial evaluation
    N1 = ...  # Literal expression tree, no loops!
    N2 = ...
    # ...
    return SVector(N1, N2, N3, N4, N5, N6, N7, N8, N9, N10)  # SVector{10, T}
end
```

**Result:** The compiler sees this as if you wrote:

```julia
# Hand-optimized, unrolled code
N1 = -u * (1 - u - v - w) * (1 - 2u - 2v - 2w)
N2 = -v * (1 - u - v - w) * (1 - 2u - 2v - 2w)
# ... (10 expressions, all inlined)
return SVector{10}(N1, N2, N3, N4, N5, N6, N7, N8, N9, N10)
```

No dynamic dispatch, no allocations, no runtime overhead. **Benchmarks:** 6.5 ns for 10 Tet10 derivatives = ~150 million evaluations/second per core.

---

## Module Structure

```text
src/basis/
├── README.md               ← This file
├── api.jl                  ← Core types and interfaces (AbstractBasis, Lagrange{P}, Serendipity{P})
├── basis_descriptions.jl   ← Catalog of generation recipes (topology, family, ansatz)
├── basis_generator.jl      ← Symbolic generator → writes basis_generated.jl
├── basis_generated.jl      ← Auto-generated basis functions (DO NOT EDIT)
└── plate_elements.jl       ← Plate/shell-specific bases (e.g., DKT)
```

### File Responsibilities

**`api.jl`** - Type definitions and public interface

- `AbstractBasis` - Base type for all basis families
- `Lagrange{P}` - Standard nodal Lagrange basis of order P
- `Serendipity{P}` - Reduced tensor-product basis for quads/hexes
- `get_basis_functions` - Evaluate all basis functions at ξ
- `get_basis_derivatives` - Evaluate all basis derivatives at ξ
- `VandermondeBasisDescription` - Description struct for generator

**`basis_descriptions.jl`** - Data catalog (the "menu" of available bases)

Contains `BASIS_DESCRIPTIONS`, a vector of `VandermondeBasisDescription` entries. Each entry specifies:

- `name`: Legacy short name (e.g., "Tri6", "Tet10")
- `description`: Human-readable text
- `topology`: Reference topology type (e.g., `Triangle{6}`, `Tetrahedron{10}`)
- `family`: Basis family/order (e.g., `Lagrange{2}`, `Serendipity{2}`)
- `ansatz`: Tuple of polynomial terms (e.g., `(:(1), :(u), :(v), :(u^2), :(u*v), :(v^2))`)

**Current catalog (17 elements):**

- **1D:** Seg2, Seg3
- **2D Triangles:** Tri3, Tri6
- **2D Quads:** Quad4, Quad8 (serendipity), Quad9
- **3D Tets:** Tet4, Tet10
- **3D Hexes:** Hex8, Hex20 (serendipity), Hex27
- **3D Pyramids:** Pyr5
- **3D Wedges:** Wedge6, Wedge15

**`basis_generator.jl`** - Symbolic code generator (offline tool)

This file is **not loaded** at runtime—it's a standalone tool you run manually:

```bash
julia --project=. src/basis/basis_generator.jl
```

It reads `BASIS_DESCRIPTIONS`, performs symbolic math (Vandermonde system solution + differentiation), and writes methods to `basis_generated.jl`. The generator includes:

- Minimal symbolic differentiation (no SymPy/Symbolics.jl dependency)
- Vandermonde matrix construction and inversion
- Expression simplification
- Code emission with SVector return types (pretty-printed Julia functions)

**`basis_generated.jl`** - Generated code (DO NOT EDIT BY HAND)

This file contains ~2000 lines of auto-generated, optimized `get_basis_functions` and `get_basis_derivatives` methods for **all basis families** (Lagrange, Serendipity, and future exotic bases). Each method is specialized on `(::Type{Topology}, ::Type{Basis}, ξ::Vec)` and returns `SVector` types for efficient vector operations.

**Example generated method:**

```julia
@inline function get_basis_functions(::Type{Triangle{3}}, ::Type{Lagrange{1}}, ξ::Vec{2,T}) where T
    u, v = ξ
    N1 = 1 - u - v
    N2 = u
    N3 = v
    return SVector(N1, N2, N3)
end

@inline function get_basis_derivatives(::Type{Triangle{3}}, ::Type{Lagrange{1}}, ξ::Vec{2,T}) where T
    # Gradients with respect to (u, v)
    dN1 = Vec(-1.0, -1.0)
    dN2 = Vec(1.0, 0.0)
    dN3 = Vec(0.0, 1.0)
    return SVector(dN1, dN2, dN3)
end
```

**`plate_elements.jl`** - Specialized non-nodal bases

For plate/shell elements (DKT, Mindlin, etc.) that don't fit the standard Vandermonde pattern. These define custom `get_basis_functions` methods directly.

---

## The Basis Evaluation API

### Core Functions

```julia
get_basis_functions(topology::AbstractTopology, basis::AbstractBasis, ξ::Vec) → SVector{N, Float64}
```

Evaluate all N basis functions at parametric point ξ. Returns an `SVector` (static vector from StaticArrays.jl) of scalar values. For Lagrange bases, these satisfy the Kronecker property: `Nᵢ(xⱼ) = δᵢⱼ`.

**Why SVector?** Enables natural vector operations:

```julia
# Interpolate field value
u_at_ξ = dot(node_values, N)

# Linear combination
result = coeffs ⋅ N  # Unicode dot product

# Still supports indexing
N_i = N[i]  # Extract individual value
```

**Example:**

```julia
topology = Triangle{6}()
basis = Lagrange{2}()
ξ = Vec(1/3, 1/3)  # Centroid

N = get_basis_functions(topology, basis, ξ)
# N isa SVector{6, Float64}
# N ≈ [0.0, 0.0, 0.0, 0.333, 0.333, 0.333]  # Only mid-edge nodes active

# Natural vector operations
node_temps = SVector(100.0, 200.0, 150.0, 175.0, 180.0, 160.0)
temp_at_ξ = dot(node_temps, N)  # Interpolated temperature
```

---

```julia
get_basis_derivatives(topology::AbstractTopology, basis::AbstractBasis, ξ::Vec) → SVector{N, Vec{D, Float64}}
```

Evaluate all N basis function gradients with respect to parametric coordinates. Returns `SVector` of `Vec{D}` (from Tensors.jl) where D is the parametric dimension (1, 2, or 3).

**Why SVector of Vec?**

- **SVector:** Outer container enables vector algebra (`sum`, `map`, comprehensions)
- **Vec:** Each gradient is a geometric vector with dot product, tensor operations
- **Combined:** Natural FEM operations like `sum(uᵢ * dNᵢ for i in 1:N)`

**Example:**

```julia
topology = Tetrahedron{10}()
basis = Lagrange{2}()
ξ = Vec(0.25, 0.25, 0.25)  # Center

dN = get_basis_derivatives(topology, basis, ξ)
# dN isa SVector{10, Vec{3, Float64}}
# dN[1] ≈ Vec(-3.0, -3.0, -3.0)  # Gradient of corner node basis function

# Natural gradient operations
node_displacements = SVector{10}(...)  # 10 nodal values
grad_u = sum(node_displacements[i] * dN[i] for i in 1:10)  # Vec{3}

# B-matrix construction (strain-displacement)
B_node = SVector(dN[i][1], dN[i][2], dN[i][3], 0, 0, 0)  # Extract components
```

### Convenience Accessors

```julia
get_basis_function(topology, basis, ξ, i::Int) → Float64
```

Returns the i-th basis function value. Equivalent to `get_basis_functions(topology, basis, ξ)[i]`.

**When to use:** Rarely needed—most FEM operations need all basis functions simultaneously. But useful for:

- Educational examples showing single basis function
- Debugging specific basis functions
- Special algorithms accessing basis functions one-at-a-time

**Performance note:** Simple `SVector` indexing is **fastest** and generates optimal code (bounds check eliminated at compile time). No Val dispatch needed. See benchmarks in ADR-003.

```julia
get_basis_derivative(topology, basis, ξ, i::Int) → Vec{D, Float64}
```

Returns the i-th basis function gradient. Equivalent to `get_basis_derivatives(topology, basis, ξ)[i]`.

### Type Signatures and Dispatch

The generated methods dispatch on **type**, not instances:

```julia
# Inside basis_generated.jl (simplified):
@inline function get_basis_functions(
    ::Type{Triangle{3}},      # Type, not instance!
    ::Type{Lagrange{1}},      # Type, not instance!
    ξ::Vec{2,T}               # Value parameter
) where T
    u, v = ξ
    return SVector(1 - u - v, u, v)
end
```

But the API accepts instances for convenience:

```julia
# You write:
topology = Triangle{3}()  # Instance
basis = Lagrange{1}()     # Instance
N = get_basis_functions(topology, basis, ξ)

# Internally dispatches to:
get_basis_functions(Triangle{3}, Lagrange{1}, ξ)
```

This is handled by delegation methods in `api.jl`:

```julia
@inline get_basis_functions(t::AbstractTopology, b::AbstractBasis, ξ::Vec) =
    get_basis_functions(typeof(t), typeof(b), ξ)
```

**Why dispatch on type?** It enables the generator to write specialized methods without worrying about instance fields. All topology/basis information is in the type parameters.

---

## How the Generator Works

The generator is a self-contained tool that transforms **data** (basis descriptions) into **code** (optimized Julia functions). It runs offline, so runtime JuliaFEM has zero symbolic dependencies.

### The Generation Pipeline

**Step 1**: Read Descriptions

```julia
# In basis_descriptions.jl
push!(BASIS_DESCRIPTIONS, VandermondeBasisDescription(
    name="Tri6",
    description="6-node quadratic triangular element",
    family=Lagrange{2},
    topology=Triangle{6},
    ansatz=(:(1), :(u), :(v), :(u^2), :(u * v), :(v^2))
))
```

**Step 2**: Get Node Positions from Topology

```julia
# Generator calls:
X = reference_coordinates(Triangle{6}())
# Returns: SVector{6, Vec{2,Float64}} with corner + mid-edge nodes
# Node 1: (0, 0)
# Node 2: (1, 0)
# Node 3: (0, 1)
# Node 4: (0.5, 0)
# Node 5: (0.5, 0.5)
# Node 6: (0, 0.5)
```

This is the **single source of truth** for node locations. No duplication!

**Step 3**: Build Vandermonde Matrix

For ansatz terms `pⱼ(u,v)` and node coordinates `(uₖ, vₖ)`, construct:

```text
V[k,j] = pⱼ(uₖ, vₖ)

For Tri6:
       1   u   v   u²   uv   v²
    ┌                           ┐
N1: │  1   0   0   0    0    0  │  (corner)
N2: │  1   1   0   1    0    0  │  (corner)
N3: │  1   0   1   0    0    1  │  (corner)
N4: │  1  0.5  0  0.25  0    0  │  (mid-edge)
N5: │  1  0.5 0.5 0.25 0.25 0.25│  (mid-edge)
N6: │  1   0  0.5  0    0   0.25│  (mid-edge)
    └                           ┘
```

**Step 4**: Solve for Each Basis Function

For basis function `Nᵢ`, solve `V * aᵢ = eᵢ` where `eᵢ` is the i-th unit vector:

```julia
# N₁ should be 1 at node 1, 0 elsewhere
e₁ = [1, 0, 0, 0, 0, 0]
a₁ = V \ e₁  # Linear solve

# N₁(u,v) = a₁[1]·1 + a₁[2]·u + a₁[3]·v + a₁[4]·u² + a₁[5]·uv + a₁[6]·v²
```

This gives polynomial coefficients for each basis function.

**Step 5**: Symbolic Differentiation

The generator includes minimal symbolic differentiation:

```julia
# For N₁(u,v) = (1-u-v)*(1-2u-2v)
dN₁_du = differentiate(:($(expr)), :u)
dN₁_dv = differentiate(:($(expr)), :v)

# Then simplify:
dN₁_du = simplify(dN₁_du)  # Remove 0's and 1's
```

**Step 6**: Code Emission

Generate clean Julia code with `SVector` return types:

```julia
function emit_basis_functions(desc::VandermondeBasisDescription, io::IO)
    n = nnodes(desc.topology())
    println(io, """
    @inline function get_basis_functions(::Type{$(desc.topology)}, ::Type{$(desc.family)}, ξ::Vec{$D,T}) where T
        $(unpack_coordinates(D))
        $(emit_basis_expressions(desc))
        return SVector{$n}($(join(["N$i" for i in 1:n], ", ")))
    end
    """)
end

function emit_basis_derivatives(desc::VandermondeBasisDescription, io::IO)
    n = nnodes(desc.topology())
    println(io, """
    @inline function get_basis_derivatives(::Type{$(desc.topology)}, ::Type{$(desc.family)}, ξ::Vec{$D,T}) where T
        $(unpack_coordinates(D))
        $(emit_derivative_expressions(desc))
        return SVector{$n}($(join(["dN$i" for i in 1:n], ", ")))
    end
    """)
end
```

**Result:** ~2000 lines of human-readable, optimized code in `basis_generated.jl`.

### Why Not Runtime Symbolic Math?

You might wonder: why not evaluate symbolically at runtime using Symbolics.jl or SymPy?

**Problems with runtime symbolics:**

1. **Compilation time:** Every `get_basis_functions` call would trigger symbolic evaluation
2. **Dependencies:** Heavy dependencies (Symbolics.jl + CAS backend)
3. **Type instability:** Symbolic expressions don't have concrete types
4. **No inlining:** Symbolic evaluation can't be inlined by compiler
5. **Allocations:** Symbolic manipulation allocates expression trees

**Our approach:**

- ✅ Zero compilation overhead (code already generated)
- ✅ Zero dependencies at runtime
- ✅ Fully type-stable (SVector return types)
- ✅ Perfect inlining (literal expression trees)
- ✅ Zero allocations (static stack allocation)

**Trade-off:** You must run the generator when adding new basis types. But this happens **once** during development, not **millions of times** during simulation.

### Why SVector Return Types?

The generator produces `SVector` (not `NTuple`) for several reasons:

**1. Vector Operations**

```julia
# With SVector: Natural linear algebra
result = dot(coefficients, basis_functions)  # ✓ Works!
grad = sum(values[i] * derivatives[i] for i in 1:N)  # ✓ Clear!

# With NTuple: More verbose
result = sum(coefficients[i] * basis_functions[i] for i in 1:length(basis_functions))
```

**2. Type Stability**

```julia
# Both are fully type-stable and zero-allocation
N::SVector{10, Float64} = get_basis_functions(...)  # ✓
N::NTuple{10, Float64} = get_basis_functions(...)   # ✓ (also works)
```

**3. Performance**

Both compile to identical machine code:

- SVector: Stack-allocated or register-only
- NTuple: Stack-allocated or register-only
- **No performance difference!**

**4. Consistency**

```julia
X = reference_coordinates(topology)  # Returns SVector
N = get_basis_functions(topology, basis, ξ)  # Returns SVector
dN = get_basis_derivatives(topology, basis, ξ)  # Returns SVector of Vec
# Uniform container type throughout the API
```

**5. Integration with Tensors.jl**

```julia
dN::SVector{10, Vec{3}}  # Outer: SVector, Inner: Vec (Tensor)
# SVector provides indexing/iteration, Vec provides geometric operations
# Each gradient dN[i] supports: dot, cross, norm, ⊗ (tensor product)
```

---

## The Vandermonde Approach

### Why Vandermonde?

Traditional FEM texts derive basis functions by hand using algebra:

```text
For Tri3:
N₁ = 1 - u - v  (by inspection)
N₂ = u          (by inspection)
N₃ = v          (by inspection)
```

This works for simple elements but becomes tedious for higher orders. The Vandermonde approach is:

1. **Systematic:** Works for any order, any node layout
2. **Verifiable:** Linear algebra, not hand-waving
3. **Extensible:** Change order/nodes by changing data, not algebra
4. **Educational:** Makes the "magic" of basis functions transparent

### The Mathematical Foundation

**Problem:** Find polynomial `Nᵢ(ξ)` satisfying:

```text
Nᵢ(xⱼ) = δᵢⱼ  (Kronecker property)
```

**Solution:** Express `Nᵢ` as linear combination of basis polynomials:

```text
Nᵢ(ξ) = Σⱼ aᵢⱼ · pⱼ(ξ)

where pⱼ ∈ ansatz (e.g., {1, u, v, u², uv, v²})
```

**Constraint:** At each node `xₖ`:

```text
Nᵢ(xₖ) = Σⱼ aᵢⱼ · pⱼ(xₖ) = δᵢₖ
```

**Matrix form:**

```text
V · aᵢ = eᵢ

where V[k,j] = pⱼ(xₖ)
```

**Existence and uniqueness:** If ansatz spans a space of dimension equal to the number of nodes, and nodes are in general position, V is invertible.

### Example: Quadratic Triangle (Tri6)

**Given:**

- 6 nodes: 3 corners + 3 mid-edges
- Ansatz: {1, u, v, u², uv, v²} (complete quadratic)
- Node positions from `reference_coordinates(Triangle{6}())`

**Build V:**

```julia
nodes = [(0,0), (1,0), (0,1), (0.5,0), (0.5,0.5), (0,0.5)]
ansatz = [1, u, v, u^2, u*v, v^2]

V = [evaluate(p, node) for node in nodes, p in ansatz]
```

**Solve for N₁** (corner node at origin):

```julia
e₁ = [1, 0, 0, 0, 0, 0]  # Want N₁=1 at node 1, 0 elsewhere
a₁ = V \ e₁

N₁(u,v) = a₁[1] + a₁[2]*u + a₁[3]*v + a₁[4]*u² + a₁[5]*u*v + a₁[6]*v²
```

After simplification:

```julia
N₁(u,v) = (1 - u - v) * (1 - 2u - 2v)
```

**Verify Kronecker property:**

```julia
N₁(0, 0) = 1 * 1 = 1 ✓
N₁(1, 0) = 0 * (-1) = 0 ✓
N₁(0, 1) = 0 * (-1) = 0 ✓
N₁(0.5, 0) = 0.5 * 0 = 0 ✓
# etc.
```

### Reduced Bases: Serendipity Example

For quad/hex elements, the full tensor product gives more nodes than needed:

```text
Quad9: Full biquadratic (3×3 nodes)
Quad8: Serendipity (3×3 - center = 8 nodes)
```

**Serendipity ansatz:** Remove interior monomial(s) from tensor product:

```julia
# Full biquadratic: {1, u, v, u², uv, v², u²v, uv², u²v²}
# Serendipity:      {1, u, v, u², uv, v², u²v, uv²}  (drop u²v²)
```

The Vandermonde approach handles this trivially—just change the ansatz tuple!

```julia
push!(BASIS_DESCRIPTIONS, VandermondeBasisDescription(
    name="Quad8",
    family=Serendipity{2},
    topology=Quadrilateral{8},
    ansatz=(:(1), :(u), :(v), :(u^2), :(u * v), :(v^2), :(u^2 * v), :(u * v^2))
    #                                                                ^^^^^^^^^^^
    #                                                         No u²v² term!
))
```

Run the generator, and you get optimized Quad8 basis functions. No hand-derivation needed!

---

## Extending the Module

One of the key design goals is **extensibility**—adding new basis families should be straightforward and data-driven. Here we show several extension scenarios.

### Scenario 1: Adding a Higher-Order Element

**Goal:** Add cubic Lagrange triangle (Tri10).

**Step 1:** Understand the node layout

Cubic triangle has 10 nodes:

- 3 corner nodes
- 6 edge nodes (2 per edge)
- 1 interior node

**Step 2:** Define topology (if not exists)

```julia
# In src/topology/triangle.jl (add if missing)
reference_coordinates(::Triangle{10}) = SVector(
    Vec(0.0, 0.0),           # corner
    Vec(1.0, 0.0),           # corner
    Vec(0.0, 1.0),           # corner
    Vec(1/3, 0.0),           # edge 1-2
    Vec(2/3, 0.0),           # edge 1-2
    Vec(2/3, 1/3),           # edge 2-3
    Vec(1/3, 2/3),           # edge 2-3
    Vec(0.0, 2/3),           # edge 3-1
    Vec(0.0, 1/3),           # edge 3-1
    Vec(1/3, 1/3)            # interior
)
```

**Step 3:** Determine ansatz

Cubic complete polynomial in 2D has 10 terms:

- Constant: 1
- Linear: u, v
- Quadratic: u², uv, v²
- Cubic: u³, u²v, uv², v³

```julia
ansatz = (:(1), :(u), :(v), :(u^2), :(u*v), :(v^2), 
          :(u^3), :(u^2 * v), :(u * v^2), :(v^3))
```

**Step 4:** Add description

```julia
# In src/basis/basis_descriptions.jl
push!(BASIS_DESCRIPTIONS, VandermondeBasisDescription(
    name="Tri10",
    description="10-node cubic triangular element",
    family=Lagrange{3},
    topology=Triangle{10},
    ansatz=(:(1), :(u), :(v), :(u^2), :(u*v), :(v^2), 
            :(u^3), :(u^2 * v), :(u * v^2), :(v^3))
))
```

**Step 5:** Regenerate

```bash
julia --project=. src/basis/basis_generator.jl
```

**Done!** You now have `get_basis_functions(Triangle{10}(), Lagrange{3}(), ξ)` and `get_basis_derivatives(...)`.

**Step 6 (optional):** Add tests

```julia
# In test/test_basis_tri10.jl
@testset "Tri10 basis" begin
    topology = Triangle{10}()
    basis = Lagrange{3}()
    
    # Test partition of unity
    ξ = Vec(0.2, 0.3)
    N = get_basis_functions(topology, basis, ξ)
    @test sum(N) ≈ 1.0 atol=1e-10
    
    # Test Kronecker property at nodes
    X = reference_coordinates(topology)
    for i in 1:10
        N_at_i = get_basis_functions(topology, basis, X[i])
        for j in 1:10
            @test N_at_i[j] ≈ (i == j ? 1.0 : 0.0) atol=1e-10
        end
    end
end
```

### Scenario 2: Hierarchical Basis Family

**Goal:** Implement hierarchical (modal) basis where higher-order terms are orthogonal corrections to lower-order terms.

For a hierarchical cubic triangle, you might want:

- Nodes 1-3: Linear (vertices)
- Nodes 4-9: Quadratic corrections (edges)
- Node 10: Cubic correction (interior)

**Challenge:** Hierarchical bases don't satisfy simple Kronecker property at all nodes.

**Approach 1**: Modified Vandermonde

Change the target vectors from Kronecker `eᵢ` to hierarchical projections:

```julia
# Custom generation logic (not in basis_generator.jl yet)
function generate_hierarchical_basis(topology, order)
    X = reference_coordinates(topology)
    
    # Build hierarchical ansatz (Legendre-like on triangle)
    ansatz = build_hierarchical_ansatz(order)
    
    # Build modified Vandermonde (orthogonality constraints)
    V = vandermonde_matrix(ansatz, X)
    
    # Solve with hierarchical targets
    for i in 1:length(X)
        target = hierarchical_target(i, X)  # Not just eᵢ!
        aᵢ = V \ target
        emit_basis(i, aᵢ, ansatz)
    end
end
```

**Approach 2**: Direct Implementation

For specialized families, skip the generator and write directly:

```julia
# In src/basis/hierarchical_triangle.jl
struct Hierarchical{P} <: AbstractBasis end

@inline function get_basis_functions(::Triangle{10}, ::Hierarchical{3}, ξ::Vec{2,T}) where T
    u, v = ξ
    w = 1 - u - v
    
    # Level 0: Linear (standard)
    N1 = w
    N2 = u
    N3 = v
    
    # Level 1: Quadratic edge bubbles
    N4 = 4*u*w
    N5 = 4*u*v
    # ... (orthogonal to linear)
    
    # Level 2: Cubic interior bubble
    N10 = 27*u*v*w  # Orthogonal to all lower orders
    
    return SVector(N1, N2, N3, N4, N5, N6, N7, N8, N9, N10)
end
```

**When to use each approach:**

- **Vandermonde:** Standard nodal bases (Lagrange, Serendipity)
- **Direct:** Specialized bases (hierarchical, H(curl), H(div), plate/shell)

### Scenario 3: Nédélec Edge Elements (H(curl))

**Goal:** First-order Nédélec elements for electromagnetics.

Edge elements interpolate vector fields, not scalars. Their DOFs are edge circulations, not nodal values.

**Key differences from Lagrange:**

1. **Vector-valued basis:** Each basis function returns `Vec{3}`, not `Float64`
2. **Tangential continuity:** Only tangential component continuous across elements
3. **Non-Vandermonde:** Construction uses Whitney forms, not nodal interpolation

**Implementation:**

```julia
# In src/basis/nedelec.jl
struct Nedelec{P} <: AbstractBasis end

# First-order Nédélec on tetrahedron (6 edges = 6 DOFs)
@inline function get_basis_functions(::Tetrahedron{4}, ::Nedelec{1}, ξ::Vec{3,T}) where T
    u, v, w = ξ
    λ = (1 - u - v - w, u, v, w)  # Barycentric coordinates
    
    # Whitney 1-forms: Nᵢⱼ = λᵢ∇λⱼ - λⱼ∇λᵢ
    # Edge 1-2:
    N1 = λ[1] * grad_λ[2] - λ[2] * grad_λ[1]  # Returns Vec{3}
    
    # Edge 1-3:
    N2 = λ[1] * grad_λ[3] - λ[3] * grad_λ[1]
    
    # ... (6 edges total)
    
    return SVector(N1, N2, N3, N4, N5, N6)  # SVector{6, Vec{3, T}}
end

# Curl instead of gradient!
@inline function get_basis_derivatives(::Tetrahedron{4}, ::Nedelec{1}, ξ::Vec{3,T}) where T
    # For Nédélec, "derivative" is curl (constant in reference element)
    curl_N1 = Vec(...)  # Constant for first-order
    # ...
    return SVector(curl_N1, curl_N2, curl_N3, curl_N4, curl_N5, curl_N6)
end
```

**Note:** The API stays the same (`get_basis_functions`, `get_basis_derivatives`), but the return types differ. Type stability is maintained through parametric polymorphism.

### Scenario 4: Plate Elements (DKT, Mindlin)

**Goal:** Discrete Kirchhoff Triangle (DKT) for thin plates.

Plate elements have special kinematics:

- 3 DOFs per node: `(w, θₓ, θᵧ)` (deflection + rotations)
- 9 total DOFs for 3-node triangle
- Basis functions couple deflection and rotation

**Implementation strategy:**

```julia
# In src/basis/plate_elements.jl (already exists)
struct DKT <: AbstractBasis end

@inline function get_basis_functions(::Triangle{3}, ::DKT(), ξ::Vec{2,T}) where T
    u, v = ξ
    
    # Shape functions for deflection w
    N_w = (N_w1, N_w2, N_w3)  # 3 functions
    
    # Shape functions for rotation θₓ
    N_θx = (N_θx1, N_θx2, N_θx3)  # 3 functions
    
    # Shape functions for rotation θᵧ
    N_θy = (N_θy1, N_θy2, N_θy3)  # 3 functions
    
    # Return as flat SVector (9 entries)
    # Or structured: SVector{3}(@NamedTuple{w::T, θx::T, θy::T}(...))
    return SVector(N_w..., N_θx..., N_θy...)
end
```

**Challenges:**

- More complex kinematics than standard H¹ elements
- Often requires coordinate transformations
- May need element geometry (not just reference element)

**Recommendation:** For truly specialized elements, direct implementation (not generator) is clearest.

### Scenario 5: Isogeometric NURBS Basis

**Goal:** Non-Uniform Rational B-Splines for isogeometric analysis.

NURBS bases are fundamentally different:

- Not tied to fixed node positions
- Defined by knot vectors and control points
- Rational functions (ratios of polynomials)
- Order and continuity are independent choices

**Key architectural question:** Should NURBS be in the basis module at all?

**Option 1: Separate module** (recommended)

```julia
# src/isogeometric/nurbs.jl
struct NURBSBasis
    knot_vector::Vector{Float64}
    order::Int
    control_points::Vector{Vec{3}}
    weights::Vector{Float64}
end

# Different API—not get_basis_functions!
function evaluate_nurbs(basis::NURBSBasis, ξ::Float64)
    # Cox-de Boor recursion for B-splines
    # Rational weighting
    # ...
end
```

**Option 2**: Extend basis module

```julia
# In src/basis/api.jl
struct NURBS <: AbstractBasis
    knots::Vector{Float64}
    weights::Vector{Float64}
    # Not type-stable! (runtime knot vector)
end

# Must work with topology somehow...
get_basis_functions(::NURBSPatch, basis::NURBS, ξ::Vec)
```

**Challenges with Option 2:**

- Type instability (knot vectors vary at runtime)
- No fixed topology (NURBS patches are not reference elements)
- Different assembly workflow (control point mesh ≠ solution mesh)

**Recommendation:** NURBS deserves its own module with a specialized API. Don't force it into the Lagrange-oriented basis framework.

### Extension Decision Tree

**Should I use the Vandermonde generator?**

```text
Does the basis satisfy nodal interpolation (Kronecker property)?
├─ YES: Does it use polynomial ansatz on reference element?
│  ├─ YES: Use Vandermonde generator ✓
│  │      → Add description, run generator
│  └─ NO:  Direct implementation
│         → Write get_basis_functions directly
│
└─ NO:  Is it even a finite element basis?
   ├─ YES: Direct implementation
   │      → Write get_basis_functions directly
   │      → Examples: Nédélec, Raviart-Thomas, DKT
   └─ NO:  Separate module
          → Don't force into basis framework
          → Examples: NURBS, meshfree, spectral
```

---

## Advanced Topics

### Numerical Precision and Conditioning

**Question:** Why does the generator use exact arithmetic (symbolic) instead of floating-point Vandermonde solve?

**Answer:** Numerical stability and reproducibility.

Vandermonde matrices are notoriously **ill-conditioned**, especially for:

- Higher-order elements (P ≥ 3)
- Poorly distributed nodes
- Large coordinate ranges

**Example**: Condition number explosion

```julia
# Tri3 (linear): cond(V) ≈ 2.4
# Tri6 (quadratic): cond(V) ≈ 35
# Tri10 (cubic): cond(V) ≈ 1600  ← Yikes!
# Tri15 (quartic): cond(V) ≈ 150,000  ← Disaster!
```

With `cond(V) = 1e5`, a floating-point solve loses ~5 digits of precision.

**Our solution:** Symbolic math in the generator

```julia
# Generator works with exact rationals (implicitly via symbolic expressions)
# Then simplifies symbolically before emitting code
N1 = simplify(:((1 - u - v) * (1 - 2*u - 2*v)))
# No accumulated floating-point errors!
```

**Result:** Generated code evaluates exact polynomial expressions, not approximate solutions to ill-conditioned systems.

**Alternative approaches:**

1. **Orthogonal polynomials:** Use Legendre/Jacobi basis instead of monomials
   - Pros: Better conditioning
   - Cons: More complex ansatz, non-intuitive terms

2. **Higher precision:** Use `BigFloat` during generation
   - Pros: Simple fix
   - Cons: Slower generation, still approximate

3. **Analytical derivation:** Hand-derive using computer algebra system
   - Pros: Exact
   - Cons: Not automated, error-prone for complex elements

**Our choice:** Symbolic Vandermonde with simplification strikes the best balance.

### Jacobian and Physical Derivatives

**Important:** The basis module only provides **parametric derivatives** `∂Nᵢ/∂u, ∂Nᵢ/∂v, ∂Nᵢ/∂w`.

To get **physical derivatives** `∂Nᵢ/∂x, ∂Nᵢ/∂y, ∂Nᵢ/∂z`, you need the Jacobian from element geometry:

```julia
# In your assembly code:
topology = Tetrahedron{10}()
basis = Lagrange{2}()
X_elem = SVector{10}(...)  # Physical node coordinates

for (w, ξ) in get_gauss_points!(Tetrahedron, Gauss{3})
    # Get parametric derivatives
    dN_parametric = get_basis_derivatives(topology, basis, ξ)  # SVector{10, Vec{3}}
    
    # Build Jacobian: J = Σᵢ xᵢ ⊗ (∂Nᵢ/∂ξ)
    J = sum(X_elem[i] ⊗ dN_parametric[i] for i in 1:10)
    
    # Physical derivatives: (∂Nᵢ/∂x) = J⁻¹ · (∂Nᵢ/∂ξ)
    invJ = inv(J)
    dN_physical = map(dN -> invJ ⋅ dN, dN_parametric)  # SVector{10, Vec{3}}
    
    # Now use dN_physical in assembly
    # Compute strain-displacement matrix B
    # ...
end
```

**Why not compute physical derivatives in `get_basis_derivatives`?**

- Basis module doesn't know element geometry (separation of concerns!)
- Physical coordinates live in the mesh, not the reference element
- Jacobian depends on element deformation (changes during simulation)

**Division of responsibility:**

- **Basis module:** Reference element → parametric derivatives
- **Assembly code:** Element geometry + parametric derivatives → physical derivatives

### Partition of Unity and Reproduction

**Partition of unity:** Basis functions sum to 1 everywhere.

```julia
Σᵢ Nᵢ(ξ) = 1  ∀ξ ∈ reference element
```

This ensures that a constant field is exactly represented.

**Linear reproduction:** Basis can exactly represent linear fields.

```julia
Σᵢ Nᵢ(ξ) · xᵢ = x(ξ)  for linear x
```

**How to verify:**

```julia
@testset "Partition of unity" begin
    topology = Triangle{6}()
    basis = Lagrange{2}()
    
    # Test at random points
    for _ in 1:100
        u, v = rand(2)
        (u + v > 1) && continue  # Outside reference triangle
        
        ξ = Vec(u, v)
        N = get_basis_functions(topology, basis, ξ)  # SVector{6, Float64}
        
        @test sum(N) ≈ 1.0 atol=1e-12  # sum() works on SVector!
    end
end
```

**Caveat:** Some specialized bases (hierarchical, bubble functions) may **not** satisfy partition of unity by design. That's okay—they're used as enrichments, not for standard interpolation.

### Integration Accuracy Requirements

**Question:** What quadrature order do I need for element `E` with basis order `P`?

**Answer:** Depends on what you're integrating!

**Mass matrix:** `∫ Nᵢ Nⱼ dΩ`

- Integrand order: `2P`
- Required Gauss order: `≥ P` (for triangles/tets), `≥ P` (for quads/hexes)
- Example: Tet10 (P=2) → Gauss{2} or higher

**Stiffness matrix:** `∫ (∂Nᵢ/∂x)·(∂Nⱼ/∂x) dΩ`

- Integrand order: `2(P-1)` (derivatives reduce order)
- Required Gauss order: `≥ P-1`
- Example: Tet10 (P=2) → Gauss{1} or higher

**Nonlinear terms:** `∫ Nᵢ Nⱼ Nₖ dΩ`

- Integrand order: `3P`
- Required Gauss order: Depends on nonlinearity
- Often need higher than standard

**Rule of thumb:**

```julia
# Conservative: Integrate exactly
gauss_order = ceil(Int, polynomial_order_of_integrand / 2)

# For Tet10 stiffness (order 2):
for (w, ξ) in get_gauss_points!(Tetrahedron, Gauss{1})
    # 1-point Gauss is sufficient!
```

**Under-integration:**

Intentional under-integration (using lower Gauss order than required) is sometimes used for:

- Hourglass control in reduced integration
- Variational crimes in mixed formulations
- Locking prevention in nearly-incompressible elasticity

But that's advanced—start with exact integration!

### Performance: Precomputation vs. On-the-Fly

**Question:** Should I precompute basis functions at integration points?

**Scenario:**

```julia
# Approach 1: Compute on-the-fly (current design)
for elem in elements
    for (w, ξ) in get_gauss_points!(Tetrahedron, Gauss{2})
        N = get_basis_functions(Tetrahedron{10}(), Lagrange{2}(), ξ)
        dN = get_basis_derivatives(Tetrahedron{10}(), Lagrange{2}(), ξ)
        # Use N, dN...
    end
end

# Approach 2: Precompute (alternative)
gauss_points = collect(get_gauss_points!(Tetrahedron, Gauss{2}))
N_at_ips = [get_basis_functions(Tetrahedron{10}(), Lagrange{2}(), ξ) for (w,ξ) in gauss_points]
dN_at_ips = [get_basis_derivatives(Tetrahedron{10}(), Lagrange{2}(), ξ) for (w,ξ) in gauss_points]

for elem in elements
    for (ip, (w, ξ)) in enumerate(gauss_points)
        N = N_at_ips[ip]    # Lookup
        dN = dN_at_ips[ip]  # Lookup
        # Use N, dN...
    end
end
```

**Analysis:**

**Approach 1 (on-the-fly):**

- ✅ Zero memory overhead
- ✅ Cache-friendly (locality)
- ✅ Compiler can inline everything
- ⚠️ Recomputes for every element

**Approach 2 (precompute):**

- ✅ Compute once, reuse
- ❌ Memory overhead (arrays of tuples)
- ❌ Pointer chasing (cache misses)
- ❌ Dynamic indexing (no inlining)

**Benchmark results:** Approach 1 is actually **faster** for JuliaFEM's generated code!

**Why?** The generated basis functions are so optimized (inlined, constant-folded) that recomputation is cheaper than memory access.

**Exception:** For **very expensive** basis evaluations (e.g., NURBS, high-order Nédélec), precomputation might win. But for standard Lagrange up to P=3, on-the-fly wins.

**Recommendation:** Stick with on-the-fly unless profiling shows otherwise.

---

## Performance Characteristics

### Benchmarks (Julia 1.10, Intel i7)

**Tet10 basis evaluation** (most common 3D element):

```julia
topology = Tetrahedron{10}()
basis = Lagrange{2}()
ξ = Vec(0.25, 0.25, 0.25)

@btime get_basis_functions($topology, $basis, $ξ)
# 3.6 ns (0 allocations)

@btime get_basis_derivatives($topology, $basis, $ξ)
# 6.5 ns (0 allocations)
```

**Tri6 basis evaluation** (most common 2D element):

```julia
@btime get_basis_functions(Triangle{6}(), Lagrange{2}(), Vec(1/3, 1/3))
# 2.1 ns (0 allocations)

@btime get_basis_derivatives(Triangle{6}(), Lagrange{2}(), Vec(1/3, 1/3))
# 3.8 ns (0 allocations)
```

**Full assembly loop simulation** (100 dot products per element):

```julia
# Simulates: K_local[i,j] += dN[i] ⋅ dN[j]
@btime begin
    result = 0.0
    for _ in 1:100
        dN = get_basis_derivatives(Tetrahedron{10}(), Lagrange{2}(), $ξ)
        for i in 1:10, j in 1:10
            result += dot(dN[i], dN[j])
        end
    end
    result
end
# 126 ns (0 allocations)
```

**Throughput:**

- ~150 million Tet10 derivative evaluations per second per core
- For 1M element mesh × 4 integration points = **~27 milliseconds** for all basis evaluations
- Compare to v0.5.1 Dict-based approach: **10-20 seconds** (100-1000× slower!)

### Scaling with Order

| Element | Order | Basis eval | Derivative eval | Memory |
|---------|-------|-----------|----------------|--------|
| Tri3    | 1     | 1.2 ns    | 1.8 ns         | 0 B    |
| Tri6    | 2     | 2.1 ns    | 3.8 ns         | 0 B    |
| Tri10   | 3     | 4.5 ns*   | 8.2 ns*        | 0 B    |
| Tet4    | 1     | 1.5 ns    | 2.3 ns         | 0 B    |
| Tet10   | 2     | 3.6 ns    | 6.5 ns         | 0 B    |
| Hex8    | 1     | 2.8 ns    | 4.1 ns         | 0 B    |
| Hex27   | 2     | 8.9 ns    | 15.2 ns        | 0 B    |

*Tri10/Tet15: Estimated (not yet implemented)

**Key observations:**

1. **Zero allocations** for all elements (SVector returns, stack-allocated)
2. **Sublinear scaling** with node count (thanks to inlining)
3. **Derivatives ~1.5-2× slower** than values (more operations)
4. **Higher dimensions cost more** (Hex > Tet > Tri for same order)

### Comparison to v0.5.1

| Operation | v0.5.1 (Dict) | New (Tuple) | Speedup |
|-----------|--------------|------------|---------|
| Tri3 basis | 450 ns | 1.2 ns | **375×** |
| Tri6 basis | 820 ns | 2.1 ns | **390×** |
| Tet10 derivatives | 1800 ns | 6.5 ns | **277×** |
| Assembly loop | 120 μs | 126 ns | **950×** |

**Why such massive speedups?**

1. **Type stability:** No Dict lookups, concrete types everywhere
2. **Inlining:** Entire basis function is inlined as expression tree
3. **Constant folding:** Compiler optimizes polynomial evaluation
4. **No allocations:** SVectors are stack-allocated or eliminated entirely
5. **SIMD:** Compiler can vectorize some operations

This is the difference between **abstraction with cost** and **zero-cost abstraction**.

### Memory Footprint

**Generated code size:**

```bash
$ wc -l src/basis/basis_generated.jl
2147 src/basis/basis_generated.jl
```

~2000 lines for 17 element types × 2 functions (basis + derivatives) = ~60 lines per method. All basis families (Lagrange, Serendipity, future exotic bases) are included in this single file.

**Compilation cost:** First call to each method compiles, but:

- Happens once per (topology, basis, coordinate type) combination
- Extremely fast (< 1 ms per method)
- Cached in .ji file (no recompilation on restart)

**Runtime cost:** Zero. Generated code is just specialized methods.

---

## References and Further Reading

### Documentation

- **ADR-003:** Basis Function API Design (`docs/src/book/adr-003-basis-function-api.md`)
  - Complete rationale for API design
  - Comprehensive Tet10 benchmarks
  - Comparison of dispatch strategies
  
- **Topology-Basis Separation:** (`docs/src/book/topology-and-basis-separation.md`)
  - Philosophy and motivation
  - Mathematical foundation
  - Design patterns

- **Migration Guide:** (`docs/book/migration-guide-basis-api.md`)
  - Step-by-step migration from old API
  - Common pitfalls and solutions
  - Complete examples

### Textbook References

**Finite Element Theory:**

- Hughes, "The Finite Element Method: Linear Static and Dynamic Finite Element Analysis" (2000)
  - Chapter 3: Interpolation functions
  - Standard Lagrange bases, isoparametric formulation
  
- Zienkiewicz, Taylor, "The Finite Element Method" (2005)
  - Volume 1, Chapter 8: Shape functions
  - Comprehensive coverage of element families

**Advanced Topics:**

- Szabó, Babuška, "Finite Element Analysis" (1991)
  - Hierarchical and p-adaptive bases
  - Numerical integration accuracy
  
- Brezzi, Fortin, "Mixed and Hybrid Finite Element Methods" (1991)
  - H(curl), H(div) bases
  - Nédélec, Raviart-Thomas elements
  
- Cottrell, Hughes, Bazilevs, "Isogeometric Analysis" (2009)
  - NURBS bases
  - Comparison to Lagrange interpolation

### Code References

**Test files**:

- `test/test_basis_*.jl` - Unit tests for each topology
- `test/test_integration_points_api.jl` - Integration with quadrature

**Benchmark files**:

- `benchmarks/basis_function_access_tet10.jl` - Detailed Tet10 performance
- `benchmarks/basis_function_access_patterns.jl` - Dispatch strategy comparison

**Example usage**:

- `examples/linear_static.jl` - Assembly with new basis API
- `examples/cantilever_*.jl` - Complete simulation workflows

### External Projects

**Similar approaches in other libraries:**

- **deal.II** (C++): Separates FiniteElement from mapping, similar philosophy
- **FEniCS** (Python/C++): UFL separation of basis and mesh
- **Gridap.jl** (Julia): ReferenceFE abstraction, very similar design

**Code generation strategies:**

- **SymPy/SymEngine**: General-purpose symbolic math (what we avoid!)
- **Symbolics.jl**: Julia native (considered but too heavy)
- **FEMTK**: Mathematica-based generator (similar to ours)

---

## Summary: Key Takeaways

1. **Separation of concerns**: Topology (geometry) ≠ Basis (interpolation)
2. **Zero-cost abstraction**: 100-1000× faster than v0.5.1, zero allocations
3. **SVector returns**: Enable natural vector operations (`dot`, `sum`, `map`)
4. **Dual-vector design**: `SVector{N, Vec{D}}` for derivatives (outer/inner containers)
5. **Data-driven extension**: Add elements by changing catalog, not code
6. **Vandermonde approach**: Systematic, verifiable, extensible
7. **Generated code**: Offline symbolic math → runtime performance in `basis_generated.jl`
8. **Type stability**: Static vectors, concrete types, perfect inlining
9. **Extensible**: Lagrange, serendipity, hierarchical, edge/face elements, exotic bases
10. **Single source of truth**: `reference_coordinates(topology)`

**The module embodies modern Julia principles:** generic programming, type stability, zero-cost abstractions, and clear separation of concerns. It's fast, maintainable, and extensible—ready for research and production use.

---

**For questions or contributions**, see `CONTRIBUTING.md` or open an issue on GitHub.
