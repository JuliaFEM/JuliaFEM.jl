# Elements Module

This module implements finite elements following Ciarlet's mathematical definition, adapted for computational efficiency.

## Ciarlet's Finite Element Triple (K, P, Σ)

### Mathematical Definition

A finite element is a triple **(K, P, Σ)** where:

- **K** ⊂ ℝⁿ: Compact, connected reference domain (geometric set)
- **P**: Finite-dimensional space of functions on K
- **Σ** = {σ₁, ..., σₙ}: Set of linear functionals σᵢ : P → ℝ (dual basis)

### Computational Implementation

We use **(K, P, S)** where:

- **K**: Reference domain type (e.g., `Triangle{3}`, `Tetrahedron{4}`) - **exact match**
- **P**: Polynomial space type (e.g., `Lagrange{1}`, `Lagrange{2}`) - **exact match**
- **S**: Field specification → **uniquely determines Σ** (computational encoding)

### Why S Instead of Σ?

**S does not equal Σ, but S determines Σ uniquely.**

For standard Lagrange elements:

| S specification | Resulting Σ functionals | Example |
|----------------|------------------------|---------|
| `Float64, Vertex` | σᵢ(u) = u(vertex_i) | Point evaluation (nodal values) |
| `Vec{3}, Vertex` | σᵢ(u) = uₐ(vertex_i), α=1,2,3 | Vector point evaluation |
| `Float64, Cell` | σ(u) = (1/\|K\|) ∫_K u dx | Cell-average functional |
| `Float64, Edge` | σ(u) = ∫_edge u ds | Edge integral functional |

**Rationale:**

1. Functionals are never instantiated in computational FEM
2. S contains the essential information: quantity type + entity location
3. Given (K, P, S), the functionals Σ are uniquely determined
4. Type-level encoding = zero runtime cost

## Element Structure

```julia
struct Element{K<:AbstractTopology, P<:AbstractBasis, S<:DOFSet, N}
    id::UInt                    # Element identifier (mesh index)
    dof_indices::NTuple{N,UInt64}  # Flat tuple of global DOF indices
end
```

### Design Philosophy

**Everything mathematical lives in the types.** The instance holds only:

- Identification (`id`)
- Assignment (`dof_indices`)

No connectivity, no coordinates stored in element! Mesh holds geometric data.

### Type Stability via @generated Constructor

The `dof_indices` field is typed as `NamedTuple` (without parameters), but the `@generated` constructor ensures the concrete type is inferred:

```julia
@generated function Element{K,P,S}(id::UInt, dof_indices::D) where {K,P,S,D<:NamedTuple}
    # Julia infers D = @NamedTuple{u::NTuple{12, Int64}} from the argument
    # Field access elem.dof_indices.u returns NTuple{12, Int64} (type-stable!)
end
```

This achieves zero-allocation performance without adding a 4th type parameter.

## Field Specifications

### Single-Field Elements

```julia
# Heat conduction (scalar field at vertices)
S = @NamedTuple{T::Tuple{Float64, Vertex}}
Element{Triangle{3}, Lagrange{1}, S}(UInt(1), (T=(1, 2, 3),))

# 2D elasticity (vector field at vertices)
S = @NamedTuple{u::Tuple{Vec{2}, Vertex}}
Element{Triangle{3}, Lagrange{1}, S}(UInt(1), (u=(1, 2, 3, 4, 5, 6),))
```

### Multi-Field Elements

```julia
# Thermo-mechanical coupling
S = @NamedTuple{
    T::Tuple{Float64, Vertex},      # Temperature at vertices
    u::Tuple{Vec{3}, Vertex}         # Displacement at vertices
}

Element{Tetrahedron{4}, Lagrange{1}, S}(
    UInt(1),
    (T=(1,2,3,4), u=(5,6,7,8,9,10,11,12,13,14,15,16))
)

# Access fields directly
elem.dof_indices.T  # (1, 2, 3, 4)
elem.dof_indices.u  # (5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
```

## DOF Extraction

Two extraction strategies for getting element DOFs from global solution:

### Flat Extraction

Returns scalars grouped by field:

```julia
u_global = [1.0, 2.0, ..., 20.0]
dofs = extract_element_dofs(elem, u_global)
# Returns: (u = (1.0, 2.0, 3.0, ..., 12.0),)
```

### Structured Extraction

Returns quantities matching field type (Vec, Tensor, etc.):

```julia
dofs = extract_element_dofs_structured(elem, u_global)
# Returns: (u = (Vec{3}(1,2,3), Vec{3}(4,5,6), Vec{3}(7,8,9), Vec{3}(10,11,12)),)
```

**Use case:** Structured extraction is for interpolation where tuple length must match shape function count:

```julia
u_interp = N1 * u1 + N2 * u2 + N3 * u3 + N4 * u4
```

Both are **zero-allocation** (5.5 ns) thanks to type stability and `@generated` functions.

## Local-Global DOF Mapping

For coupled multi-field assembly:

```julia
# Element with 2 fields: T (4 DOFs) + u (12 DOFs) = 16 total
map = local_to_global_map(elem)
# map[1:4] = [1,2,3,4]       Temperature DOFs
# map[5:16] = [10,...,21]    Displacement DOFs

# Assembly loop
K_local = zeros(16, 16)  # Fully coupled local matrix
# ... fill K_local with physics coupling (∂T/∂u, ∂u/∂T, etc.) ...
for i in 1:16, j in 1:16
    K_global[map[i], map[j]] += K_local[i, j]
end
```

### Field-Specific DOF Ranges

Extract local DOF ranges for field blocks (compile-time computation):

```julia
T_range = field_dof_range(elem, :T)  # 1:4
u_range = field_dof_range(elem, :u)  # 5:16

# Extract field-field coupling block
K_Tu = K_local[T_range, u_range]  # 4×12 temperature-displacement coupling
```

The range is computed at compile time via `@generated` - zero runtime cost.

## Type Queries

```julia
topology_type(elem)  # Tetrahedron{4}
basis_type(elem)     # Lagrange{1}
dof_type(elem)       # @NamedTuple{T::Tuple{Float64,Vertex}, u::Tuple{Vec{3},Vertex}}
nnodes(elem)         # 4
```

## Performance Notes

### Type Stability Achievement

The key to zero allocations was ensuring `elem.dof_indices` has a concrete type:

**Before (BAD):**

```julia
dof_indices::NamedTuple  # Type instability!
# Field access returns Any → heap allocation
```

**After (GOOD):**

```julia
@generated function Element{K,P,S}(id::UInt, dof_indices::D) where {K,P,S,D<:NamedTuple}
    # Julia infers D = @NamedTuple{u::NTuple{12,Int64}}
    # Field access returns NTuple{12,Int64} → stack allocation!
end
```

### Benchmark Results

```text
Flat extraction:       5.472 ns (0 allocations: 0 bytes)
Structured extraction: 5.474 ns (0 allocations: 0 bytes)
```

Compared to original implementation: **300× faster**, zero allocations.

## Files in This Module

- `elements.jl` - Element struct, constructors, type queries
- `extract_element_dofs.jl` - DOF extraction (flat and structured)
- `README.md` - This file (module documentation)

## See Also

- `docs/src/developer/dof_extraction.md` - Detailed DOF extraction design
- `test/elements/test_extract_element_dofs.jl` - Comprehensive test suite
