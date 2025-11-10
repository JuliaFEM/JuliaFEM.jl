---
title: "JuliaFEM Coding Standards"
description: "Required coding conventions and style guide for all contributors"
date: 2025-11-09
author: "Jukka Aho"
categories: ["development", "standards", "style guide"]
keywords: ["juliafem", "coding standards", "style", "conventions", "best practices"]
audience: "developers"
level: "required"
type: "standards"
status: "active"
---

This document defines the coding standards and conventions for JuliaFEM development.

**Last Updated:** November 9, 2025  
**Status:** Active - all new code must follow these standards

---

## Core Principles

1. **Readability over cleverness** - Code should be understandable by FEM practitioners
2. **Type stability first** - Performance depends on it (100x difference)
3. **Zero allocations in hot paths** - Profiling required
4. **Explicit over implicit** - No magic, show what happens
5. **Composition over inheritance** - Structs and free functions, not OOP

---

## Variable Naming Conventions

### No Greek Letters in Code (Critical!)

**Rule:** Never use Greek letters (ξ, η, ζ, α, β, γ, etc.) in code.

**Rationale:**

- **Keyboard accessibility** - Not all keyboards support Greek input
- **Editor compatibility** - Some editors struggle with Unicode math symbols
- **Copy-paste issues** - Greek letters cause encoding problems
- **Search/replace problems** - Text tools may not handle Unicode correctly
- **Terminal rendering** - SSH sessions may not display correctly
- **Internationalization** - Non-Western keyboards make editing difficult
- **Git diffs** - Unicode can cause merge conflicts or display issues
- **Accessibility** - Screen readers struggle with Greek letters

**Correct:**

```julia
# Reference element coordinates
function eval_basis(u::Float64, v::Float64, w::Float64)
    # Natural coordinates u, v, w ∈ [-1, 1]
    N1 = (1 - u) * (1 - v) * (1 - w) / 8
    return N1
end

# Physical coordinates
function map_to_physical(x::Vec, y::Vec, z::Vec, u::Float64, v::Float64, w::Float64)
    # Map from natural (u,v,w) to physical (x,y,z)
end
```

**Incorrect:**

```julia
# ❌ DON'T DO THIS
function eval_basis(ξ::Float64, η::Float64, ζ::Float64)
    N1 = (1 - ξ) * (1 - η) * (1 - ζ) / 8
    return N1
end
```

**Exception:** Greek letters are acceptable in:

- **Comments** - Mathematical notation for clarity: `# Shape function: Nᵢ(ξ)`
- **Documentation** - Latex math blocks: `$\\xi \\in [-1, 1]$`
- **String literals** - Plot labels: `xlabel="ξ coordinate"`
- **Error messages** - User-facing text: `"Invalid ξ coordinate"`

**Standard variable names:**

- Reference coordinates: `u`, `v`, `w` (not ξ, η, ζ)
- Physical coordinates: `x`, `y`, `z`
- Derivatives: `du`, `dv`, `dw` or `dudx`, `dudy`, etc.
- Jacobian: `J` or `jac` (not ∂)
- Determinant: `detJ` (not |J|)
- Inverse: `invJ` or `Jinv` (not J⁻¹)

---

## Type Naming

### Structs and Types

- **PascalCase** - All type names: `Element`, `Problem`, `Material`
- **No abbreviations** - `Quadrilateral` not `Quad` (unless established convention)
- **Descriptive suffixes** - Purpose clear from name

**Basis types** - Append "Basis" suffix to distinguish from topology:

```julia
# ✅ Correct - No name collision
struct Tri3Basis <: AbstractBasis{2} end  # Interpolation scheme
struct Tri3 <: AbstractTopology end        # Element geometry

# ❌ Wrong - Name collision!
struct Tri3 <: AbstractBasis{2} end
struct Tri3 <: AbstractTopology end  # ERROR: type Tri3 already defined
```

### Functions

- **snake_case** - All function names: `assemble_element`, `solve_static`
- **Verb-noun pattern** - Action clear: `compute_stiffness`, `evaluate_basis`
- **Boolean predicates** - `is_*` or `has_*`: `is_converged`, `has_contact`

### Constants

- **UPPER_CASE** - Module-level constants: `MAX_ITERATIONS`, `TOLERANCE`
- **Type-stable** - Always specify type: `const MAX_ITER::Int = 100`

### Internal/Private

- **Underscore prefix** - Not exported: `_compute_internal_forces`
- **Not API** - Can change between versions

---

## Code Organization

### File Structure

```julia
# Standard file header
# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE

"""
Brief one-line description of file purpose.

Extended description if needed.
"""

# Imports (grouped)
using LinearAlgebra
using SparseArrays

# Local imports
using ..JuliaFEM: AbstractElement, AbstractBasis

# Type definitions
struct MyType
    # ...
end

# Function implementations
function my_function(args)
    # ...
end
```

### Import Style

```julia
# ✅ Correct - Explicit imports
using LinearAlgebra: norm, dot, cross
using SparseArrays: sparse, spzeros

# ❌ Avoid - Blanket imports (pollutes namespace)
using LinearAlgebra
using SparseArrays
```

---

## Performance Guidelines

### Type Stability

```julia
# ✅ Type-stable - Return type inferrable
function compute_mass(element::Quad4, density::Float64)
    m::Float64 = 0.0
    # ...
    return m
end

# ❌ Type-unstable - Return type changes!
function compute_mass(element, density)
    if density > 0
        return 1.0      # Float64
    else
        return nothing  # Nothing - type-unstable!
    end
end
```

### Zero Allocations

```julia
# ✅ Pre-allocated cache
struct AssemblyCache
    K_local::Matrix{Float64}
    f_local::Vector{Float64}
end

function assemble!(cache::AssemblyCache, element)
    fill!(cache.K_local, 0.0)
    # Reuse cache.K_local - no allocations
end

# ❌ Allocates every call
function assemble(element)
    K_local = zeros(8, 8)  # Allocates!
    return K_local
end
```

### Tuple Returns (Zero Allocation)

```julia
# ✅ Tuple return - no allocation
function eval_basis(element::Tri3, u::Float64, v::Float64)
    N1 = 1 - u - v
    N2 = u
    N3 = v
    return (N1, N2, N3)  # NTuple{3,Float64} - stack allocated
end

# ❌ Vector return - allocates!
function eval_basis(element::Tri3, u::Float64, v::Float64)
    return [1 - u - v, u, v]  # Vector{Float64} - heap allocation
end
```

---

## Documentation Style

### Docstrings

Use Julia's docstring format with standard sections:

`````markdown
"""
    assemble_element(element::Quad4, u::Vector{Float64}) -> Matrix{Float64}

Assemble element stiffness matrix for 4-node quadrilateral element.

# Arguments
- `element::Quad4`: Quadrilateral element with nodal connectivity
- `u::Vector{Float64}`: Nodal displacement vector (8 DOFs: u1,v1,u2,v2,...)

# Returns
- `K::Matrix{Float64}`: 8×8 element stiffness matrix in global DOFs

# Theory
Uses 2×2 Gauss quadrature with bilinear shape functions:
```math
K_{ij} = \\int_{\\Omega_e} B_i^T D B_j \\, d\\Omega
```

# Example
```julia
nodes = [Node(0,0), Node(1,0), Node(1,1), Node(0,1)]
element = Quad4(nodes)
u = zeros(8)
K = assemble_element(element, u)
```

# Performance
This function allocates. For zero-allocation assembly, use `assemble_element!` 
with pre-allocated cache.
"""
function assemble_element(element::Quad4, u::Vector{Float64})
    # Implementation
end
`````

### Comments

```julia
# ✅ Good comments - Why, not what
# Use RCM ordering to minimize bandwidth (10x faster solve)
perm = rcm_permutation(mesh)

# Check convergence: ||Δu|| < ε||u||
# Relative norm prevents scale-dependent tolerance
if norm(Δu) < tol * norm(u)
    break
end

# ❌ Bad comments - Obvious from code
# Loop over elements
for element in elements
    # Add to global matrix
    K_global += K_local
end
```

---

## Testing Standards

### Test Organization

```julia
@testset "Quad4 element" begin
    @testset "Stiffness matrix" begin
        # Unit square, E=1, ν=0.3
        element = Quad4([Node(0,0), Node(1,0), Node(1,1), Node(0,1)])
        K = stiffness_matrix(element, E=1.0, ν=0.3)
        
        # Symmetry
        @test issymmetric(K)
        
        # Positive definite (after BC)
        @test all(eigvals(K[3:end, 3:end]) .> 0)
    end
    
    @testset "Patch test" begin
        # Linear displacement field must be exact
        # ... validation test ...
    end
end
```

### Floating Point Comparisons

```julia
# ✅ Use tolerances
@test result ≈ expected atol=1e-10
@test isapprox(result, expected, rtol=1e-6)

# ❌ Never exact equality for floats
@test result == expected  # Fragile!
```

---

## Anti-Patterns (Don't Do This!)

### 1. Dict Without Type Parameters

```julia
# ❌ Type-unstable Dict (100x slower!)
fields = Dict("displacement" => u, "velocity" => v)

# ✅ Type-stable alternative
fields = (displacement=u, velocity=v)  # NamedTuple
# or
struct Fields
    displacement::Vector{Float64}
    velocity::Vector{Float64}
end
```

### 2. Abstract Types in Structs

```julia
# ❌ Type-unstable struct
struct Element
    nodes::AbstractVector  # Type-unstable!
end

# ✅ Parametric struct
struct Element{N}
    nodes::NTuple{N, Node}  # Type-stable, zero-allocation
end
```

### 3. Global Variables

```julia
# ❌ Global mutable state
global_stiffness = zeros(1000, 1000)

function assemble!(element)
    global_stiffness .+= K_local  # Spooky action at a distance!
end

# ✅ Explicit parameters
function assemble!(K_global::Matrix, element)
    K_global .+= K_local  # Clear data flow
end
```

### 4. Type Piracy

```julia
# ❌ Extending methods on types you don't own
Base.+(a::Vector, b::Matrix) = ...  # DON'T!

# ✅ Wrapper type or different function
struct MyVector
    data::Vector
end
Base.+(a::MyVector, b::Matrix) = ...  # OK - our type
```

---

## Git Commit Style

### Commit Messages

```text
feat(topology): Add Pyr5 pyramid element topology

- Implement 5-node pyramid reference element
- Add connectivity information (faces, edges)
- Zero-allocation tuple interface
- Tests: reference coordinates, edge/face queries

Closes #123
```

**Format:** `<type>(<scope>): <subject>`

**Types:**

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `refactor`: Code restructuring (no behavior change)
- `perf`: Performance improvement
- `test`: Adding tests
- `chore`: Tooling, dependencies

**Scope:** Module or component (topology, assembly, solver, etc.)

**Subject:**

- Imperative mood: "Add feature" not "Added feature"
- No period at end
- Max 72 characters

---

## Editor Configuration

### Recommended Settings

```julia
# .editorconfig
[*.jl]
charset = utf-8
end_of_line = lf
indent_style = space
indent_size = 4
trim_trailing_whitespace = true
insert_final_newline = true
```

### JuliaFormatter.jl

```julia
# .JuliaFormatter.toml
indent = 4
margin = 92
always_for_in = true
whitespace_typedefs = true
whitespace_ops_in_indices = true
remove_extra_newlines = true
```

---

## Summary Checklist

Before submitting code, verify:

- [ ] No Greek letters in variable names (use u, v, w)
- [ ] All types are PascalCase
- [ ] All functions are snake_case
- [ ] Type-stable (check with `@code_warntype`)
- [ ] Zero allocations in hot paths (check with `@allocations` or `@btime`)
- [ ] Docstrings for exported functions
- [ ] Tests pass locally
- [ ] Comments explain "why" not "what"
- [ ] No type piracy
- [ ] No global mutable state

---

## References

- **Performance tips:** https://docs.julialang.org/en/v1/manual/performance-tips/
- **Style guide:** https://docs.julialang.org/en/v1/manual/style-guide/
- **JuliaFEM vision:** `llm/VISION_2.0.md`
- **Architecture:** `llm/ARCHITECTURE.md`
- **Technical lessons:** `llm/TECHNICAL_VISION.md`

---

**Enforcement:** These standards are enforced through code review. All PRs must follow these conventions.

**Evolution:** This document evolves with the project. Propose changes via pull request.
