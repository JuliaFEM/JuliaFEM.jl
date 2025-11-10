---
title: "JuliaFEM System Architecture: Core Concepts"
date: 2025-11-10
author: "JuliaFEM Team"
status: "Authoritative"
last_updated: 2025-11-10
tags: ["architecture", "design", "concepts", "user-guide"]
---

## Introduction

This document explains **why** JuliaFEM's core data structures exist and **how**
they work together. Every struct has a purpose, every abstraction has a reason.

**Target audience:** Users who want to understand the system deeply,
contributors implementing new physics, anyone asking "why is it designed this
way?"

---

## The Big Picture: Method of Lines

JuliaFEM follows the **Method of Lines** approach to FEM:

1. **Spatial discretization** (FEM) → System of ODEs
2. **Time discretization** (if transient) → Nonlinear algebraic equations
3. **Linearization** (Newton) → Linear system solve
4. **Repeat** until convergence

**Key insight:** Separate concerns cleanly at each level!

```text
Mesh + Physics → Element Assembly → Global System → Solver → Solution
  ↓                    ↓                  ↓            ↓         ↓
Geometry          Material Models    Sparse Matrix  Krylov    u(x,t)
```

---

## Core Abstractions

### 1. `Element` - The Geometric Container

**Role:** Holds geometric and connectivity information for a single finite element.

**What it knows:**

- Element type (Tri3, Quad4, Tet10, etc.)
- Node connectivity
- Basis functions (shape functions)
- Integration points

**What it does NOT know:**

- Physics equations
- Material properties
- Boundary conditions

**Why this design?**

- ✅ Element is **reusable** across different physics (same Tet10 for elasticity, heat, fluid)
- ✅ Geometry is **immutable** (connectivity doesn't change during analysis)
- ✅ **Type-stable dispatch** on element type enables compiler optimizations

**Example:**

```julia
# Create a 10-node tetrahedral element
nodes = [1, 5, 12, 23, 14, 8, 19, 27, 31, 16]
element = Element(Tet10, nodes)

# Element knows its topology
@assert nnodes(element) == 10
@assert dim(element) == 3

# But element doesn't know about stress, temperature, etc.
# That's the job of Physics!
```

---

### 2. `Physics` - The Equation Selector (RENAMED from "Problem")

**Role:** Multiple dispatch tag + configuration for physical equations.

**What it is:**

- A **type** that selects assembly methods via multiple dispatch
- A **struct** that holds physics-specific configuration
- A **name provider** for field names ("displacement", "temperature", etc.)

**What it is NOT:**

- ❌ Not the mesh (that's separate)
- ❌ Not the material models (those are parameters)
- ❌ Not the solver (that's a different layer)

**Why "Physics" instead of "Problem"?**

- ✅ **Positive connotation** ("solve physics" vs "solve problem")
- ✅ **Accurate description** (selecting physical equations)
- ✅ **Clear role** (what physics are we simulating?)

**Types of Physics:**

```julia
# Elasticity: Solves ∇⋅σ = ρü + b
struct ElasticityPhysics <: AbstractPhysics
    formulation::Symbol    # :plane_stress, :plane_strain, :continuum
    finite_strain::Bool    # Geometric nonlinearity
    geometric_stiffness::Bool  # σ-dependent stiffness for buckling
    store_fields::Vector{Symbol}  # Output fields to save
end

# Heat transfer: Solves ∇⋅(k∇T) = ρcₚ∂T/∂t + Q
struct HeatPhysics <: AbstractPhysics
    formulation::Symbol    # :steady_state, :transient
    nonlinear::Bool        # Temperature-dependent properties
    store_fields::Vector{Symbol}
end

# Contact mechanics: Solves contact constraints
struct ContactPhysics <: AbstractPhysics
    algorithm::Symbol      # :penalty, :lagrange, :augmented_lagrange
    friction_model::Symbol # :coulomb, :frictionless
    # ...
end
```

**Multiple Dispatch in Action:**

```julia
# Compiler selects correct assembly method based on physics type!

function assemble!(assembly, physics::ElasticityPhysics, elements, time)
    # Elasticity-specific assembly:
    # - Compute strain from displacement
    # - Call material model: ε → (σ, 𝔻)
    # - Build stiffness matrix K and force vector f
end

function assemble!(assembly, physics::HeatPhysics, elements, time)
    # Heat transfer-specific assembly:
    # - Compute temperature gradient
    # - Call thermal conductivity: ∇T → q
    # - Build capacity matrix C and conductivity matrix K
end

function assemble!(assembly, physics::ContactPhysics, elements, time)
    # Contact-specific assembly:
    # - Detect penetration
    # - Compute contact forces
    # - Build constraint equations
end
```

**This is Julia's superpower!** No runtime type checks, no vtables, just fast compiled code for each physics type.

---

### 3. `Material` - The Constitutive Model

**Role:** Maps kinematic quantities to stress/flux/response.

**Interface (Elasticity example):**

```julia
abstract type AbstractMaterial end
abstract type AbstractMaterialState end

# Material computes: (ε, state_old, Δt) → (σ, 𝔻, state_new)
function compute_stress(
    material::AbstractMaterial,
    ε::SymmetricTensor{2,3},
    state_old::AbstractMaterialState,
    Δt::Float64
) -> Tuple{SymmetricTensor{2,3}, SymmetricTensor{4,3}, AbstractMaterialState}
    # Returns: (stress, tangent, state_new)
end
```

**Why separate from Physics?**

- ✅ **Modularity:** Change material without touching assembly code
- ✅ **Testability:** Unit-test materials independently
- ✅ **Performance:** Compiler specializes on material type
- ✅ **Clarity:** Material logic isolated from kinematics

**Material examples:**

```julia
# Stateless material (no history)
struct LinearElastic <: AbstractMaterial
    λ::Float64  # Lamé parameter
    μ::Float64  # Shear modulus
end

# Stateful material (history-dependent)
struct PerfectPlasticity <: AbstractMaterial
    E::Float64   # Young's modulus
    ν::Float64   # Poisson's ratio
    σ_y::Float64 # Yield stress
end

struct PlasticityState{T} <: AbstractMaterialState
    εₚ::SymmetricTensor{2,3,T}  # Plastic strain
    α::T                         # Equivalent plastic strain
end
```

**Performance:** 20-70 ns per material evaluation (validated Nov 10, 2025)

---

### 4. `Assembly` - The Global System Builder

**Role:** Accumulate element contributions into global matrices/vectors.

**What it holds:**

- Global stiffness matrix `K` (sparse)
- Global force vector `f`
- (Optional) Mass matrix `M`, damping `C`, geometric stiffness `Kg`

**What it does:**

- Pre-allocates sparse matrix structure
- Accumulates element contributions: `K += Kₑ`, `f += fₑ`
- Handles DOF mapping: local element DOFs → global system DOFs

**Why separate from Physics?**

- ✅ **Reusability:** Same Assembly struct for all physics types
- ✅ **Optimization:** Pre-allocated structure, efficient COO→CSC conversion
- ✅ **Parallelism:** (Future) Thread-safe assembly with color-based locking

**Example:**

```julia
# Create assembly
assembly = Assembly()

# Loop over elements
for element in elements
    # Compute element stiffness and force
    Kₑ, fₑ = assemble_element(physics, element, time)
    
    # Get global DOF indices
    gdofs = get_gdofs(element)
    
    # Add to global system
    add!(assembly.K, gdofs, gdofs, Kₑ)
    add!(assembly.f, gdofs, fₑ)
end

# Solve global system
u = assembly.K \ assembly.f
```

---

### 5. `IntegrationPoint` - The Quadrature Point

**Role:** Location and weight for numerical integration.

**What it knows:**

- Position in reference element: `ξ ∈ [-1,1]ᵈⁱᵐ`
- Integration weight: `w`
- Index/ID for state storage

**What it does NOT know:**

- Material state (that's stored per-element per-IP)
- Stress/strain (that's computed on-the-fly)

**Why this design?**

- ✅ **Immutable:** Integration points never change
- ✅ **Topology-specific:** Different rules for Tri3 vs Quad4
- ✅ **Pre-computed:** Created once, reused forever

**Example:**

```julia
# Get integration points for element type
ips = integration_points(Gauss{2}, Quad4)

# Each IP knows position and weight
for ip in ips
    ξ = ip.ξ      # Position: NTuple{2, Float64}
    w = ip.weight # Weight: Float64
    
    # Evaluate basis functions at this point
    N, ∇N_ref = evaluate_basis(basis, ξ)
    
    # Do integration: ∫f dΩ ≈ ∑ᵢ f(ξᵢ)⋅w(ξᵢ)
end
```

---

### 6. `BasisInfo` - The Shape Function Cache

**Role:** Pre-allocated workspace for basis function evaluation.

**What it caches:**

- Basis function values `N`
- Gradients in reference config `∇N_ref`
- Gradients in current config `∇N`
- Jacobian `J`, determinant `detJ`

**Why cache?**

- ✅ **Zero allocation:** Reuse same arrays for every element
- ✅ **Type stability:** All sizes known at compile time
- ✅ **Performance:** Avoid repeated memory allocation

**Example:**

```julia
# Create cache for Tet10 elements
bi = BasisInfo(Tet10)

# Reuse for every element
for element in elements
    for ip in integration_points(element)
        # Evaluate into pre-allocated cache
        eval_basis!(bi, element.X, ip)
        
        # Access cached results
        N = bi.N          # Shape functions
        ∇N = bi.grad      # Gradients ∂N/∂x
        w = ip.weight * bi.detJ  # Integration weight
    end
end
```

---

## Data Flow: From Mesh to Solution

### Step 1: Problem Setup

```julia
# Define physics
physics = ElasticityPhysics(
    formulation = :continuum,
    finite_strain = false,
    store_fields = [:stress, :strain]
)

# Define material
material = LinearElastic(E=200e9, ν=0.3)

# Create elements with material
elements = [Element(Tet10, conn) for conn in connectivity]
for el in elements
    el.material = material
    el.states_old = [NoState() for _ in 1:n_integration_points]
    el.states_new = [NoState() for _ in 1:n_integration_points]
end
```

### Step 2: Assembly Loop

```julia
assembly = Assembly()

for element in elements
    # Get element data
    X = element.geometry  # Nodal coordinates
    u = element.displacement  # Nodal displacements
    
    # Initialize element matrices
    Kₑ = zeros(ndofs, ndofs)
    fₑ = zeros(ndofs)
    
    # Integration point loop
    for (ip_idx, ip) in enumerate(element.integration_points)
        
        # 1. KINEMATICS: u → ε
        ∇N = shape_function_gradients(element, ip)
        ε = compute_strain_from_gradients(∇N, u)
        
        # 2. MATERIAL: ε → (σ, 𝔻, state)
        state_old = element.states_old[ip_idx]
        σ, 𝔻, state_trial = compute_stress(material, ε, state_old, Δt)
        
        # 3. ASSEMBLY: (∇N, σ, 𝔻) → (Kₑ, fₑ)
        w = integration_weight(ip)
        accumulate_stiffness!(Kₑ, ∇N, 𝔻, w)
        accumulate_internal_forces!(fₑ, ∇N, σ, w)
        
        # DON'T update states yet (Newton iterations!)
    end
    
    # 4. GLOBAL: Kₑ → K, fₑ → f
    gdofs = get_gdofs(element)
    add!(assembly.K, gdofs, gdofs, Kₑ)
    add!(assembly.f, gdofs, fₑ)
end
```

### Step 3: Solve

```julia
# Linear solve: K⋅Δu = f
Δu = assembly.K \ assembly.f

# Newton iteration (if nonlinear)
while norm(residual) > tolerance
    # Re-assemble with trial displacement
    u_trial = u_old + Δu
    
    # Solve linearized system
    Δu = assembly.K \ assembly.f
    
    # Update
    u_trial += Δu
end

# Converged! Commit state
for element in elements
    element.states_old .= element.states_new
end
```

---

## Design Principles

### 1. Separation of Concerns

**Each struct has ONE job:**

- `Element` → Geometry
- `Physics` → Equation selection
- `Material` → Constitutive model
- `Assembly` → Global system
- `Solver` → Linear algebra

**Benefits:**

- ✅ Easy to test (unit test each component)
- ✅ Easy to extend (add new material without touching assembly)
- ✅ Easy to optimize (profile each layer independently)

### 2. Type Stability

**Every function has concrete return type:**

```julia
# ✅ GOOD: Compiler knows return type
function compute_stress(m::LinearElastic, ε) -> Tuple{SymmetricTensor{2,3}, SymmetricTensor{4,3}, NoState}
    # ...
end

# ❌ BAD: Compiler doesn't know (Dict lookup)
function compute_stress(element, ip)
    stress = element.fields["stress"]  # Unknown type!
    # ...
end
```

**Performance impact:** 10-100× speedup from type stability alone!

### 3. Zero Allocation

**Hot paths allocate NOTHING:**

```julia
# Pre-allocate once
bi = BasisInfo(Tet10)
Kₑ = zeros(30, 30)

# Reuse in loop (zero allocations!)
for element in elements
    fill!(Kₑ, 0.0)
    for ip in integration_points(element)
        eval_basis!(bi, X, ip)  # Fills cache, no allocation
        # ... assembly logic
    end
end
```

**Validated:** All material models achieve 0 bytes allocation (Nov 10, 2025)

### 4. Multiple Dispatch

**Use Julia's type system:**

```julia
# Same function name, different implementations
assemble!(assembly, ::ElasticityPhysics, elements, time)
assemble!(assembly, ::HeatPhysics, elements, time)
assemble!(assembly, ::ContactPhysics, elements, time)

# Compiler generates specialized code for each!
```

**No runtime overhead, no vtables, just fast native code.**

---

## Common Questions

### Q: Why not use classes with methods?

**A:** Julia's multiple dispatch is more powerful than OOP:

```julia
# OOP way (single dispatch on first argument)
element.assemble(physics)  # Only element type matters

# Julia way (multiple dispatch on ALL arguments)
assemble!(assembly, physics, element, time)  # All types matter!
```

This enables:

- Compiler specialization on ALL argument types
- Adding new methods without modifying existing types
- True separation of concerns (no "god objects")

### Q: Why immutable structs?

**A:** Performance and safety:

- Structs with all concrete types are stack-allocated
- Immutability enables compiler optimizations
- No accidental mutation bugs

**Rule:** Use immutable structs unless you NEED mutability (like Assembly accumulation)

### Q: Why Tensors.jl instead of matrices?

**A:** Performance and clarity:

- `SymmetricTensor{2,3,Float64,6}` is **stack-allocated** (48 bytes on stack)
- Regular `Matrix{Float64}` is **heap-allocated** (pointer + malloc)
- Code looks like math: `σ = λ⋅tr(ε)⋅I + 2μ⋅ε`
- Type stability: Compiler knows exact size at compile time

---

## Summary: The JuliaFEM Way

**Core philosophy:**

1. **Separate concerns** - Each struct has ONE job
2. **Type stability** - Compiler knows ALL types
3. **Zero allocation** - Hot paths reuse memory
4. **Multiple dispatch** - Compiler specializes for each case
5. **Immutability** - Stack allocation + safety
6. **Tensors.jl** - Mathematical clarity + performance

**Result:** Fast, maintainable, extensible FEM code that looks like the math it implements.

---

## Further Reading

- **`docs/book/material_modeling.md`** - Material model implementation guide
- **`docs/book/elasticity_refactoring_plan.md`** - Elasticity system design
- **`docs/book/element_architecture.md`** - Element composition philosophy
- **`llm/ARCHITECTURE.md`** - Eight-layer system architecture
- **`llm/VISION_2.0.md`** - Project vision and philosophy

---

**Last Updated:** November 10, 2025  
**Status:** Authoritative (reflects current design decisions and implementation)
