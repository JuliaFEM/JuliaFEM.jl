---
title: "GPU Elasticity Refactoring Plan"
date: 2025-11-10
author: "Jukka Aho"
status: "Active"
tags: ["design", "gpu", "elasticity", "refactoring"]
---

## Motivation

Current `src/gpu_elasticity.jl` (477 lines) doesn't follow JuliaFEM architecture:

❌ **Current Issues:**

- Custom `ElasticityPhysics` struct instead of `Problem{Elasticity}`
- Depends on `GmshMesh` (Gmsh-specific)
- BCs defined as vectors, not as `Problem{Dirichlet}` / `Problem{Neumann}`
- Elements don't store coordinates via `update!(element, "geometry", nodes)`
- Solver is monolithic function, not modular assembly

✅ **Target Architecture (JuliaFEM Convention):**

```julia
# Field problem
physics_elasticity = Problem(Elasticity, "body", 3)
add_elements!(physics_elasticity, body_elements)

# Boundary conditions
bc_fixed = Problem(Dirichlet, "fixed end", 3, "displacement")
add_elements!(bc_fixed, boundary_elements)

bc_pressure = Problem(Neumann, "pressure load", 3, "displacement")  
add_elements!(bc_pressure, surface_elements)

# Solve
solver = Solver(GPU)
add_problems!(solver, [physics_elasticity, bc_fixed, bc_pressure])
solve!(solver)
```

## Architecture Comparison

### Old (current gpu_elasticity.jl)

```julia
struct ElasticityPhysics
    mesh::GmshMesh                  # ❌ Gmsh-specific
    material::ElasticMaterial       # ❌ Not Element property
    fixed_nodes::Vector{Int}        # ❌ Should be Problem{Dirichlet}
    pressure_nodes::Vector{Int}     # ❌ Should be Problem{Neumann}
    pressure_value::Float64
end

# Usage
physics = ElasticityPhysics(mesh, material, fixed_nodes, pressure_nodes, 1e6)
result = solve_elasticity_gpu(physics)  # Monolithic
```

### New (JuliaFEM convention)

```julia
# 1. Create elements with geometry (immutable API)
body = Element(Tet4, Lagrange{Tet4, 1}, [1,2,3,4];
               fields=(geometry = nodes,
                      youngs_modulus = 210e9,
                      poissons_ratio = 0.3))

# 2. Create field problem
physics = Problem(Elasticity, "body", 3)
add_elements!(physics, [body])

# 3. Create boundary conditions (separate problems!)
bc_fixed = Problem(Dirichlet, "fixed", 3, "displacement")
fixed_element = Element(Tri3, Lagrange{Tri3, 1}, [1,2,3];
                       fields=(geometry = nodes,
                              displacement_1 = 0.0,
                              displacement_2 = 0.0,
                              displacement_3 = 0.0))
add_elements!(bc_fixed, [fixed_element])

bc_pressure = Problem(Neumann, "pressure", 3, "displacement")
pressure_element = Element(Tri3, Lagrange{Tri3, 1}, [4,5,6];
                          fields=(geometry = nodes,
                                 displacement_traction_force_3 = -1e6))
add_elements!(bc_pressure, [pressure_element])

# 4. Solve with GPU
solver = Solver(GPU, Linear)
add_problems!(solver, [physics, bc_fixed, bc_pressure])
solve!(solver, time)
```

## Key Design Principles

### 1. Elements Store Coordinates

**JuliaFEM Convention:**

```julia
# Element creation with immutable fields
el = Element(Tet4, Lagrange{Tet4, 1}, [1, 2, 3, 4];
             fields=(geometry = nodes,
                    youngs_modulus = 210e9,
                    poissons_ratio = 0.3))

# Access during assembly
X = el.fields.geometry  # Returns 3×4 matrix
E = el.fields.youngs_modulus
```

**Why:** Immutable, type-stable (40-130× faster), GPU-compatible.

### 2. Problem{T} for Physics Types

**JuliaFEM Hierarchy:**

```julia
AbstractProblem
  ├─ FieldProblem (volume elements)
  │   ├─ Elasticity
  │   ├─ Heat
  │   └─ Truss
  └─ BoundaryProblem (surface elements)
      ├─ Dirichlet
      ├─ Neumann (implicit, via "traction force" field)
      └─ Mortar (contact)
```

**Pattern:**

```julia
mutable struct Elasticity <: FieldProblem
    formulation::Symbol         # :plane_stress, :plane_strain, :continuum
    finite_strain::Bool
    geometric_stiffness::Bool
    store_fields::Vector{Symbol}
end

# Problem wraps physics type
mutable struct Problem{P<:AbstractProblem}
    name::String
    dimension::Int                          # DOFs per node
    parent_field_name::String              # For BCs: "displacement"
    elements::Vector{Element}
    dofmap::Dict{Element, Vector{Int}}
    assembly::Assembly                      # K, f, C, g matrices
    fields::Dict{String, AbstractField}
    properties::P                           # Elasticity instance
end
```

### 3. Boundary Conditions as Problems

**Dirichlet (fixed displacement):**

```julia
bc = Problem(Dirichlet, "fixed end", 3, "displacement")

# Surface element on boundary (immutable)
fixed_surf = Element(Tri3, Lagrange{Tri3, 1}, [1, 2, 3];
                    fields=(geometry = nodes,
                           displacement_1 = 0.0,
                           displacement_2 = 0.0,
                           displacement_3 = 0.0))

add_elements!(bc, [fixed_surf])
```

**Neumann (traction/pressure):**

```julia
# Surface element with traction (immutable)
pressure_surf = Element(Tri3, Lagrange{Tri3, 1}, [4, 5, 6];
                       fields=(geometry = nodes,
                              displacement_traction_force_3 = -1e6))

# Add to physics problem directly
add_elements!(physics, [pressure_surf])
```

### 4. Material Properties on Elements

```julia
# Material defined per element (immutable - set at construction)
steel_el = Element(Tet4, Lagrange{Tet4, 1}, conn;
                   fields=(geometry = X,
                          youngs_modulus = 210e9,
                          poissons_ratio = 0.3))

aluminum_el = Element(Tet4, Lagrange{Tet4, 1}, conn;
                     fields=(geometry = X,
                            youngs_modulus = 69e9,
                            poissons_ratio = 0.33))

# Heterogeneous materials: create different elements with different properties
```

## Implementation Plan

### Phase 1: Element-Based Storage ✅ (Current Session)

**Goal:** Remove `GmshMesh` dependency, use Element fields with immutable API.

**Changes:**

```julia
# OLD
struct ElasticityPhysics
    mesh::GmshMesh  # ❌
    ...
end

# NEW - Immutable elements with fields at construction
body = Element(Tet4, Lagrange{Tet4, 1}, [1,2,3,4];
               fields=(geometry = X_nodes,
                      youngs_modulus = E,
                      poissons_ratio = ν))
```

**File:** `src/gpu_physics_elasticity.jl` (new immutable-based implementation)

**Tasks:**

1. Remove `GmshMesh`, `ElasticityPhysics`, `ElasticMaterial` structs
2. Accept `Vector{Element}` instead of mesh
3. Extract coordinates from `element.fields.geometry`
4. Extract material from `element.fields.youngs_modulus`
5. Update demo to use immutable element creation

### Phase 2: Problem{Elasticity} Integration

**Goal:** Replace custom solver with `Problem{Elasticity}` pattern.

**Changes:**

```julia
# Create problem
physics = Problem(Elasticity, "continuum", 3)
physics.properties.formulation = :continuum
physics.properties.finite_strain = false

# Add elements
for el in body_elements
    add_elements!(physics, el)
end

# Assembly (GPU kernel)
assembly = Assembly()
assemble!(assembly, physics, physics.elements, time)
```

**File:** Extend `src/problems_elasticity.jl` with GPU dispatch

**Tasks:**

1. Add `assemble!(::Assembly, ::Problem{Elasticity}, ::Vector{Element}, time, ::Val{:gpu})`
2. Reuse GPU kernels from Phase 1
3. Keep CPU version untouched (backward compatibility)
4. Support both CPU and GPU via dispatch

### Phase 3: Boundary Conditions as Problems

**Goal:** Replace node vectors with `Problem{Dirichlet}` using immutable elements.

**Changes:**

```julia
# OLD
fixed_nodes = [1, 5, 12, ...]
pressure_nodes = [3, 7, 9, ...]

# NEW - Immutable boundary elements
bc_fixed = Problem(Dirichlet, "fixed end", 3, "displacement")
bc_pressure = Problem(Neumann, "pressure", 3, "displacement")

# Add boundary elements
for node in fixed_node_ids
    bc_el = Element(Poi1, Lagrange{Poi1, 1}, [node];
                   fields=(geometry = nodes[:, node:node],
                          displacement_1 = 0.0,
                          displacement_2 = 0.0,
                          displacement_3 = 0.0))
    add_elements!(bc_fixed, bc_el)
end
```

**File:** `src/problems_dirichlet.jl` (GPU dispatch)

**Tasks:**

1. Add GPU assembly for `Problem{Dirichlet}`
2. Integrate with GPU solver
3. Neumann BC: Add `Problem{Neumann}` type (future)

### Phase 4: Solver Integration

**Goal:** Unified GPU/CPU solver interface.

**Changes:**

```julia
# Create solver
solver = Solver(GPU, Linear)
add_problems!(solver, physics, bc_fixed)

# Solve
solver(0.0)  # Assemble at time=0
u = solver.assembly.u  # Solution
```

**File:** `src/solvers_gpu.jl` (new)

**Tasks:**

1. Create `Solver` struct with GPU field
2. Dispatch `assemble!()` to GPU kernels
3. Matrix-free CG on GPU
4. Export results to `assembly.u`

## Migration Path (Backward Compatibility)

### Keep Old API Working

```julia
# OLD API (deprecated but functional)
physics = ElasticityPhysics(mesh, material, fixed, pressure, 1e6)
result = solve_elasticity_gpu(physics)

# NEW API (preferred)
physics = Problem(Elasticity, "body", 3)
bc = Problem(Dirichlet, "fixed", 3, "displacement")
solver = Solver(GPU, Linear)
solve!(solver)
```

**Strategy:**

1. Keep `gpu_elasticity.jl` as `gpu_elasticity_legacy.jl`
2. Create new `gpu_elasticity_v2.jl` with Problem pattern
3. Add deprecation warnings to old API
4. Update demos to new pattern
5. Remove legacy after 2-3 versions

## Benefits of Refactoring

### Architectural

✅ **Follows JuliaFEM conventions** - Consistent with CPU solver  
✅ **Mesh-agnostic** - Works with any mesh format  
✅ **Modular** - Field + Boundary problems separate  
✅ **Extensible** - Easy to add Neumann, Mortar, etc.

### Practical

✅ **Heterogeneous materials** - Per-element properties  
✅ **Multiple BC types** - Dirichlet, Neumann, point loads  
✅ **Reusable components** - GPU kernels work for nonlinear too  
✅ **Testing** - Use existing test infrastructure

### Performance

✅ **Zero allocation** - Element fields are pre-allocated  
✅ **Type-stable** - Material access via Element, not Dict  
✅ **GPU-friendly** - Nodal assembly pattern unchanged

## Example: Cantilever Beam (New API)

```julia
using JuliaFEM

# 1. Create mesh (any format)
nodes = [...]  # 3×n_nodes matrix
connectivity = [...]  # 4×n_elements (Tet4)

# 2. Create body elements (immutable API)
body_elements = Element[]
for i in 1:n_elements
    conn = connectivity[:, i]
    X = nodes[:, conn]
    
    # Immutable: all fields at construction
    el = Element(Tet4, Lagrange{Tet4, 1}, conn;
                 fields=(geometry = X,
                        youngs_modulus = 210e9,
                        poissons_ratio = 0.3))
    
    push!(body_elements, el)
end

# 3. Create field problem
physics = Problem(Elasticity, "cantilever", 3)
physics.properties.formulation = :continuum
add_elements!(physics, body_elements)

# 4. Fixed end (Dirichlet BC - immutable elements)
bc_fixed = Problem(Dirichlet, "fixed end", 3, "displacement")
fixed_node_ids = [1, 2, 5, 9, ...]  # X=0 plane

for node_id in fixed_node_ids
    X_node = nodes[:, node_id:node_id]
    bc_el = Element(Poi1, Lagrange{Poi1, 1}, [node_id];
                   fields=(geometry = X_node,
                          displacement_1 = 0.0,
                          displacement_2 = 0.0,
                          displacement_3 = 0.0))
    add_elements!(bc_fixed, bc_el)
end

# 5. Pressure load (surface traction - immutable)
pressure_element_ids = [...]  # Top surface Tri3 elements
for el_id in pressure_element_ids
    conn = surface_connectivity[:, el_id]
    X = nodes[:, conn]
    el = Element(Tri3, Lagrange{Tri3, 1}, conn;
                 fields=(geometry = X,
                        displacement_traction_force_3 = -1e6))
    add_elements!(physics, el)
end

# 6. Solve with GPU
solver = Solver(GPU, Linear)
add_problems!(solver, physics, bc_fixed)
solver(0.0)  # Assemble and solve at time=0

# 7. Results
u = solver.assembly.u
println("Max displacement: ", maximum(abs.(u)))
```

## Status

- **Phase 1:** ✅ DESIGN COMPLETE (this document)
- **Phase 2:** 🔄 IN PROGRESS (implementing element-based storage)
- **Phase 3:** ⏳ PENDING (Problem{Elasticity} integration)
- **Phase 4:** ⏳ PENDING (BC as Problems)
- **Phase 5:** ⏳ PENDING (Solver integration)

## Next Steps

1. **Create `src/gpu_physics_elasticity.jl`** with immutable element API (no GmshMesh)
2. **Update demos** to use immutable element creation with fields at construction
3. **Test** that GPU kernels work with `element.fields.geometry` access
4. **Integrate** with `Problem{Elasticity}` pattern
5. **Add** `Problem{Dirichlet}` GPU assembly
6. **Document** new immutable API in user manual

## References

- `src/problems_elasticity.jl` - CPU elasticity implementation
- `src/problems_dirichlet.jl` - Dirichlet BC pattern
- `src/assembly/problems.jl` - Problem definition
- `test/test_elasticity_*.jl` - Example usage patterns
