# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
Core API Documentation - JuliaFEM Modular Architecture.

This file documents the modular API structure. All type definitions now live
in domain-specific api.jl files.

# Modular API Architecture

JuliaFEM follows a systematic modular design where each domain owns its abstractions
in a dedicated `api.jl` file:

## Domain API Files (in dependency order)

1. **src/formulations/api.jl** - Discretization strategies
   - AbstractFormulation - Base type for all formulations
   - ContinuumFormulation{Theory} - Standard FEM (FullThreeD, PlaneStress, PlaneStrain, Axisymmetric)

2. **src/fields/api.jl** - Field variables
   - AbstractField - Base type for all fields
   - Displacement{Dim}, Temperature, DisplacementRotation{Dim}
   - dofs_per_node() - DOF counting

3. **src/materials/api.jl** - Material models
   - AbstractMaterial - Base type for all materials
   - AbstractElasticMaterial, AbstractPlasticMaterial
   - compute_stress(), elasticity_tensor()

4. **src/mesh/api.jl** - Mesh structures
   - AbstractMesh - Base type for all meshes
   - nnodes_total(), nelements(), connectivity_matrix()
   - get_elements_for_node() - For nodal assembly

5. **src/beams/api.jl** - Beam formulations
   - AbstractBeamTheory - Beam theory variants
   - BeamFormulation{Theory} - Euler-Bernoulli, Timoshenko

6. **src/shells/api.jl** - Shell formulations
   - AbstractShellTheory - Shell theory variants
   - ShellFormulation{Theory} - Reissner-Mindlin, Kirchhoff-Love

7. **src/trusses/api.jl** - Truss formulations
   - AbstractTrussTheory - Truss theory variants
   - TrussFormulation{Theory} - SimpleTruss

8. **src/physics/api.jl** - Physics coupling
   - AbstractPhysics - Complete problem coupling
   - assemble!(), solve!(), add_dirichlet!(), add_neumann!()

9. **src/topology/api.jl** - Element topologies
   - AbstractTopology{N} - Element geometry (N = node count)
   - nnodes(), dim(), reference_coordinates(), edges(), faces()

# Design Philosophy

**Zero Duplication:**
Each abstract type is defined in exactly ONE place. No type appears in multiple api.jl files.

**Domain Ownership:**
Each domain owns its abstractions. Want to know about materials? Look in src/materials/api.jl.
Want to know about meshes? Look in src/mesh/api.jl.

**Minimal Core:**
This file (src/api.jl) contains NO type definitions - only documentation.
All abstractions live in their domain-specific api.jl files.

**Clear Boundaries:**
Each api.jl file documents:
- What types it defines
- What interfaces it declares
- What belongs in that domain
- What does NOT belong (and where to find it)

**Systematic Pattern:**
All domains follow identical structure:
- Abstract base types
- Interface function stubs
- Comprehensive documentation with examples
- Dispatch patterns explained

# Include Order (in src/JuliaFEM.jl)

```julia
# Core formulation abstractions
include("formulations/api.jl")

# Field types (formulations work with fields)
include("fields/api.jl")

# Material models
include("materials/api.jl")

# Mesh structures
include("mesh/api.jl")

# Structural formulations
include("beams/api.jl")
include("shells/api.jl")
include("trusses/api.jl")

# Physics coupling (brings it all together)
include("physics/api.jl")
include("physics.jl")  # Concrete implementations

# Topology (geometry abstractions)
include("topology/api.jl")
include("topology/topology.jl")  # Helpers
```

# Type Hierarchy Overview

```
AbstractFormulation (src/formulations/api.jl)
├── ContinuumFormulation{Theory}
│   ├── ContinuumFormulation{FullThreeD}
│   ├── ContinuumFormulation{PlaneStress}
│   ├── ContinuumFormulation{PlaneStrain}
│   └── ContinuumFormulation{Axisymmetric}
├── BeamFormulation{Theory} (src/beams/api.jl)
├── ShellFormulation{Theory} (src/shells/api.jl)
└── TrussFormulation{Theory} (src/trusses/api.jl)

AbstractField (src/fields/api.jl)
├── Displacement{Dim}
├── Temperature
└── DisplacementRotation{Dim}

AbstractMaterial (src/materials/api.jl)
├── AbstractElasticMaterial
│   ├── LinearElastic
│   └── NeoHookean
└── AbstractPlasticMaterial
    ├── J2Plasticity
    └── DruckerPrager

AbstractMesh (src/mesh/api.jl)
└── (Concrete mesh implementations)

AbstractTopology{N} (src/topology/api.jl)
├── Seg2, Seg3
├── Tri3, Tri6, Tri7
├── Quad4, Quad8, Quad9
├── Tet4, Tet10
├── Hex8, Hex20, Hex27
├── Pyr5
└── Wedge6, Wedge15

AbstractPhysics (src/physics/api.jl)
└── Physics{Formulation, Field, Mesh, Material}
```

# Assembly Dispatch Pattern

Specialized methods dispatch on formulation × field combinations:

```julia
# 3D continuum mechanics
function assemble!(physics::Physics{ContinuumFormulation{FullThreeD}, Displacement{3}, M, Mat})
    # Implementation in src/assembly/continuum_3d.jl
end

# 2D plane stress
function assemble!(physics::Physics{ContinuumFormulation{PlaneStress}, Displacement{2}, M, Mat})
    # Implementation in src/assembly/continuum_2d.jl
end

# Beam elements
function assemble!(physics::Physics{BeamFormulation{Timoshenko}, DisplacementRotation{3}, M, Mat})
    # Implementation in src/assembly/beams.jl
end

# Heat transfer
function assemble!(physics::Physics{ContinuumFormulation{FullThreeD}, Temperature, M, Mat})
    # Implementation in src/assembly/thermal.jl
end
```

# Advantages of Modular API

1. **No Duplication** - Each type defined once, referenced everywhere
2. **Clear Ownership** - Easy to find where types are defined
3. **Maintainability** - Changes localized to single domain
4. **Discoverability** - Systematic pattern (domain/api.jl)
5. **Documentation** - Each api.jl is self-contained
6. **Testing** - Can test each domain independently
7. **Extensibility** - Add new domains without touching core

# See Domain API Files

For concrete type definitions and interfaces, see the domain-specific api.jl files listed above.
This file is intentionally documentation-only to avoid duplication.
"""

# No type definitions here - all abstractions live in domain-specific api.jl files
