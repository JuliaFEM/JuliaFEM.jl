# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
    AbstractPhysics

Base type for all physics implementations in JuliaFEM.

Physics objects serve as:
1. **Dispatch tags** - Select correct assembly method via multiple dispatch
2. **Configuration holders** - Store physics-specific options
3. **Field name providers** - Define primary field ("displacement", "temperature", etc.)

# Design Philosophy

Each physics type represents a specific set of governing equations:
- `ElasticityPhysics` → ∇⋅σ = ρü + b
- `HeatPhysics` → ∇⋅(k∇T) = ρcₚ∂T/∂t + Q
- `ContactPhysics` → Contact constraints and friction
- etc.

# Multi-Physics Coupling (Future)

The design supports coupled physics via composition:

```julia
# Future: Thermo-mechanical coupling
coupled = CoupledPhysics(
    ElasticityPhysics(...),
    HeatPhysics(...)
)

# Solver handles coupling automatically
assemble!(assembly, coupled, elements, time)
```

This enables:
- Sequential coupling (operator splitting)
- Monolithic coupling (solve simultaneously)
- Staggered schemes (iterative coupling)

# GPU Compatibility

All physics implementations must be GPU-friendly:
- ✅ Type-stable (no Dict lookups)
- ✅ Zero allocation in hot paths
- ✅ Kernel-compatible functions
- ✅ Minimal host-device transfers

Iterations run entirely on GPU. Only after convergence do we transfer results
to host for postprocessing.

# See Also

- `docs/user/system_architecture.md` - Design rationale
- `docs/book/elasticity_refactoring_plan.md` - Implementation details
"""
abstract type AbstractPhysics end

"""
    get_unknown_field_name(physics::AbstractPhysics) -> String

Return the name of the primary unknown field for this physics.

# Examples

```julia
get_unknown_field_name(ElasticityPhysics()) # "displacement"
get_unknown_field_name(HeatPhysics())       # "temperature"
get_unknown_field_name(FluidPhysics())      # "velocity"
```
"""
function get_unknown_field_name(physics::AbstractPhysics)
    error("get_unknown_field_name not implemented for $(typeof(physics))")
end

"""
    get_formulation_type(physics::AbstractPhysics) -> Symbol

Return the formulation type: `:incremental`, `:total`, or `:rate`.

- `:incremental` - Solve for Δu, update u ← u + Δu (elasticity)
- `:total` - Solve for u directly (Poisson equation)
- `:rate` - Solve for ∂u/∂t (transient heat, fluid dynamics)
"""
function get_formulation_type(physics::AbstractPhysics)
    error("get_formulation_type not implemented for $(typeof(physics))")
end

"""
    get_unknown_field_dimension(physics::AbstractPhysics) -> Int

Return the dimension of the unknown field (DOFs per node).

# Examples

```julia
get_unknown_field_dimension(ElasticityPhysics()) # 3 (3D displacement)
get_unknown_field_dimension(HeatPhysics())       # 1 (scalar temperature)
get_unknown_field_dimension(FluidPhysics())      # 4 (velocity + pressure)
```
"""
function get_unknown_field_dimension(physics::AbstractPhysics)
    error("get_unknown_field_dimension not implemented for $(typeof(physics))")
end

"""
    assemble!(assembly::Assembly, physics::AbstractPhysics, elements, time)

Assemble global system for given physics.

This is the main dispatch point for physics-specific assembly. Each physics
type implements its own method.

# GPU-Friendly Design

The assembly loop must be GPU-compatible:

```julia
# CPU version (reference)
for element in elements
    Kₑ, fₑ = assemble_element(physics, element, time)
    add_to_global!(assembly, Kₑ, fₑ, element.gdofs)
end

# GPU version (future)
@cuda threads=256 assemble_kernel!(assembly, physics, elements, time)
```

Key principles:
- No heap allocations inside element loop
- All buffers pre-allocated
- Type-stable throughout
- Deterministic execution order (for GPU atomics)
"""
function assemble!(assembly, physics::AbstractPhysics, elements, time)
    error("assemble! not implemented for $(typeof(physics))")
end
