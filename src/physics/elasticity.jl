# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
    ElasticityPhysics <: AbstractPhysics

Solid mechanics with geometric and material nonlinearity.

# Governing Equation

Strong form:
```
ρ₀ ∂²u/∂t² = ∇⋅σ + b₀   in Ω
u = ū                     on Γᵤ
σ⋅n = t̄                   on Γₜ
```

Weak form: Find u ∈ U such that ∀v ∈ V:
```
∫ ρ₀ ∂²u/∂t²⋅v dV + ∫ σ:∇ˢv dV = ∫ b₀⋅v dV + ∫ t̄⋅v dA
```

where:
- u = displacement field
- σ = stress tensor (from material model)
- b₀ = body force (per unit undeformed volume)
- t̄ = traction on boundary

# Formulations

Three geometric configurations:

- `:plane_stress` - 2D, σ₃₃ = 0 (thin structures)
- `:plane_strain` - 2D, ε₃₃ = 0 (long structures)
- `:continuum` - 3D general

# Material Nonlinearity

Material model computes: (ε, state_old, Δt) → (σ, 𝔻, state_new)

See `docs/book/material_modeling.md` for details.

# Geometric Nonlinearity

When `finite_strain = true`:
- Use Green-Lagrange strain: E = ½(∇u + ∇uᵀ + ∇uᵀ∇u)
- Update configuration each step
- Geometric stiffness from stress state

# Field Storage

Two types of field storage:

1. **Converged fields** (`store_fields`) - Always computed after convergence
   - Used for postprocessing, visualization
   - Efficient computation on GPU then transfer to host
   - Examples: `:stress`, `:strain`, `:plastic_strain`

2. **Iteration fields** (`store_iteration_fields`) - For debugging only
   - Stored DURING Newton iterations (before convergence)
   - Inefficient (lots of data), use sparingly
   - Examples: `:residual_norm`, `:trial_stress`

# GPU Design

All operations GPU-compatible:
- Type-stable (no Dict lookups)
- Zero allocation in hot paths
- Kernel-friendly (small element loops)
- Minimal host-device transfers

Workflow:
```
1. Transfer geometry + BCs to GPU
2. Run Newton iterations ON GPU
3. After convergence: compute postprocessing fields
4. Transfer results to host
```

# Example

```julia
# Linear elasticity
physics = ElasticityPhysics(
    formulation = :continuum,
    finite_strain = false,
    geometric_stiffness = false,
    store_fields = [:stress, :strain]
)

# Finite strain plasticity with debugging
physics = ElasticityPhysics(
    formulation = :continuum,
    finite_strain = true,
    geometric_stiffness = true,
    store_fields = [:stress, :strain, :plastic_strain],
    store_iteration_fields = [:residual_norm]  # For debugging convergence
)
```

# References

- Bathe, "Finite Element Procedures"
- Belytschko et al., "Nonlinear Finite Elements"
- Simo & Hughes, "Computational Inelasticity"
"""
struct ElasticityPhysics <: AbstractPhysics
    """Geometric formulation: `:plane_stress`, `:plane_strain`, `:continuum`"""
    formulation::Symbol

    """Use finite strain kinematics (Green-Lagrange strain)"""
    finite_strain::Bool

    """Include geometric stiffness (σ-dependent) for buckling analysis"""
    geometric_stiffness::Bool

    """Fields to store after convergence (postprocessing)"""
    store_fields::Vector{Symbol}

    """Fields to store during iterations (debugging only, inefficient!)"""
    store_iteration_fields::Vector{Symbol}

    # Inner constructor with validation
    function ElasticityPhysics(;
        formulation::Symbol=:continuum,
        finite_strain::Bool=false,
        geometric_stiffness::Bool=false,
        store_fields::Vector{Symbol}=Symbol[],
        store_iteration_fields::Vector{Symbol}=Symbol[]
    )
        # Validate formulation
        if !(formulation in [:plane_stress, :plane_strain, :continuum])
            error("Invalid formulation: $formulation. Must be :plane_stress, :plane_strain, or :continuum")
        end

        # Geometric stiffness only makes sense with finite strain
        if geometric_stiffness && !finite_strain
            @warn "geometric_stiffness=true but finite_strain=false. Geometric stiffness typically used with finite strain."
        end

        new(formulation, finite_strain, geometric_stiffness, store_fields, store_iteration_fields)
    end
end

# Convenience constructor for common case
function ElasticityPhysics()
    return ElasticityPhysics(formulation=:continuum, finite_strain=false, geometric_stiffness=false)
end

# Interface implementations
function get_unknown_field_name(::ElasticityPhysics)
    return "displacement"
end

function get_formulation_type(::ElasticityPhysics)
    return :incremental
end

function get_unknown_field_dimension(physics::ElasticityPhysics)
    if physics.formulation in [:plane_stress, :plane_strain]
        return 2  # 2D problem
    else
        return 3  # 3D problem
    end
end

"""
    should_store_field(physics::ElasticityPhysics, field::Symbol, converged::Bool) -> Bool

Check if a field should be stored at this point.

# Arguments
- `physics`: The physics object
- `field`: Field name to check
- `converged`: Whether Newton iteration has converged

# Returns
- `true` if field should be stored, `false` otherwise

# Logic

- **After convergence** (`converged=true`): Store if in `store_fields`
- **During iterations** (`converged=false`): Store if in `store_iteration_fields`

# Example

```julia
physics = ElasticityPhysics(
    store_fields = [:stress],
    store_iteration_fields = [:residual_norm]
)

should_store_field(physics, :stress, true)          # true (converged)
should_store_field(physics, :stress, false)         # false (not converged yet)
should_store_field(physics, :residual_norm, false)  # true (debugging)
should_store_field(physics, :residual_norm, true)   # false (only during iterations)
```
"""
function should_store_field(physics::ElasticityPhysics, field::Symbol, converged::Bool)
    if converged
        return field in physics.store_fields
    else
        return field in physics.store_iteration_fields
    end
end
