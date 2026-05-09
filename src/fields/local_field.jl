# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
Local field evaluation structure for material evaluation.

Provides comprehensive field data at a point including:
- Field values (T, u, p, etc.)
- Spatial gradients (∇T, ∇u, ∇p, etc.)
- Time derivatives (Ṫ, u̇, ṗ, etc.)
- Time derivatives of gradients (∇Ṫ, ∇u̇, ∇ṗ, etc.)
"""

using Tensors

"""
    LocalField{T, G, R, GR}

Field quantities evaluated at a single point (typically an integration point).

# Fields
- `value::T`: Field value (e.g., displacement `u`, temperature `T`)
- `gradient::G`: Spatial gradient (e.g., `∇u`, `∇T`)
- `rate::R`: Time derivative (e.g., velocity `u̇`, temperature rate `Ṫ`)
- `gradient_rate::GR`: Time derivative of gradient (e.g., `∇u̇`, `∇Ṫ`)

# Unified Dynamic and Quasi-Static Treatment

Quasi-static is treated as a special case of dynamic where `rate = 0`.
The `gradient_rate` is always computed from increments: `(∇u_new - ∇u_old)/Δt`

Dynamic simulations:
- `rate` = actual time derivative (velocity `u̇`, temperature rate `Ṫ`)
- `gradient_rate` = computed from increment for accuracy

Quasi-static simulations:
- `rate` = `0` (no dynamic effects)
- `gradient_rate` = computed from increment (needed for rate-dependent materials)

# Examples

```julia
# Dynamic case
u_local = LocalField(u_new, ∇u_new, u̇, ∇u_rate)

# Quasi-static case (rate = 0, but gradient_rate still computed)
u_local = LocalField(u_new, ∇u_new, zero(Vec{3}), ∇u_rate)

# Temperature field
T_local = LocalField(T, ∇T, Ṫ, ∇Ṫ)
```
"""
struct LocalField{T, G, R, GR}
    value::T           # Field value
    gradient::G        # Spatial gradient
    rate::R            # Time derivative
    gradient_rate::GR  # Time derivative of gradient
end

# Note: create_local_field_namedtuple() was removed because interpolate_local_fields()
# in src/elements/interpolate.jl directly creates LocalField objects from elements.
# The helper function was not needed for the actual implementation.
