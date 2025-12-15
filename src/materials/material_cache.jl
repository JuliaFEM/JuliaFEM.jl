# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
Material cache for storing state variables at integration points.

Uses the compositional state variable trait system to automatically construct
caches based on material requirements.

# Design Philosophy

1. **Automatic construction**: Cache structure inferred from material traits
2. **Type-stable**: NamedTuple-based storage with compile-time known types
3. **Zero-allocation**: Immutable state updates (functional style)
4. **Compositional**: State variables are independent, combinable building blocks

# Example

```julia
# For LinearElastic (stateless)
material = LinearElastic(E=210e9, ν=0.3)
cache = create_global_material_cache(material, n_ips=8, n_elems=100)
# → GlobalMaterialCache{NamedTuple{(),Tuple{}}} with empty states

# For PerfectPlasticity (stateful)
material = PerfectPlasticity(E=210e9, ν=0.3, σ_y=250e6, H=1e9)
cache = create_global_material_cache(material, n_ips=8, n_elems=100)
# → GlobalMaterialCache{NamedTuple{(:ε_p, :α, :κ), Tuple{...}}} with state variables
```
"""

using Tensors

# ============================================================================
# MATERIAL CACHE STRUCTURE
# ============================================================================

"""
    GlobalMaterialCache{StateType}

Global material state cache for all integration points across all elements.

Stores material state variables using the compositional state variable system.
Used for time-stepping and persistent state storage.

# Type Parameters
- `StateType`: NamedTuple type for state variables (inferred from material)

# Fields
- `states::Matrix{StateType}`: State at each IP and element [nips, nelems]
- `states_old::Matrix{StateType}`: State from previous time step [nips, nelems]

# Design

The cache uses NamedTuples to store state variables, where:
- Keys are state variable symbols (e.g., `:ε_p`, `:α`, `:κ`)
- Values are concrete types (e.g., `SymmetricTensor{2,3,Float64,6}`, `Float64`)

For stateless materials, `StateType = NamedTuple{(),Tuple{}}` (empty).

# Example

```julia
# Stateful material (plasticity)
StateType = NamedTuple{(:ε_p, :α, :κ), Tuple{SymmetricTensor{2,3,Float64,6}, SymmetricTensor{2,3,Float64,6}, Float64}}
cache = GlobalMaterialCache{StateType}(n_ips, n_elems)
cache.states[q, elem_id]  # → (ε_p=..., α=..., κ=...)

# Stateless material (linear elastic)
StateType = NamedTuple{(),Tuple{}}
cache = GlobalMaterialCache{StateType}(n_ips, n_elems)
cache.states[q, elem_id]  # → NamedTuple()
```
"""
struct GlobalMaterialCache{StateType<:NamedTuple}
    states::Matrix{StateType}
    states_old::Matrix{StateType}

    function GlobalMaterialCache{StateType}(n_ips::Int, n_elems::Int) where {StateType<:NamedTuple}
        # Initialize with zero/default states
        zero_state = create_zero_state(StateType)
        states = [zero_state for _ in 1:n_ips, _ in 1:n_elems]
        states_old = [zero_state for _ in 1:n_ips, _ in 1:n_elems]
        new{StateType}(states, states_old)
    end
end

"""
    create_zero_state(::Type{StateType}) where {StateType<:NamedTuple}

Create a zero-initialized state of the given NamedTuple type.

# Example
```julia
StateType = NamedTuple{(:ε_p, :α, :κ), Tuple{SymmetricTensor{2,3,Float64,6}, SymmetricTensor{2,3,Float64,6}, Float64}}
zero_state = create_zero_state(StateType)
# → (ε_p=zero(SymmetricTensor{2,3}), α=zero(SymmetricTensor{2,3}), κ=0.0)
```
"""
function create_zero_state(::Type{StateType}) where {StateType<:NamedTuple}
    # Get field names and types
    field_names = fieldnames(StateType)
    field_types = [fieldtype(StateType, name) for name in field_names]

    if isempty(field_names)
        return NamedTuple()
    else
        # Create zero value for each field
        zero_values = [zero(T) for T in field_types]
        return NamedTuple{field_names}(zero_values)
    end
end

# ============================================================================
# CACHE CONSTRUCTION FROM MATERIAL TRAITS
# ============================================================================

"""
    material_state_type(material::AbstractMaterial) -> Type{<:NamedTuple}

Infer state NamedTuple type from material's required state variables.

Uses the compositional trait system:
1. Query `required_state_variables(material)` → tuple of state variable types
2. For each state variable, query `state_variable_type(var)` → concrete type
3. For each state variable, query `default_symbol(var)` → symbol name
4. Construct NamedTuple type from symbols and concrete types

# Example
```julia
# LinearElastic (stateless)
material = LinearElastic(E=210e9, ν=0.3)
StateType = material_state_type(material)
# → NamedTuple{(),Tuple{}}

# PerfectPlasticity (stateful)
material = PerfectPlasticity(E=210e9, ν=0.3, σ_y=250e6, H=1e9)
StateType = material_state_type(material)
# → NamedTuple{(:ε_p, :α, :κ), Tuple{SymmetricTensor{2,3,Float64,6}, SymmetricTensor{2,3,Float64,6}, Float64}}
```
"""
function material_state_type(material::AbstractMaterial)
    # Query required state variables from material instance
    vars = required_state_variables(material)

    if isempty(vars)
        # Stateless material
        return NamedTuple{(),Tuple{}}
    else
        # Stateful material - build NamedTuple type
        symbols = tuple([default_symbol(var) for var in vars]...)
        types = tuple([state_variable_type(var) for var in vars]...)

        return NamedTuple{symbols, Tuple{types...}}
    end
end

"""
    create_global_material_cache(material::AbstractMaterial; n_ips::Int, n_elems::Int)

Create global material cache for the given material, integration points, and elements.

Automatically infers the correct cache structure from material traits.

# Arguments
- `material`: Material model instance
- `n_ips`: Number of integration points per element
- `n_elems`: Number of elements

# Returns
`GlobalMaterialCache{StateType}` where `StateType` is inferred from material

# Example
```julia
# Stateless material
mat = LinearElastic(E=210e9, ν=0.3)
cache = create_global_material_cache(mat, n_ips=8, n_elems=100)
# → GlobalMaterialCache{NamedTuple{(),Tuple{}}}

# Stateful material
mat = PerfectPlasticity(E=210e9, ν=0.3, σ_y=250e6, H=1e9)
cache = create_global_material_cache(mat, n_ips=8, n_elems=100)
# → GlobalMaterialCache{NamedTuple{(:ε_p, :α, :κ), Tuple{...}}}
```
"""
function create_global_material_cache(material::AbstractMaterial; n_ips::Int, n_elems::Int)
    StateType = material_state_type(material)
    return GlobalMaterialCache{StateType}(n_ips, n_elems)
end

# ============================================================================
# CACHE ACCESS AND UPDATE FUNCTIONS
# ============================================================================

"""
    get_state(cache::GlobalMaterialCache, ip::Int, elem_id::Int) -> NamedTuple

Get current state at integration point for element.

# Arguments
- `cache`: Global material cache
- `ip`: Integration point index (1-based)
- `elem_id`: Element ID (1-based)

# Returns
NamedTuple of state variables (empty for stateless materials)
"""
@inline function get_state(cache::GlobalMaterialCache, ip::Int, elem_id::Int)
    return cache.states[ip, elem_id]
end

"""
    get_old_state(cache::GlobalMaterialCache, ip::Int, elem_id::Int) -> NamedTuple

Get state from previous time step at integration point for element.

# Arguments
- `cache`: Global material cache
- `ip`: Integration point index (1-based)
- `elem_id`: Element ID (1-based)

# Returns
NamedTuple of state variables from previous step
"""
@inline function get_old_state(cache::GlobalMaterialCache, ip::Int, elem_id::Int)
    @inbounds return cache.states_old[ip, elem_id]
end

"""
    set_state!(cache::GlobalMaterialCache, ip::Int, elem_id::Int, state::NamedTuple)

Update current state at integration point for element (in-place).

# Arguments
- `cache`: Global material cache
- `ip`: Integration point index (1-based)
- `elem_id`: Element ID (1-based)
- `state`: New state NamedTuple

# Note
This is an in-place operation.
"""
@inline function set_state!(cache::GlobalMaterialCache, ip::Int, elem_id::Int, state::NamedTuple)
    @inbounds cache.states[ip, elem_id] = state
    return nothing
end

"""
    update_cache!(cache::GlobalMaterialCache)

Copy current states to states_old (for time stepping).

Call this at the beginning of each time/load step to save the converged
state from the previous step.

# Example
```julia
# At the start of a new time step
update_cache!(cache)

# Now solve for new states
for elem_id in 1:n_elems
    for ip in 1:n_ips
        state_old = get_old_state(cache, ip, elem_id)
        # ... solve for state_new ...
        set_state!(cache, ip, elem_id, state_new)
    end
end
```
"""
function update_cache!(cache::GlobalMaterialCache)
    cache.states_old .= cache.states
    return nothing
end

"""
    reset_cache!(cache::GlobalMaterialCache)

Reset all states to zero (for restarting analysis).

# Example
```julia
reset_cache!(cache)
# All states now zero-initialized
```
"""
function reset_cache!(cache::GlobalMaterialCache{StateType}) where {StateType}
    zero_state = create_zero_state(StateType)
    fill!(cache.states, zero_state)
    fill!(cache.states_old, zero_state)
    return nothing
end

# ============================================================================
# HELPER FUNCTIONS FOR STATE VARIABLE ACCESS
# ============================================================================

"""
    get_state_variable(state::NamedTuple, var_type::Type{<:AbstractStateVariable})

Extract a specific state variable from a state NamedTuple.

# Arguments
- `state`: State NamedTuple (e.g., from `get_state(cache, ip)`)
- `var_type`: State variable type (e.g., `PlasticStrain`)

# Returns
Value of the requested state variable

# Example
```julia
state = get_state(cache, 1)  # → (ε_p=..., α=..., κ=...)
ε_p = get_state_variable(state, PlasticStrain)
α = get_state_variable(state, Backstress)
κ = get_state_variable(state, EquivalentPlasticStrain)
```
"""
@generated function get_state_variable(state::StateType, var_type::Type{VarType}) where {StateType<:NamedTuple, VarType<:AbstractStateVariable}
    # Get the symbol for this state variable
    sym = default_symbol(VarType)

    # Check if state has this field
    if hasfield(StateType, sym)
        return :(state.$sym)
    else
        error("State type $StateType does not have field $sym for variable $VarType")
    end
end

"""
    set_state_variable(state::NamedTuple, var_type::Type{<:AbstractStateVariable}, value)

Create new state NamedTuple with updated variable value.

Returns a new NamedTuple (immutable update).

# Arguments
- `state`: Current state NamedTuple
- `var_type`: State variable type to update
- `value`: New value for the state variable

# Returns
New state NamedTuple with updated value

# Example
```julia
state = get_state(cache, 1)
new_state = set_state_variable(state, PlasticStrain, new_ε_p)
set_state!(cache, 1, new_state)
```
"""
@generated function set_state_variable(state::StateType, var_type::Type{VarType}, value) where {StateType<:NamedTuple, VarType<:AbstractStateVariable}
    # Get the symbol for this state variable
    sym = default_symbol(VarType)

    # Check if state has this field
    if hasfield(StateType, sym)
        return :(merge(state, NamedTuple{($(QuoteNode(sym)),)}((value,))))
    else
        error("State type $StateType does not have field $sym for variable $VarType")
    end
end
