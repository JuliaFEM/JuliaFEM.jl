# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
Field interpolation at quadrature points.

Given element DOFs and a point in reference coordinates, interpolate field
values, gradients, and rates. The four entry points (`interpolate_fields`,
`interpolate_field`, `interpolate_field_value`, `interpolate_local_fields`)
are all `@generated` functions; their per-field expansion is built from a
single small set of expression-level helpers defined at the top of this
file.

See `src/elements/README.md` for usage examples.
"""

using Tensors

# ---------------------------------------------------------------------------
# Compile-time expression helpers shared by every @generated entry point.
#
# These are *plain* functions that return `Expr` values. The `@generated`
# bodies below call them while the specialization is being constructed, so
# the produced expressions are inlined into the final method body and the
# helpers themselves never run at execution time.
# ---------------------------------------------------------------------------

"""
    _classify_field_quantity(Q) -> (kind::Symbol, vec_dim::Int)

Classify a quantity type `Q` (as produced by `quantity_type(field_spec)`).

- Returns `(:scalar, 1)` for `Float64`.
- Returns `(:vector, D)` for first-order `Tensor` types of dimension `D`
  (e.g. `Vec{3}`).

Throws an error for any other quantity type.
"""
function _classify_field_quantity(Q)
    if Q === Float64
        return (:scalar, 1)
    elseif Q isa UnionAll && Q.body <: Tensor && Q.body.parameters[1] == 1
        return (:vector, Q.body.parameters[2])
    else
        error("Unsupported quantity type: $Q")
    end
end

"""
    _per_node_dof_exprs(varname, offset, n_nodes, kind, vec_dim) -> Vector{Expr}

Build the per-node DOF expressions read out of `varname[elem.dof_indices[...]]`
for a field that starts at the given flat `offset`.

For a scalar field each entry is just `varname[elem.dof_indices[i]]`.
For a vector field of dimension `vec_dim` each entry packs the `vec_dim`
component reads into a `Vec{vec_dim}(...)` literal.
"""
function _per_node_dof_exprs(
    varname::Symbol, offset::Int, n_nodes::Int, kind::Symbol, vec_dim::Int,
)
    if kind === :scalar
        return [:($(varname)[elem.dof_indices[$(offset + i)]]) for i in 1:n_nodes]
    else  # :vector
        out = Vector{Expr}(undef, n_nodes)
        for node in 0:(n_nodes - 1)
            comps = [
                :($(varname)[elem.dof_indices[$(offset + node * vec_dim + comp)]])
                for comp in 1:vec_dim
            ]
            out[node + 1] = :(Vec{$vec_dim}($(Expr(:tuple, comps...))))
        end
        return out
    end
end

"""
    _value_expr(per_node_exprs, basis_var) -> Expr

Sum-of-products `∑ᵢ basis_var[i] * uᵢ` where `uᵢ` is the i-th expression
in `per_node_exprs`.
"""
function _value_expr(per_node_exprs::Vector{Expr}, basis_var::Symbol)
    terms = [:($(basis_var)[$i] * $(per_node_exprs[i])) for i in eachindex(per_node_exprs)]
    return Expr(:call, :+, terms...)
end

"""
    _grad_expr(per_node_exprs, dN_var, kind) -> Expr

Sum-of-products `∑ᵢ dN_var[i] op uᵢ`. The product operator depends on the
field kind: scalar fields use `*` (Vec × Float → Vec), vector fields use
`⊗` (Vec ⊗ Vec → Tensor{2}).
"""
function _grad_expr(per_node_exprs::Vector{Expr}, dN_var::Symbol, kind::Symbol)
    op = kind === :scalar ? :* : :⊗
    terms = [
        Expr(:call, op, :($(dN_var)[$i]), per_node_exprs[i])
        for i in eachindex(per_node_exprs)
    ]
    return Expr(:call, :+, terms...)
end

"""
    _field_dof_count(kind, n_nodes, vec_dim) -> Int

Number of flat DOF slots consumed by one field block.
"""
_field_dof_count(kind::Symbol, n_nodes::Int, vec_dim::Int) =
    kind === :scalar ? n_nodes : n_nodes * vec_dim

"""
    _classify_dofset_field(S, fname) -> (kind, vec_dim)

Read the quantity classification for the named field of a `DOFSet`. Also
asserts the field lives on `Vertex` entities, which is the only entity
type the interpolators currently support.
"""
function _classify_dofset_field(S, fname::Symbol)
    field_spec = fieldtype(S, fname)
    entity_type = field_spec.parameters[2]
    if entity_type !== Vertex
        error("Unsupported entity type: $entity_type (only Vertex supported for now)")
    end
    Q = quantity_type(field_spec)
    return _classify_field_quantity(Q)
end

# ---------------------------------------------------------------------------
# Generated entry points
# ---------------------------------------------------------------------------

"""
    interpolate_fields(elem::Element{K,P,S,N}, u_global::AbstractVector, ξ::Vec) → NamedTuple

Interpolate all fields and their gradients at reference point ξ.

Returns NamedTuple with field values and gradients:
- Scalar fields: `field => value::Float64, ∇field => gradient::Vec`
- Vector fields: `field => value::Vec, ∇field => gradient::Tensor{2}`

# Arguments
- `elem`: Element with field specification S
- `u_global`: Global solution vector
- `ξ`: Point in reference coordinates (e.g., `Vec((0.5, 0.5))` for 2D)

# Example
```julia
S = @DOFSet{T::DOF{Temperature,Vertex}, u::DOF{Displacement{3},Vertex}}
elem = Element{Tetrahedron{4}, Lagrange{1}, S}(UInt(1), (1,2,3,4,5,...,16))
u_global = rand(100)

# Interpolate at reference center
ξ = Vec((0.25, 0.25, 0.25))
vals = interpolate_fields(elem, u_global, ξ)
# vals = (T = 2.5, ∇T = Vec{3}(...), u = Vec{3}(...), ∇u = Tensor{2,3}(...))
```

# Performance
Zero-allocation @generated function. All field access and basis evaluation
happens at compile time.
"""
@generated function interpolate_fields(
    elem::Element{K,P,S,N},
    u_global::AbstractVector,
    ξ::Vec,
) where {K,P,S<:DOFSet,N}
    topology = K()
    basis = P()
    n_nodes = nnodes(topology)

    field_exprs = Expr[]
    offset = 0

    for fname in fieldnames(S)
        kind, vec_dim = _classify_dofset_field(S, fname)
        per_node = _per_node_dof_exprs(:u_global, offset, n_nodes, kind, vec_dim)

        push!(field_exprs, Expr(:(=), fname, _value_expr(per_node, :Nvals)))
        push!(field_exprs, Expr(:(=), Symbol("∇", fname), _grad_expr(per_node, :dN, kind)))

        offset += _field_dof_count(kind, n_nodes, vec_dim)
    end

    nt_expr = Expr(:tuple, field_exprs...)

    return quote
        @inbounds begin
            Nvals = get_basis_functions($topology, $basis, ξ)
            dN = get_basis_derivatives($topology, $basis, ξ)
            return $nt_expr
        end
    end
end

"""
    interpolate_field(elem::Element{K,P,S,N}, u_global::AbstractVector, field::Symbol, ξ::Vec) → Tuple{value, gradient}

Interpolate a single field and its gradient at reference point ξ.

More efficient than `interpolate_fields` when you only need one field.

# Returns
- For scalar fields: `(value::Float64, gradient::Vec)`
- For vector fields: `(value::Vec, gradient::Tensor{2})`

# Example
```julia
val, grad = interpolate_field(elem, u_global, :T, Vec((0.25, 0.25, 0.25)))
# val::Float64, grad::Vec{3}
```
"""
@generated function interpolate_field(
    elem::Element{K,P,S,N},
    u_global::AbstractVector,
    field::Symbol,
    ξ::Vec,
) where {K,P,S<:DOFSet,N}
    topology = K()
    basis = P()
    n_nodes = nnodes(topology)

    branches = Expr[]
    offset = 0

    for fname in fieldnames(S)
        kind, vec_dim = _classify_dofset_field(S, fname)
        per_node = _per_node_dof_exprs(:u_global, offset, n_nodes, kind, vec_dim)

        value_expr = _value_expr(per_node, :Nvals)
        grad_expr = _grad_expr(per_node, :dN, kind)

        push!(branches, quote
            if field === $(QuoteNode(fname))
                value = $value_expr
                grad = $grad_expr
                return (value, grad)
            end
        end)

        offset += _field_dof_count(kind, n_nodes, vec_dim)
    end

    push!(branches, :(error("Field ", field, " not found in element type $S")))

    return quote
        @inbounds begin
            Nvals = get_basis_functions($topology, $basis, ξ)
            dN = get_basis_derivatives($topology, $basis, ξ)
            $(branches...)
        end
    end
end

"""
    interpolate_field_value(elem::Element{K,P,S,D}, u_global::AbstractVector, field::Symbol, ξ::Vec) → value

Interpolate only field value (no gradient) at reference point ξ.

Most efficient when gradient is not needed.

# Example
```julia
T_val = interpolate_field_value(elem, u_global, :T, ξ)
u_val = interpolate_field_value(elem, u_global, :u, ξ)  # Returns Vec{3}
```
"""
@generated function interpolate_field_value(
    elem::Element{K,P,S,N},
    u_global::AbstractVector,
    field::Symbol,
    ξ::Vec,
) where {K,P,S<:DOFSet,N}
    topology = K()
    basis = P()
    n_nodes = nnodes(topology)

    branches = Expr[]
    offset = 0

    for fname in fieldnames(S)
        kind, vec_dim = _classify_dofset_field(S, fname)
        per_node = _per_node_dof_exprs(:u_global, offset, n_nodes, kind, vec_dim)

        value_expr = _value_expr(per_node, :Nvals)

        push!(branches, quote
            if field === $(QuoteNode(fname))
                return $value_expr
            end
        end)

        offset += _field_dof_count(kind, n_nodes, vec_dim)
    end

    push!(branches, :(error("Field ", field, " not found in element type $S")))

    return quote
        @inbounds begin
            Nvals = get_basis_functions($topology, $basis, ξ)
            $(branches...)
        end
    end
end

"""
    interpolate_local_fields(
        elem::Element{K,P,S,N},
        u_global::AbstractVector,
        u_old::AbstractVector,
        u_rate::AbstractVector,
        Δt::Float64,
        ξ::Vec
    ) → NamedTuple of LocalField

Interpolate all fields as LocalField structures at reference point ξ.

Returns a NamedTuple where each field is a LocalField containing:
- `value`: Current field value
- `gradient`: Current field gradient
- `rate`: Time derivative (from u_rate for dynamic, zero for quasi-static)
- `gradient_rate`: Time derivative of gradient (computed from increments)

# Arguments
- `elem`: Element with field specification S
- `u_global`: Current solution vector
- `u_old`: Previous time step solution vector
- `u_rate`: Rate DOFs (velocity for dynamic, zeros for quasi-static)
- `Δt`: Time step size
- `ξ`: Point in reference coordinates

# Unified Dynamic/Quasi-Static Treatment

Quasi-static:
```julia
local_fields = interpolate_local_fields(elem, u_new, u_old, zero(u_new), Δt, ξ)
# rate = 0, but gradient_rate computed from (∇u_new - ∇u_old)/Δt
```

Dynamic:
```julia
local_fields = interpolate_local_fields(elem, u_new, u_old, u_rate, Δt, ξ)
# rate = u̇, gradient_rate from increments (more accurate than ∇(u̇))
```

# Performance
Zero-allocation @generated function. All field access happens at compile time.
"""
@generated function interpolate_local_fields(
    elem::Element{K,P,S,N},
    u_global::AbstractVector,
    u_old::AbstractVector,
    u_rate::AbstractVector,
    Δt::Float64,
    ξ::Vec,
) where {K,P,S<:DOFSet,N}
    topology = K()
    basis = P()
    n_nodes = nnodes(topology)

    field_exprs = Expr[]
    offset = 0

    for fname in fieldnames(S)
        kind, vec_dim = _classify_dofset_field(S, fname)

        per_node_new = _per_node_dof_exprs(:u_global, offset, n_nodes, kind, vec_dim)
        per_node_old = _per_node_dof_exprs(:u_old, offset, n_nodes, kind, vec_dim)
        per_node_rate = _per_node_dof_exprs(:u_rate, offset, n_nodes, kind, vec_dim)

        value_expr = _value_expr(per_node_new, :Nvals)
        grad_expr = _grad_expr(per_node_new, :dN, kind)
        grad_old_expr = _grad_expr(per_node_old, :dN, kind)
        rate_expr = _value_expr(per_node_rate, :Nvals)
        grad_rate_expr = :(($grad_expr - $grad_old_expr) / Δt)

        local_field_expr = :(LocalField($value_expr, $grad_expr, $rate_expr, $grad_rate_expr))
        push!(field_exprs, Expr(:(=), fname, local_field_expr))

        offset += _field_dof_count(kind, n_nodes, vec_dim)
    end

    nt_expr = Expr(:tuple, field_exprs...)

    return quote
        @inbounds begin
            Nvals = get_basis_functions($topology, $basis, ξ)
            dN = get_basis_derivatives($topology, $basis, ξ)
            return $nt_expr
        end
    end
end
