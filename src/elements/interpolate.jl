# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
Field interpolation at quadrature points.

Given element DOFs and a point in reference coordinates, interpolate field values
and gradients. Returns a NamedTuple with interpolated quantities.

See `src/elements/README.md` for usage examples.
"""

using Tensors

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
    ξ::Vec
) where {K,P,S<:DOFSet,N}
    field_names = fieldnames(S)
    topology = K()
    basis = P()
    n_nodes = nnodes(topology)
    
    # Build expressions for each field interpolation
    field_exprs = Expr[]
    offset = 0  # Track position in flat dof_indices tuple
    
    for fname in field_names
        field_spec = fieldtype(S, fname)
        field_type = field_spec.parameters[1]  # Displacement{3}
        entity_type = field_spec.parameters[2]
        
        # Extract quantity type via trait
        Q = quantity_type(field_spec)  # Vec{3} or Float64
        
        if entity_type === Vertex
            # Standard nodal basis
            if Q === Float64
                # Scalar field interpolation
                # value = ∑ Nᵢ(ξ) * uᵢ
                # gradient = ∑ ∇Nᵢ(ξ) * uᵢ
                
                value_terms = Expr[]
                grad_terms = Expr[]
                
                for i in 1:n_nodes
                    push!(value_terms, :(Nvals[$i] * u_global[elem.dof_indices[$(offset+i)]]))
                    push!(grad_terms, :(dN[$i] * u_global[elem.dof_indices[$(offset+i)]]))
                end
                
                value_expr = Expr(:call, :+, value_terms...)
                grad_expr = Expr(:call, :+, grad_terms...)
                
                # Add field value and gradient
                push!(field_exprs, Expr(:(=), fname, value_expr))
                push!(field_exprs, Expr(:(=), Symbol("∇", fname), grad_expr))
                
                offset += n_nodes
                
            elseif Q isa UnionAll && Q.body <: Tensor && Q.body.parameters[1] == 1
                # Vector field interpolation
                # value = ∑ Nᵢ(ξ) * uᵢ  (each uᵢ is a Vec)
                # gradient = ∑ ∇Nᵢ(ξ) ⊗ uᵢ  (tensor product)
                
                vec_dim = Q.body.parameters[2]
                
                value_terms = Expr[]
                grad_terms = Expr[]
                
                for node in 0:(n_nodes-1)
                    # Extract vector components for this node from flat tuple
                    vec_comps = [:(u_global[elem.dof_indices[$(offset+node*vec_dim+comp)]]) for comp in 1:vec_dim]
                    u_node = :(Vec{$vec_dim}($(Expr(:tuple, vec_comps...))))
                    
                    node_idx = node + 1
                    # value += N_i * u_i
                    push!(value_terms, :(Nvals[$node_idx] * $u_node))
                    # gradient += ∇N_i ⊗ u_i
                    push!(grad_terms, :(dN[$node_idx] ⊗ $u_node))
                end
                
                value_expr = Expr(:call, :+, value_terms...)
                grad_expr = Expr(:call, :+, grad_terms...)
                
                # Add field value and gradient
                push!(field_exprs, Expr(:(=), fname, value_expr))
                push!(field_exprs, Expr(:(=), Symbol("∇", fname), grad_expr))
                
                offset += n_nodes * vec_dim
            else
                error("Unsupported quantity type: $Q")
            end
        else
            error("Unsupported entity type: $entity_type (only Vertex supported for now)")
        end
    end
    
    # Build complete function body
    # 1. Evaluate basis functions and derivatives
    # 2. Compute all interpolations
    # 3. Return NamedTuple
    
    nt_expr = Expr(:tuple, field_exprs...)
    
    return quote
        @inbounds begin
            # Evaluate basis functions once
            Nvals = get_basis_functions($topology, $basis, ξ)
            dN = get_basis_derivatives($topology, $basis, ξ)
            
            # Return interpolated values
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
    ξ::Vec
) where {K,P,S<:DOFSet,N}
    field_names = fieldnames(S)
    topology = K()
    basis = P()
    n_nodes = nnodes(topology)
    
    # Generate separate branches for each field
    branches = Expr[]
    offset = 0
    
    for fname in field_names
        field_spec = fieldtype(S, fname)
        field_type = field_spec.parameters[1]  # Displacement{3}
        entity_type = field_spec.parameters[2]
        
        # Extract quantity type via trait
        Q = quantity_type(field_spec)  # Vec{3} or Float64
        
        if entity_type === Vertex
            if Q === Float64
                # Scalar field
                value_terms = Expr[]
                grad_terms = Expr[]
                
                for i in 1:n_nodes
                    push!(value_terms, :(Nvals[$i] * u_global[elem.dof_indices[$(offset+i)]]))
                    push!(grad_terms, :(dN[$i] * u_global[elem.dof_indices[$(offset+i)]]))
                end
                
                value_expr = Expr(:call, :+, value_terms...)
                grad_expr = Expr(:call, :+, grad_terms...)
                
                push!(branches, quote
                    if field === $(QuoteNode(fname))
                        value = $value_expr
                        grad = $grad_expr
                        return (value, grad)
                    end
                end)
                
                offset += n_nodes
                
            elseif Q isa UnionAll && Q.body <: Tensor && Q.body.parameters[1] == 1
                # Vector field
                vec_dim = Q.body.parameters[2]
                
                value_terms = Expr[]
                grad_terms = Expr[]
                
                for node in 0:(n_nodes-1)
                    vec_comps = [:(u_global[elem.dof_indices[$(offset+node*vec_dim+comp)]]) for comp in 1:vec_dim]
                    u_node = :(Vec{$vec_dim}($(Expr(:tuple, vec_comps...))))
                    
                    node_idx = node + 1
                    push!(value_terms, :(Nvals[$node_idx] * $u_node))
                    push!(grad_terms, :(dN[$node_idx] ⊗ $u_node))
                end
                
                value_expr = Expr(:call, :+, value_terms...)
                grad_expr = Expr(:call, :+, grad_terms...)
                
                push!(branches, quote
                    if field === $(QuoteNode(fname))
                        value = $value_expr
                        grad = $grad_expr
                        return (value, grad)
                    end
                end)
                
                offset += n_nodes * vec_dim
            end
        end
    end
    
    # Add error case
    push!(branches, :(error("Field ", field, " not found in element type $S")))
    
    # Build complete function
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
    ξ::Vec
) where {K,P,S<:DOFSet,N}
    field_names = fieldnames(S)
    topology = K()
    basis = P()
    n_nodes = nnodes(topology)

    branches = Expr[]
    offset = 0

    for fname in field_names
        field_spec = fieldtype(S, fname)
        field_type = field_spec.parameters[1]  # Displacement{3}
        entity_type = field_spec.parameters[2]

        # Extract quantity type via trait
        Q = quantity_type(field_spec)  # Vec{3} or Float64

        if entity_type === Vertex
            if Q === Float64
                value_terms = Expr[]
                for i in 1:n_nodes
                    push!(value_terms, :(Nvals[$i] * u_global[elem.dof_indices[$(offset+i)]]))
                end
                value_expr = Expr(:call, :+, value_terms...)

                push!(branches, quote
                    if field === $(QuoteNode(fname))
                        return $value_expr
                    end
                end)

                offset += n_nodes

            elseif quantity_type isa UnionAll && quantity_type.body <: Tensor && quantity_type.body.parameters[1] == 1
                vec_dim = quantity_type.body.parameters[2]
                value_terms = Expr[]

                for node in 0:(n_nodes-1)
                    vec_comps = [:(u_global[elem.dof_indices[$(offset+node*vec_dim+comp)]]) for comp in 1:vec_dim]
                    u_node = :(Vec{$vec_dim}($(Expr(:tuple, vec_comps...))))
                    node_idx = node + 1
                    push!(value_terms, :(Nvals[$node_idx] * $u_node))
                end
                value_expr = Expr(:call, :+, value_terms...)

                push!(branches, quote
                    if field === $(QuoteNode(fname))
                        return $value_expr
                    end
                end)

                offset += n_nodes * vec_dim
            end
        end
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

**Quasi-static:**
```julia
local_fields = interpolate_local_fields(elem, u_new, u_old, zero(u_new), Δt, ξ)
# rate = 0, but gradient_rate computed from (∇u_new - ∇u_old)/Δt
```

**Dynamic:**
```julia
local_fields = interpolate_local_fields(elem, u_new, u_old, u_rate, Δt, ξ)
# rate = u̇, gradient_rate from increments (more accurate than ∇(u̇))
```

# Example
```julia
S = @DOFSet{u::DOF{Displacement{3},Vertex}}
elem = Element{Tetrahedron, Lagrange{Tetrahedron,1}, S}(...)

# Quasi-static loading
u_new = [...]  # Current configuration
u_old = [...]  # Previous load step
Δt = 1.0
ξ = Vec((0.25, 0.25, 0.25))

local_fields = interpolate_local_fields(elem, u_new, u_old, zero(u_new), Δt, ξ)
# → (u = LocalField(u_val, ∇u, zero(Vec{3}), ∇u_rate), ...)

# Extract strain for material evaluation
ε = extract_strain(local_fields.u.gradient)
ε̇ = extract_strain_rate(local_fields.u.gradient_rate)
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
    ξ::Vec
) where {K,P,S<:DOFSet,N}
    field_names = fieldnames(S)
    topology = K()
    basis = P()
    n_nodes = nnodes(topology)

    # Build expressions for LocalField creation for each field
    field_exprs = Expr[]
    offset = 0

    for fname in field_names
        field_spec = fieldtype(S, fname)
        field_type = field_spec.parameters[1]  # Displacement{3}
        entity_type = field_spec.parameters[2]

        # Extract quantity type via trait
        Q = quantity_type(field_spec)  # Vec{3} or Float64

        if entity_type === Vertex
            if Q === Float64
                # Scalar field interpolation
                value_terms = Expr[]
                grad_terms = Expr[]
                value_old_terms = Expr[]
                grad_old_terms = Expr[]
                rate_terms = Expr[]

                for i in 1:n_nodes
                    idx = offset + i
                    # Current value and gradient
                    push!(value_terms, :(Nvals[$i] * u_global[elem.dof_indices[$idx]]))
                    push!(grad_terms, :(dN[$i] * u_global[elem.dof_indices[$idx]]))
                    # Old value and gradient (for gradient_rate)
                    push!(value_old_terms, :(Nvals[$i] * u_old[elem.dof_indices[$idx]]))
                    push!(grad_old_terms, :(dN[$i] * u_old[elem.dof_indices[$idx]]))
                    # Rate
                    push!(rate_terms, :(Nvals[$i] * u_rate[elem.dof_indices[$idx]]))
                end

                value_expr = Expr(:call, :+, value_terms...)
                grad_expr = Expr(:call, :+, grad_terms...)
                grad_old_expr = Expr(:call, :+, grad_old_terms...)
                rate_expr = Expr(:call, :+, rate_terms...)

                # Gradient rate from increment
                grad_rate_expr = :(($grad_expr - $grad_old_expr) / Δt)

                # Create LocalField
                local_field_expr = :(LocalField($value_expr, $grad_expr, $rate_expr, $grad_rate_expr))
                push!(field_exprs, Expr(:(=), fname, local_field_expr))

                offset += n_nodes

            elseif Q isa UnionAll && Q.body <: Tensor && Q.body.parameters[1] == 1
                # Vector field interpolation
                vec_dim = Q.body.parameters[2]

                value_terms = Expr[]
                grad_terms = Expr[]
                value_old_terms = Expr[]
                grad_old_terms = Expr[]
                rate_terms = Expr[]

                for node in 0:(n_nodes-1)
                    node_idx = node + 1

                    # Current values
                    vec_comps = [:(u_global[elem.dof_indices[$(offset+node*vec_dim+comp)]]) for comp in 1:vec_dim]
                    u_node = :(Vec{$vec_dim}($(Expr(:tuple, vec_comps...))))
                    push!(value_terms, :(Nvals[$node_idx] * $u_node))
                    push!(grad_terms, :(dN[$node_idx] ⊗ $u_node))

                    # Old values (for gradient_rate)
                    vec_comps_old = [:(u_old[elem.dof_indices[$(offset+node*vec_dim+comp)]]) for comp in 1:vec_dim]
                    u_node_old = :(Vec{$vec_dim}($(Expr(:tuple, vec_comps_old...))))
                    push!(value_old_terms, :(Nvals[$node_idx] * $u_node_old))
                    push!(grad_old_terms, :(dN[$node_idx] ⊗ $u_node_old))

                    # Rate values
                    vec_comps_rate = [:(u_rate[elem.dof_indices[$(offset+node*vec_dim+comp)]]) for comp in 1:vec_dim]
                    u_node_rate = :(Vec{$vec_dim}($(Expr(:tuple, vec_comps_rate...))))
                    push!(rate_terms, :(Nvals[$node_idx] * $u_node_rate))
                end

                value_expr = Expr(:call, :+, value_terms...)
                grad_expr = Expr(:call, :+, grad_terms...)
                grad_old_expr = Expr(:call, :+, grad_old_terms...)
                rate_expr = Expr(:call, :+, rate_terms...)

                # Gradient rate from increment
                grad_rate_expr = :(($grad_expr - $grad_old_expr) / Δt)

                # Create LocalField
                local_field_expr = :(LocalField($value_expr, $grad_expr, $rate_expr, $grad_rate_expr))
                push!(field_exprs, Expr(:(=), fname, local_field_expr))

                offset += n_nodes * vec_dim
            else
                error("Unsupported quantity type: $Q")
            end
        else
            error("Unsupported entity type: $entity_type (only Vertex supported for now)")
        end
    end

    nt_expr = Expr(:tuple, field_exprs...)

    return quote
        @inbounds begin
            # Evaluate basis functions once
            Nvals = get_basis_functions($topology, $basis, ξ)
            dN = get_basis_derivatives($topology, $basis, ξ)

            # Return NamedTuple of LocalField
            return $nt_expr
        end
    end
end
