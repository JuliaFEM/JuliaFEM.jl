# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
DOF extraction from global solution vector.

See `src/elements/README.md` for usage examples and performance notes.
"""

using Tensors

"""
    extract_element_dofs(elem::Element{K,P,S,N}, u_global::AbstractVector) → NamedTuple

Extract element DOFs as flat tuples of scalars. Zero-allocation @generated function.
See `src/elements/README.md` for examples.
"""
@generated function extract_element_dofs(
    elem::Element{K,P,S,N},
    u_global::AbstractVector
) where {K,P,S<:DOFSet,N}
    field_names = fieldnames(S)
    
    # Build expressions for accessing each DOF value from flat tuple
    all_field_vals = Expr[]
    offset = 0
    
    for fname in field_names
        # Get number of DOFs for this field from type. `field_tuple_type`
        # is the per-field spec (`DOF{Quantity, Entity}`); we read the
        # entity slot directly and the quantity through `quantity_type`.
        field_tuple_type = fieldtype(S, fname)
        entity_type = field_tuple_type.parameters[2]
        n_entities = (entity_type === Vertex) ? nnodes(K()) :
                     (entity_type === Edge) ? length(edges(K())) :
                     (entity_type === Face) ? length(faces(K())) :
                     error("Unsupported entity type: $entity_type")

        Q = quantity_type(field_tuple_type)  # e.g. Vec{3} or Float64

        if Q === Float64
            n_dofs = n_entities
        elseif Q isa UnionAll && Q.body <: Tensor && Q.body.parameters[1] == 1
            dim = Q.body.parameters[2]
            n_dofs = n_entities * dim
        else
            error("Unsupported quantity type: $Q")
        end

        # Generate tuple for this field using flat dof_indices.
        # Access elem.dof_indices[offset+1], elem.dof_indices[offset+2], ...
        dof_vals = [:(u_global[elem.dof_indices[$(offset+i)]]) for i in 1:n_dofs]
        push!(all_field_vals, Expr(:tuple, dof_vals...))

        offset += n_dofs
    end
    
    # Build complete NamedTuple as a single expression
    # Result: (T = (v1, v2, v3), u = (v4, v5, ...))
    nt_expr = Expr(:tuple, [Expr(:(=), fname, all_field_vals[i]) for (i, fname) in enumerate(field_names)]...)
    
    return :(@inbounds $nt_expr)
end

"""
    extract_element_dofs_structured(elem::Element{K,P,S,N}, u_global::AbstractVector) → NamedTuple

Extract element DOFs as structured quantities (Vec, Tensor) matching S specification.
Tuple length equals number of entities. Zero-allocation @generated function.
See `src/elements/README.md` for examples.
"""
@generated function extract_element_dofs_structured(
    elem::Element{K,P,S,N},
    u_global::AbstractVector
) where {K,P,S<:DOFSet,N}
    field_names = fieldnames(S)
    n_nodes = nnodes(K())
    
    # Build expressions for each field using flat tuple with offset
    all_field_vals = Expr[]
    offset = 0
    
    for fname in field_names
        field_tuple_type = fieldtype(S, fname)
        entity_type = field_tuple_type.parameters[2]
        n_entities = (entity_type === Vertex) ? n_nodes :
                     (entity_type === Edge) ? length(edges(K())) :
                     (entity_type === Face) ? length(faces(K())) :
                     error("Unsupported entity type: $entity_type")

        Q = quantity_type(field_tuple_type)  # e.g. Vec{3} or Float64
        
        if Q === Float64
            # Scalar field: tuple of scalars from flat dof_indices
            dof_vals = [:(u_global[elem.dof_indices[$(offset+i)]]) for i in 1:n_entities]
            push!(all_field_vals, Expr(:tuple, dof_vals...))
            offset += n_entities
        elseif Q isa UnionAll && Q.body <: Tensor && Q.body.parameters[1] == 1
            # Vector field: tuple of Vec instances from flat dof_indices
            dim = Q.body.parameters[2]
            vec_vals = Expr[]
            for entity in 0:(n_entities-1)
                comp_vals = [:(u_global[elem.dof_indices[$(offset+entity*dim+comp)]]) for comp in 1:dim]
                push!(vec_vals, :(Vec{$dim}($(Expr(:tuple, comp_vals...)))))
            end
            push!(all_field_vals, Expr(:tuple, vec_vals...))
            offset += n_entities * dim
        else
            error("Unsupported quantity type: $Q for field $fname")
        end
    end
    
    # Build complete NamedTuple as a single expression  
    nt_expr = Expr(:tuple, [Expr(:(=), fname, all_field_vals[i]) for (i, fname) in enumerate(field_names)]...)
    
    return :(@inbounds $nt_expr)
end

# Single-field versions (same implementation, Julia will dispatch to multi-field @generated)
