# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

abstract AbstractProblem
abstract FieldProblem <: AbstractProblem
abstract BoundaryProblem <: AbstractProblem
abstract MixedProblem <: AbstractProblem

"""
General linearized problem to solve
    (K₁+K₂)Δu  +   C1*Δλ = f₁+f₂
         C2Δu  +    D*Δλ = g
"""
type Assembly

    M :: SparseMatrixCOO  # mass matrix

    # for field assembly
    K :: SparseMatrixCOO   # stiffness matrix
    Kg :: SparseMatrixCOO  # geometric stiffness matrix
    f :: SparseMatrixCOO   # force vector
    fg :: SparseMatrixCOO  #

    # for boundary assembly
    C1 :: SparseMatrixCOO
    C2 :: SparseMatrixCOO
    D :: SparseMatrixCOO
    g :: SparseMatrixCOO
    c :: SparseMatrixCOO

    u :: Vector{Float64}  # solution vector u
    u_prev :: Vector{Float64}  # previous solution vector u
    u_norm_change :: Real  # change of norm in u

    la :: Vector{Float64}  # solution vector la
    la_prev :: Vector{Float64}  # previous solution vector u
    la_norm_change :: Real # change of norm in la

end

function Assembly()
    return Assembly(
        SparseMatrixCOO(),
        SparseMatrixCOO(),
        SparseMatrixCOO(),
        SparseMatrixCOO(),
        SparseMatrixCOO(),
        SparseMatrixCOO(),
        SparseMatrixCOO(),
        SparseMatrixCOO(),
        SparseMatrixCOO(),
        SparseMatrixCOO(),
        [], [], Inf,
        [], [], Inf)
end

function empty!(assembly::Assembly)
    empty!(assembly.K)
    empty!(assembly.Kg)
    empty!(assembly.f)
    empty!(assembly.fg)
    empty!(assembly.C1)
    empty!(assembly.C2)
    empty!(assembly.D)
    empty!(assembly.g)
    empty!(assembly.c)
end

function isempty(assembly::Assembly)
    T = isempty(assembly.K)
    T &= isempty(assembly.Kg)
    T &= isempty(assembly.f)
    T &= isempty(assembly.fg)
    T &= isempty(assembly.C1)
    T &= isempty(assembly.C2)
    T &= isempty(assembly.D)
    T &= isempty(assembly.g)
    T &= isempty(assembly.c)
    return T
end

function get_dofs(assembly::Assembly)
    return sort(unique(assembly.K.J))
end

type Problem{P<:AbstractProblem}
    name :: AbstractString           # descriptive name for problem
    dimension :: Int                 # degrees of freedom per node
    parent_field_name :: AbstractString # (optional) name of parent field e.g. "displacement"
    elements :: Vector{Element}
    dofmap :: Dict{Element, Vector{Int64}} # connects element local dofs to global dofs
    assembly :: Assembly
    fields :: Dict{AbstractString, Field}
    properties :: P
end

""" Construct a new field problem.

Examples
--------
Create vector-valued (dim=3) elasticity problem:

julia> prob1 = Problem(Elasticity, "this is my problem", 3)
julia> prob2 = Problem(Elasticity, 3)

"""
function Problem{P<:FieldProblem}(::Type{P}, name::AbstractString, dimension::Int64)
    return Problem{P}(name, dimension, "none", [], Dict(), Assembly(), Dict(), P())
end
function Problem{P<:FieldProblem}(::Type{P}, dimension::Int64)
    return Problem{P}("$P problem", dimension, "none", [], Dict(), Assembly(), Dict(), P())
end

""" Construct a new boundary problem.

Examples
--------
Create Dirichlet boundary problem for vector-valued (dim=3) elasticity problem.

julia> bc1 = Problem(Dirichlet, "support", 3, "displacement")
solver.
"""
function Problem{P<:BoundaryProblem}(::Type{P}, name, dimension, parent_field_name)
    return Problem{P}(name, dimension, parent_field_name, [], Dict(), Assembly(), Dict(), P())
end
function Problem{P<:BoundaryProblem}(::Type{P}, main_problem::Problem)
    name = "$P problem"
    dimension = get_unknown_field_dimension(main_problem)
    parent_field_name = get_unknown_field_name(main_problem)
    return Problem{P}(name, dimension, parent_field_name, [], Dict(), Assembly(), Dict(), P())
end

function get_formulation_type{P<:FieldProblem}(problem::Problem{P})
    return :incremental
end

function get_formulation_type{P<:BoundaryProblem}(problem::Problem{P})
    return :incremental
end

function get_assembly(problem)
    return problem.assembly
end


""" Initialize element ready for calculation. """
function initialize!(problem::Problem, element::Element, time::Float64)
    field_name = get_unknown_field_name(problem)
    field_dim = get_unknown_field_dimension(problem)
    nnodes = length(element)

    # initialize primary field
    if !haskey(element, field_name)
        if field_dim == 1
            update!(element, field_name, time => zeros(nnodes))
        else
            update!(element, field_name, time => [zeros(field_dim) for i=1:nnodes])
        end
    end

    # if boundary problem, initialize field for main problem too
    is_boundary_problem(problem) || return
    field_name = get_parent_field_name(problem)
    if !haskey(element, field_name)
        if field_dim == 1
            update!(element, field_name, time => zeros(nnodes))
        else
            update!(element, field_name, time => [zeros(field_dim) for i=1:nnodes])
        end
    end
end

function initialize!(problem::Problem, time::Float64=0.0)
    for element in get_elements(problem)
        initialize!(problem, element, time)
    end
end

""" Update problem solution vector for assembly. """
function update!(problem::Problem, assembly::Assembly, u::Vector, la::Vector; verbose=false)

    # resize & fill with zeros vectors if length mismatch with current solution

    if length(u) != length(assembly.u)
        info("resizing solution vector u")
        resize!(assembly.u, length(u))
        fill!(assembly.u, 0.0)
    end

    if length(la) != length(assembly.la)
        info("resizing lagrange multipliers vector u")
        resize!(assembly.la, length(la))
        fill!(assembly.la, 0.0)
    end

    # copy current solutions to previous ones and add/replace new solution
    # TODO: here we have couple of options and they need to be clarified
    # for total formulation we are solving total quantity Ku = f while in
    # incremental formulation we solve KΔu = f and u = u + Δu
    assembly.u_prev = copy(assembly.u)
    assembly.la_prev = copy(assembly.la)

    if get_formulation_type(problem) == :total
        verbose && info("$(problem.name): total formulation, replacing solution vector with new values")
        assembly.u = u
        assembly.la = la
    elseif get_formulation_type(problem) == :incremental
        verbose && info("$(problem.name): incremental formulation, adding increment to solution vector")
        assembly.u += u
        assembly.la = la
    elseif get_formulation_type(problem) == :forwarddiff
        verbose && info("$(problem.name): forwarddiff formulation, adding increment to solution vector and reaction force vector")
        assembly.u += u
        assembly.la += la
    else
        info("$(problem.name): unknown formulation type, don't know what to do with results")
        error("serious failure with problem formulation: $(get_formulation_type(problem))")
    end

    # calculate change of norm
    assembly.u_norm_change = norm(assembly.u - assembly.u_prev)
    assembly.la_norm_change = norm(assembly.la - assembly.la_prev)
    return assembly.u, assembly.la
end

""" Return global solution (u, la) for problem.

Notes
-----
If length of solution vector != number of nodes, i.e. field dimension is
something other than 1, reshape vectors so it's length matches to the
number of nodes so that one can easily get nodal results.
"""
function get_global_solution(problem::Problem, assembly::Assembly)
    u = assembly.u
    la = assembly.la
    field_dim = get_unknown_field_dimension(problem)
    if field_dim == 1
        return u, la
    else
        nnodes = round(Int, length(u)/field_dim)
        u = reshape(u, field_dim, nnodes)
        u = Vector{Float64}[u[:,i] for i in 1:nnodes]
        la = reshape(la, field_dim, nnodes)
        la = Vector{Float64}[la[:,i] for i in 1:nnodes]
        return u, la
    end
end

""" Update solution from assebly to elements. """
function update!{P<:FieldProblem}(problem::Problem{P}, assembly::Assembly, elements::Vector{Element}, time::Float64)
    u, la = get_global_solution(problem, assembly)
    field_name = get_unknown_field_name(problem)
    # update solution u for elements
    for element in elements
        connectivity = get_connectivity(element)
        update!(element, field_name, time => u[connectivity])
    end
end

function update!{P<:BoundaryProblem}(problem::Problem{P}, assembly::Assembly, elements::Vector{Element}, time::Float64)
    u, la = get_global_solution(problem, assembly)
    parent_field_name = get_parent_field_name(problem) # displacement
    field_name = get_unknown_field_name(problem) # reaction force
    # update solution u and reaction force λ for boundary elements
    for element in elements
        connectivity = get_connectivity(element)
        update!(element, parent_field_name, time => u[connectivity])
        # FIXME
        update!(element, field_name, time => -la[connectivity])
    end
end

function get_elements(problem::Problem)
    return problem.elements
end

function get_assembly(problem::Problem)
    return problem.assembly
end

function length(problem::Problem)
    return length(problem.elements)
end

function update!(problem::Problem, field_name::AbstractString, data)
    #if haskey(problem.fields, field_name)
    #    update!(problem.fields[field_name], field_name::AbstractString, data)
    #else
    #    problem.fields[field_name] = Field(data)
    #end
    update!(problem.elements, field_name::AbstractString, data)
end

function haskey(problem::Problem, field_name::AbstractString)
    return haskey(problem.fields, field_name)
end

function getindex(problem::Problem, field_name::AbstractString)
    return problem.fields[field_name]
end

""" Return field calculated to nodal points for elements in problem p. """
function call(problem::Problem, field_name::AbstractString, time::Float64=0.0)
    #if haskey(problem, field_name)
    #    return problem[field_name](time)
    #end
    f = nothing
    for element in get_elements(problem)
        haskey(element, field_name) || continue
        for (c, v) in zip(get_connectivity(element), element(field_name, time))
            if f == nothing
                f = Dict(c => v)
                continue
            end
            if haskey(f, c)
                if !isapprox(f[c], v)
                    info("several values for single node when returning field $field_name")
                    info("already have: $(f[c]), and trying to set $v")
                end
            else
                f[c] = v
            end
        end
    end
    #f == nothing && return f
    #update!(problem, field_name, time => f)
    return f
end

""" Return the dimension of the unknown field of this problem. """
function get_unknown_field_dimension(problem::Problem)
    return problem.dimension
end

""" Return the name of the unknown field of this problem. """
function get_unknown_field_name{P}(problem::Problem{P})
    return get_unknown_field_name(P)
end

""" Return the name of the parent field of this (boundary) problem. """
function get_parent_field_name{P<:BoundaryProblem}(problem::Problem{P})
    return problem.parent_field_name
end

function push!(problem::Problem, elements...)
    push!(problem.elements, elements...)
end

function push!(problem::Problem, elements::Vector)
    push!(problem.elements, elements...)
end

function push!(problem::Problem, elements_::Vector...)
    for elements in elements_
        push!(problem.elements, elements...)
    end
end

function get_connectivity(problem::Problem)
    return union([get_connectivity(element) for element in get_elements(problem)]...)
end

function get_gdofs(element::Element, dim::Int)
    conn = get_connectivity(element)
    if length(conn) == 0
        error("element connectivity not defined, cannot determine global dofs for element: $element")
    end
    gdofs = vec([dim*(i-1)+j for j=1:dim, i in conn])
    return gdofs
end

function get_dofs(problem::Problem)
    return get_dofs(problem.assembly)
end

function empty!(problem::Problem)
    empty!(problem.assembly)
end

""" Return global degrees of freedom for element.

Notes
-----
First look dofs from problem.dofmap, it not found, update dofmap from
element.element connectivity using formula gdofs = [dim*(nid-1)+j for j=1:dim]
1. look element dofs from problem.dofmap
2. if not found, use element.connectivity to update dofmap and 1.
"""
function get_gdofs(problem::Problem, element::Element)
    if !haskey(problem.dofmap, element)
        dim = get_unknown_field_dimension(problem)
        problem.dofmap[element] = get_gdofs(element, dim)
    end
    return problem.dofmap[element]
end

""" Find dofs corresponding to nodes. """
function find_dofs_by_nodes(problem::Problem, nodes)
    dim = get_unknown_field_dimension(problem)
    return find_dofs_by_nodes(dim, nodes)
end
function find_dofs_by_nodes(dim::Int, nodes)
    dofs = Int64[]
    for node in nodes
        for j=1:dim
            push!(dofs, dim*(node-1)+j)
        end
    end
    return dofs
end

""" Find nodes corresponding to dofs. """
function find_nodes_by_dofs(problem::Problem, dofs)
    dim = get_unknown_field_dimension(problem)
    return find_nodes_by_dofs(dim, dofs)
end
function find_nodes_by_dofs(dim, dofs)
    nodes = Int64[]
    for dof in dofs
        j = Int(ceil(dof/dim))
        j in nodes && continue
        push!(nodes, j)
    end
    return nodes
end
