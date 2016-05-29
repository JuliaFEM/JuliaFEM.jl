# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

abstract AbstractProblem
abstract FieldProblem <: AbstractProblem
abstract BoundaryProblem <: AbstractProblem
abstract MixedProblem <: AbstractProblem

"""
General linearized problem to solve
    (K₁+K₂)*Δu +   C1.T*λ = f₁+f₂
         C2*Δu +      D*λ = g
"""
type Assembly
    # for field assembly
    M :: SparseMatrixCOO  # mass matrix
    K :: SparseMatrixCOO  # stiffness matrix
    Kg :: SparseMatrixCOO  # geometric stiffness matrix
    f :: SparseMatrixCOO  # force vector
#   f2 :: SparseMatrixCOO
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

    changed :: Bool  # flag to control is reassembly needed
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
        [], [], Inf,
        [], [], Inf,
        true)
end

function empty!(assembly::Assembly)
    empty!(assembly.M)
    empty!(assembly.K)
    empty!(assembly.Kg)
    empty!(assembly.f)
    empty!(assembly.C1)
    empty!(assembly.C2)
    empty!(assembly.D)
    empty!(assembly.g)
    empty!(assembly.c)
    assembly.changed = true
end

type Problem{P<:AbstractProblem}
    name :: ASCIIString              # descriptive name for problem
    dimension :: Int                 # degrees of freedom per node
    parent_field_name :: ASCIIString # (optional) name of parent field e.g. "displacement"
    elements :: Vector{Element}
    dofmap :: Dict{Element, Vector{Int64}} # connects element local dofs to global dofs
    assembly :: Assembly
    properties :: P
end

""" Construct a new field problem.

Examples
--------
Create vector-valued (dim=3) elasticity problem:

julia> prob = Problem(Elasticity, "this is my problem", 3)

"""
function Problem{P<:FieldProblem}(::Type{P}, name, dimension, elements=[], dofmap=Dict())
    Problem{P}(name, dimension, "none", elements, dofmap, Assembly(), P())
end

""" Construct a new boundary problem.

Examples
--------
Create Dirichlet boundary problem for vector-valued (dim=3) elasticity problem.

julia> bc1 = Problem(Dirichlet, "support", 3, "displacement")

"""
function Problem{P<:BoundaryProblem}(::Type{P}, name, dimension, parent_field_name, elements=[], dofmap=Dict())
    Problem{P}(name, dimension, parent_field_name, elements, dofmap, Assembly(), P())
end

function get_formulation_type{P<:FieldProblem}(problem::Problem{P})
    return :total
end

function get_formulation_type{P<:BoundaryProblem}(problem::Problem{P})
    return :total
end

function get_assembly(problem)
    return problem.assembly
end

""" Initialize unknown field ready for nonlinear iterations, i.e.,
    take last known value and set it as a initial quess for next
    time increment.
"""
function initialize!(problem::Problem, time::Real)
    field_name = get_unknown_field_name(problem)
    field_dim = get_unknown_field_dimension(problem)
    for element in get_elements(problem)
        gdofs = get_gdofs(problem, element)
        if haskey(element, field_name)
            # if field is found, copy last known solution to new time as initial guess
            if !isapprox(last(element[field_name]).time, time)
                last_data = copy(last(element[field_name]).data)
                push!(element[field_name], time => last_data)
            end
        else # if field not found at all, initialize new zero field.
            data = Vector{Float64}[zeros(field_dim) for i in 1:length(element)]
            element[field_name] = (time => data)
        end
    end
    # if this is boundary problem and not dirichlet problem, initialize field
    # for primary variable too
    is_boundary_problem(problem) || return
    #is_dirichlet_problem(problem) && return
    field_name = get_parent_field_name(problem)
    for element in get_elements(problem)
        gdofs = get_gdofs(problem, element)
        if haskey(element, field_name)
            # if field is found, copy last known solution to new time as initial guess
            if !isapprox(last(element[field_name]).time, time)
                last_data = copy(last(element[field_name]).data)
                push!(element[field_name], time => last_data)
            end
        else # if field not found at all, initialize new zero field.
            data = Vector{Float64}[zeros(field_dim) for i in 1:length(element)]
            element[field_name] = (time => data)
        end
    end
end

""" Update problem solution vector for assembly. """
function update_assembly!(problem, u, la)

    assembly = get_assembly(problem)

    # resize & fill with zeros vectors if length mismatch with current solution
    if length(u) != length(assembly.u)
        resize!(assembly.u, length(u))
        fill!(assembly.u, 0.0)
    end
    if length(la) != length(assembly.la)
        resize!(assembly.la, length(la))
        fill!(assembly.la, 0.0)
    end

    # copy current solutions to previous ones and add/replace new solution
    # TODO: here we have couple of options and they needs to be clarified
    # for total formulation we are solving total quantity Ku=f while in
    # incremental formulation we solve KΔu = f and u = u + Δu
    assembly.u_prev = copy(assembly.u)
    assembly.la_prev = copy(assembly.la)
    if get_formulation_type(problem) == :total
        info("$(problem.name): total formulation, replacing solution vector with new values")
        assembly.u = u
        assembly.la = la
    elseif get_formulation_type(problem) == :incremental
        info("$(problem.name): incremental formulation, adding increment to solution vector")
        assembly.u += u
        assembly.la = la
    elseif get_formulation_type(problem) == :forwarddiff
        info("$(problem.name): forwarddiff formulation, adding increment to solution vector")
        assembly.u += u
        assembly.la += la
    else
        info("$(problem.name): unknown formulation type, don't know what to do with results")
        error("serious failure with problem formulation: $(get_formulation_type(problem))")
    end

    # calculate change of norm
    assembly.u_norm_change = norm(assembly.u - assembly.u_prev)
    assembly.la_norm_change = norm(assembly.la - assembly.la_prev)
    #return assembly.u_norm_change, assembly.la_norm_change
    return assembly.u, assembly.la
end

""" Update solutions to elements.

Notes
-----
This assumes that element is properly initialized so that last known field data
is from current time. For boundary problems solution is updated from lambda vector
and for field problems from actual solution vector.
"""
function update_elements!{P<:FieldProblem}(problem::Problem{P}, u, la)
    field_name = get_unknown_field_name(problem)
    field_dim = get_unknown_field_dimension(problem)
    nnodes = round(Int, length(u)/field_dim)
    solution = reshape(u, field_dim, nnodes)
    for element in get_elements(problem)
        connectivity = get_connectivity(element) # node ids
        local_sol = Vector{Float64}[solution[:, node_id] for node_id in connectivity]
        last(element[field_name]).data = local_sol
    end
end
function update_elements!{P<:BoundaryProblem}(problem::Problem{P}, u, la)
    field_name = get_unknown_field_name(problem)
    field_dim = get_unknown_field_dimension(problem)
    nnodes = round(Int, length(u)/field_dim)
    solution = reshape(la, field_dim, nnodes)
    for element in get_elements(problem)
        connectivity = get_connectivity(element) # node ids
        local_sol = Vector{Float64}[solution[:, node_id] for node_id in connectivity]
        last(element[field_name]).data = local_sol
    end
    # if boundary problem is not dirichlet, update also data of main problem
    # is_dirichlet_problem(problem) && return
    field_name = get_parent_field_name(problem)
    solution = reshape(u, field_dim, nnodes)
    for element in get_elements(problem)
        connectivity = get_connectivity(element) # node ids
        local_sol = Vector{Float64}[solution[:, node_id] for node_id in connectivity]
        last(element[field_name]).data = local_sol
    end
end

#=
function add_postprocessor!(problem::Union{FieldProblem, BoundaryProblem}, postprocessor_name::Symbol, args...; kwargs...)
    push!(problem.postprocessors, (postprocessor_name, args, kwargs))
end

function add_preprocessor!(problem::Union{FieldProblem, BoundaryProblem}, preprocessor_name::Symbol, args...; kwargs...)
    push!(problem.preprocessors, (preprocessor_name, args, kwargs))
end

=#

function get_elements(problem)
    return problem.elements
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

function push!(problem::Problem, element)
    push!(problem.elements, element)
end

function get_gdofs(element::Element, dim::Int)
    conn = get_connectivity(element)
    if length(conn) == 0
        error("element connectivity not defined, cannot determine global dofs for element: $element")
    end
    gdofs = vec([dim*(i-1)+j for j=1:dim, i in conn])
    return gdofs
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
