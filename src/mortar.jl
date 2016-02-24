# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
Currently two strategies exists:

a) Remove inactive inequality constraints in element level. This is done in
   assemble! if normal_condition is set to :Contact. For some reason this
   leads to convergence issues.
b) Remove inactive inequality constraints in assembly level. This is done in
   posthook algorithm if inequality_constraints is set to true. This gives
   more robust behavior.

   Either use inequality_constraints=True OR :Contact + :Slip, but do not mix.

   minimum_distance can be used to roughly skip integration of mortar
   projections for elements that are "far enough" from each other. Increases
   performance.

"""
type Mortar <: BoundaryProblem
    formulation :: Symbol # :total, :incremental, :autodiff
    dual_basis :: Bool
    inequality_constraints :: Bool # Launch PDASS to solve inequality constraints
    normal_condition :: Symbol # Tie or Contact
    tangential_condition :: Symbol # Stick or Slip
    maximum_distance :: Float64 # don't check for a contact if elements are far enough
    store_debug_info :: Bool # for making debugging easier
    always_inactive :: Vector{Int64}
    always_in_contact :: Vector{Int64} # nodes in this list always in contact
    always_in_stick :: Vector{Int64} # nodes in this list always in stick
    always_in_slip :: Vector{Int64} # nodes in this list always in slip
    contact :: Bool
    friction :: Bool
    gap_sign :: Int # gap sign convention
    rotate_normals :: Bool
end

function Mortar()
    Mortar(:total, true, false, :Tie, :Stick, Inf, false, [], [], [], [], false, false, -1, false)
end

function get_unknown_field_name(::Type{Mortar})
    return "reaction force"
end

function get_formulation_type(problem::Problem{Mortar})
    return problem.properties.formulation
end

macro debug(msg)
    haskey(ENV, "DEBUG") || return
    return msg
end

include("mortar_2d.jl")
include("mortar_2d_autodiff.jl")
include("mortar_3d.jl")

""" Remove inactive inequality constraints by using primal-dual active set strategy. """
function boundary_assembly_posthook!(solver::Solver, problem::Problem{Mortar}, C1, C2, D, g)
    problem.properties.inequality_constraints || return
    info("PDASS: Starting primal-dual active set strategy to determine active constraints")
    S = Set{Int64}()
    for element in get_elements(problem)
        haskey(element, "master elements") || continue
        push!(S, get_connectivity(element)...)
    end
    S = sort(collect(S))
    dim = get_unknown_field_dimension(problem)
    ndofs = solver.ndofs
    nnodes = round(Int, ndofs/dim)

    c = reshape(full(problem.assembly.c, ndofs, 1), dim, nnodes)
    A = find(c[1,:] .> 0)
    A = intersect(A, S)
    I = setdiff(S, A)

    info("PDASS: contact nodes: $(sort(collect(S)))")
    info("PDASS: active nodes: $(sort(collect(A)))")
    info("PDASS: inactive nodes: $(sort(collect(I)))")

    # remove any inactive nodes
    for j in I
        dofs = [dim*(j-1)+i for i=1:dim]
        C1[dofs,:] = 0
        C2[dofs,:] = 0
        D[dofs,:] = 0
        g[dofs,:] = 0
    end

    # handle tangential condition for active nodes
    if problem.properties.tangential_condition == :Slip
        for j in A
            dofs = [dim*(j-1)+i for i=1:dim]
            tangential_dofs = dofs[2:end]
            D[tangential_dofs,dofs] = C2[tangential_dofs,dofs]
            C2[tangential_dofs,:] = 0
            g[tangential_dofs,:] = 0
        end
    end

    return
end

function assemble_prehook!(problem::Problem{Mortar}, time::Real)
    info("mortar assemble prehook at time $time")
    slaves = Set{Element}()
    for element in get_elements(problem)
        haskey(element, "master elements") || continue
        push!(slaves, element)
    end
    info("$(length(slaves)) slave elements")
    length(slaves) != 0 || error("no slave elements found for problem (forget to add masters?).")
    info("mortar: update normal-tangential system.")
    calculate_normal_tangential_coordinates!(collect(slaves), time)
    info("mortar assemble prehook done.")
end
