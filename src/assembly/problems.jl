# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/FEMBase.jl/blob/master/LICENSE

abstract type AbstractProblem end
abstract type FieldProblem <: AbstractProblem end
abstract type BoundaryProblem <: AbstractProblem end
abstract type MixedProblem <: AbstractProblem end

"""
General linearized problem to solve
    (K₁+K₂)Δu  +   C1'*Δλ = f₁+f₂
         C2Δu  +     D*Δλ = g
"""
mutable struct Assembly

    M::SparseMatrixCOO  # mass matrix

    # for field assembly
    K::SparseMatrixCOO   # stiffness matrix
    Kg::SparseMatrixCOO  # geometric stiffness matrix
    f::SparseMatrixCOO   # force vector
    fg::SparseMatrixCOO  #

    # for boundary assembly
    C1::SparseMatrixCOO
    C2::SparseMatrixCOO
    D::SparseMatrixCOO
    g::SparseMatrixCOO
    c::SparseMatrixCOO

    u::Vector{Float64}  # solution vector u
    u_prev::Vector{Float64}  # previous solution vector u
    u_norm_change::Real  # change of norm in u

    la::Vector{Float64}  # solution vector la
    la_prev::Vector{Float64}  # previous solution vector u
    la_norm_change::Real # change of norm in la

    removed_dofs::Vector{Int} # manually remove dofs from assembly
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
        [], [], Inf,
        [])
end

function empty!(assembly::Assembly)
    empty!(assembly.M)
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
    T = isempty(assembly.M)
    T &= isempty(assembly.K)
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

"""
    Problem{P<:AbstractProblem}

Defines a new problem of type `P`, where `P` characterizes the physics of the
problem. `P` can be for example `Elasticity`, if the physics of the system is
described by Cauchy's stress equation ∇⋅σ + b = ̈ρu, or `Heat`, if the physics
of the problem is described by heat equation -∇⋅(k∇u) = f.

"""
mutable struct Problem{P<:AbstractProblem}
    name::AbstractString           # descriptive name for the problem
    dimension::Int                 # degrees of freedom per node
    parent_field_name::AbstractString # (optional) name of the parent field e.g. "displacement"
    elements::Vector{Element}
    dofmap::Dict{Element,Vector{Int}} # connects the element local dofs to the global dofs
    assembly::Assembly
    fields::Dict{String,AbstractField}
    postprocess_fields::Vector{String}
    properties::P
end

"""
    Problem(problem_type, problem_name, problem_dimension)

Construct a new field problem.

`problem_type` must be a subtype of `FieldProblem` (`Elasticity`, `Heat`, etc..).
`problem_dimensions` is the number of degrees of freedom each node is containing.

# Examples

To create vector-valued elasticity problem, having 3 dofs / node:
```julia
problem1 = Problem(Elasticity, "test problem", 3)
```

To create scalar-valued Poisson problem:
```julia
problem2 = Problem(Heat, "test problem 2", 1)
```

"""
function Problem(::Type{P}, name::AbstractString, dimension::Int) where P<:FieldProblem
    parent_field_name = "none"
    elements = []
    dofmap = Dict()
    assembly = Assembly()
    fields = Dict()
    postprocess_fields = Vector()
    properties = P()
    problem = Problem{P}(name, dimension, parent_field_name, elements, dofmap,
        assembly, fields, postprocess_fields, properties)
    @info("Creating a new problem of type $P, having name `$name` and " *
          "dimension $dimension dofs/node.")
    return problem
end

"""
    Problem(problem_type, problem_name, problem_dimension, parent_field_name)

Construct a new boundary problem.

`problem_type` must be a subtype of `BoundaryProblem` (`Dirichlet`, `Contact`,
etc..). `problem_dimensions` is the number of degrees of freedom each node is
containing. `parent_field_name` is describing the field, where the boundary
problem is affecting.

# Examples

To create a Dirichlet boundary condition for a vector-valued elasticity problem,
having 3 dofs / node:
```julia
bc1 = Problem(Dirichlet, "fix displacement on support", 3, "displacement")
```

To create a Dirichlet boundary condition for scalar-valued Poisson problem:
```julia
bc2 = Problem(Dirichlet, "fix surface temperature", 1, "temperature")
```
"""
function Problem(::Type{P}, name, dimension, parent_field_name) where P<:BoundaryProblem
    elements = []
    dofmap = Dict()
    assembly = Assembly()
    fields = Dict()
    postprocess_fields = Vector()
    properties = P()
    problem = Problem{P}(name, dimension, parent_field_name, elements, dofmap,
        assembly, fields, postprocess_fields, properties)
    @info("Creating a new boundary problem of type $P, having name `$name` and " *
          "dimension $dimension dofs/node. This boundary problems fixes field " *
          "`$parent_field_name`.")
    return problem
end

function get_formulation_type(::Problem)
    return :incremental
end

"""
    get_unknown_field_dimension(problem)

Return the dimension of the unknown field of this problem.
"""
function get_unknown_field_dimension(problem::Problem)
    return problem.dimension
end

"""
    get_unknown_field_name(problem)

Default function if unknown field name is not defined for some problem.
"""
function get_unknown_field_name(::P) where P<:AbstractProblem
    @warn("The name of unknown field (e.g. displacement, temperature, ...) of the " *
          "problem type must be given by defining a function " *
          "`get_unknown_field_name(::$P)`")
    return "N/A"
end

""" Return the name of the unknown field of this problem. """
function get_unknown_field_name(problem::Problem{P}) where P
    return get_unknown_field_name(problem.properties)
end

""" Return the name of the parent field of this (boundary) problem. """
function get_parent_field_name(problem::Problem{P}) where P<:BoundaryProblem
    return problem.parent_field_name
end

function get_unknown_field_name(::P) where P<:BoundaryProblem
    return "lambda"
end

is_field_problem(::Problem) = false
is_field_problem(::Problem{P}) where {P<:FieldProblem} = true
is_boundary_problem(::Problem) = false
is_boundary_problem(::Problem{P}) where {P<:BoundaryProblem} = true

function get_elements(problem::Problem)
    return problem.elements
end

function update!(problem::P, attr::Pair{String,String}...) where P<:AbstractProblem
    for (name, value) in attr
        setfield!(problem, Meta.parse(name), Meta.parse(value))
    end
end

"""
    function initialize!(problem_type, element_name, time)

Initialize the element ready for calculation, where `problem_type` is the type
of the problem (Elasticity, Dirichlet, etc.), `element_name` is the name of a
constructed element (see Element(element_type, connectivity_vector)) and `time`
is the starting time of the initializing process.
"""
function initialize!(problem::Problem, element::AbstractElement, time::Float64)
    field_name = get_unknown_field_name(problem)
    field_dim = get_unknown_field_dimension(problem)
    nnodes = length(element)
    if field_dim == 1  # scalar field
        empty_field = tuple(zeros(nnodes)...)
    else  # vector field
        # FIXME: the most effective way to do
        # ([0.0,0.0], [0.0,0.0], ..., [0.0,0.0]) ?
        empty_field = tuple(map((x) -> zeros(field_dim) * x, 1:nnodes)...)
    end

    # initialize primary field
    if !haskey(element, field_name)
        update!(element, field_name, time => empty_field)
    end

    # if a boundary problem, initialize also a field for the main problem
    is_boundary_problem(problem) || return
    field_name = get_parent_field_name(problem)
    if !haskey(element, field_name)
        update!(element, field_name, time => empty_field)
    end
end

function initialize!(problem::Problem, time::Float64=0.0)
    for element in get_elements(problem)
        initialize!(problem, element, time)
    end
end

function update!(problem::Problem, assembly::Assembly, u::Vector, la::Vector)

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
    # TODO: here we have couple of options and they need to be clarified
    # for total formulation we are solving total quantity Ku = f while in
    # incremental formulation we solve KΔu = f and u = u + Δu
    assembly.u_prev = copy(assembly.u)
    assembly.la_prev = copy(assembly.la)

    if get_formulation_type(problem) == :total
        assembly.u = u
        assembly.la = la
    elseif get_formulation_type(problem) == :incremental
        assembly.u += u
        assembly.la = la
    elseif get_formulation_type(problem) == :forwarddiff
        assembly.u += u
        assembly.la += la
    else
        @info("$(problem.name): unknown formulation type, don't know what to do with results")
        error("serious failure with problem formulation: $(get_formulation_type(problem))")
    end

    # calculate change of norm
    assembly.u_norm_change = norm(assembly.u - assembly.u_prev)
    assembly.la_norm_change = norm(assembly.la - assembly.la_prev)
    return assembly.u, assembly.la
end

"""
    get_global_solution(problem, assembly)

Return a global solution (u, la) for a problem.

Notes
-----
If the length of solution vector != number of nodes, i.e. the field dimension is
something else than 1, reshape vectors so that their length matches to the
number of nodes. This helps to get nodal results easily.
"""
function get_global_solution(problem::Problem, assembly::Assembly)
    u = assembly.u
    la = assembly.la
    field_dim = get_unknown_field_dimension(problem)
    if field_dim == 1
        return u, la
    else
        nnodes = round(Int, length(u) / field_dim)
        u = reshape(u, field_dim, nnodes)
        u = Vector{Float64}[u[:, i] for i in 1:nnodes]
        la = reshape(la, field_dim, nnodes)
        la = Vector{Float64}[la[:, i] for i in 1:nnodes]
        return u, la
    end
end

function update!(problem::Problem{P}, assembly::Assembly, elements::Vector{Element}, time::Float64) where P<:FieldProblem
    u, la = get_global_solution(problem, assembly)
    field_name = get_unknown_field_name(problem)
    # update solution u for elements
    for element in elements
        connectivity = get_connectivity(element)
        update!(element, field_name, time => tuple(u[connectivity]...))
    end
end

function update!(problem::Problem{P}, assembly::Assembly, elements::Vector{Element}, time::Float64) where P<:BoundaryProblem
    u, la = get_global_solution(problem, assembly)
    parent_field_name = get_parent_field_name(problem) # displacement
    field_name = get_unknown_field_name(problem) # lambda
    # update solution and lagrange multipliers for boundary elements
    for element in elements
        connectivity = get_connectivity(element)
        update!(element, parent_field_name, time => tuple(u[connectivity]...))
        update!(element, field_name, time => tuple(la[connectivity]...))
    end
end

"""
    add_element!(problem, element1, element2, ...)

Add element(s) to the problem.
"""
function add_element!(problem, elements...)
    for element in elements
        push!(problem.elements, element)
    end
    return nothing
end

"""
    add_elements!(problem, element_set_1, element_set_2, ...)

Add vectors/tuples of element(s) to the problem.
"""
function add_elements!(problem, element_sets::Union{Vector,Tuple}...)
    for elements in element_sets
        nelements = length(elements)
        @info("Adding $nelements elements to problem `$(problem.name)`")
        add_element!(problem, elements...)
    end
    return nothing
end

add_elements!(problem, elements::Element...) = add_element!(problem, elements...)

function add_elements!(problem, elements_or_lists_of_elements...)
    for item in elements_or_lists_of_elements
        add_elements!(problem, item)
    end
end

get_assembly(problem::Problem) = problem.assembly
Base.length(problem::Problem) = length(problem.elements)

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

function getindex(problem::Problem, field_name::String)
    return problem.fields[field_name]
end

#""" Return field calculated to nodal points for elements in problem p. """
function (problem::Problem)(field_name::String, time::Float64)
    #if haskey(problem, field_name)
    #    return problem[field_name](time)
    #end
    f = Dict{Int,Any}()
    for element in get_elements(problem)
        haskey(element, field_name) || continue
        for (c, v) in zip(get_connectivity(element), element(field_name, time))
            if haskey(f, c)
                if !isapprox(f[c], v)
                    @info("several values for single node when returning field $field_name")
                    @info("already have: $(f[c]), and trying to set $v")
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

# REMOVED: push!(problem, elements) - Use add_elements!(problem, elements) instead
# This violated Julia semantics (push! should be for collections, not domain logic)
# The modern Physics API uses add_elements!, add_dirichlet!, add_neumann!

"""
    set_gdofs!(problem, element)

Set element global degrees of freedom.
"""
function set_gdofs!(problem, element, dofs)
    problem.dofmap[element] = dofs
end

"""
    get_gdofs(problem, element)

Return the global degrees of freedom for element.

First make lookup from problem dofmap. If not defined there, make implicit
assumption that dofs follow formula `gdofs = [dim*(nid-1)+j for j=1:dim]`,
where `nid` is node id and `dim` is the dimension of problem. This formula
arranges dofs so that first comes all dofs of node 1, then node 2 and so on:
(u11, u12, u13, u21, u22, u23, ..., un1, un2, un3) for 3 dofs/node setting.
"""
function get_gdofs(problem::Problem, element::AbstractElement)
    if haskey(problem.dofmap, element)
        return problem.dofmap[element]
    end
    conn = get_connectivity(element)
    if length(conn) == 0
        error("element connectivity not defined, cannot determine global ",
            "degrees of freedom for element #: $(element.id)")
    end
    dim = get_unknown_field_dimension(problem)
    gdofs = [dim * (i - 1) + j for i in conn for j = 1:dim]
    return gdofs
end
