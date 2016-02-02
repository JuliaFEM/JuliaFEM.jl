# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

abstract AbstractProblem

function get_formulation_type{P<:AbstractProblem}(::Type{P})
    return :total
end

type FieldAssembly
    mass_matrix :: SparseMatrixCOO
    stiffness_matrix :: SparseMatrixCOO
    force_vector :: SparseMatrixCOO
    solution :: Vector{Float64}
    previous_solution :: Vector{Float64}
    solution_norm_change :: Real
    prehooks :: Vector{Tuple{Symbol,Any,Any}}
    posthooks :: Vector{Tuple{Symbol,Any,Any}}
    changed :: Bool  # flag to control is reassembly needed
end

function FieldAssembly()
    return FieldAssembly(
        SparseMatrixCOO(),
        SparseMatrixCOO(),
        SparseMatrixCOO(),
        [], [], Inf, [], [], true)
end

function Base.empty!(assembly::FieldAssembly)
    empty!(assembly.mass_matrix)
    empty!(assembly.stiffness_matrix)
    empty!(assembly.force_vector)
    assembly.changed = true
end

typealias Assembly FieldAssembly

""" Construct a new field problem.

Examples
--------
Create vector-valued (dim=3) elasticity problem:

julia> prob = FieldProblem(ElasticityProblem, "this is my problem", 3)

"""
type FieldProblem{T}
    name :: ASCIIString
    dim :: Int
    elements :: Vector{Element}
    assembly :: FieldAssembly
    properties :: T
end
function FieldProblem(problem::DataType, name::ASCIIString, dim::Int,
                      elements=[])
    FieldProblem{problem}(name, dim, elements, FieldAssembly(), problem())
end


function update!{P}(problem::FieldProblem{P}, solution::Vector{Float64})
    # resize & fill with zeros solution vector if length mismatch with current solution
    if length(solution) != length(problem.assembly.solution)
        resize!(problem.assembly.solution, length(solution))
        fill!(problem.assembly.solution, 0.0)
    end
    problem.assembly.previous_solution = copy(problem.assembly.solution)
    if get_formulation_type(P) == :incremental
        problem.assembly.solution += solution
    else
        problem.assembly.solution = solution
    end
    problem.assembly.solution_norm_change = norm(problem.assembly.solution - problem.assembly.previous_solution)
end


"""
Interface matrices C₁, C₂ & D, g for general problem type
Au + C₁'λ = f
C₂u + Dλ  = g
"""
type BoundaryAssembly
    C1 :: SparseMatrixCOO
    C2 :: SparseMatrixCOO
    D :: SparseMatrixCOO
    g :: SparseMatrixCOO
    solution :: Vector{Float64}
    previous_solution :: Vector{Float64}
    solution_norm_change :: Real
    prehooks :: Vector{Tuple{Symbol,Any,Any}}
    posthooks :: Vector{Tuple{Symbol,Any,Any}}
    changed :: Bool   # flag to control is reassembly needed
end

function BoundaryAssembly()
    return BoundaryAssembly(
        SparseMatrixCOO(),
        SparseMatrixCOO(),
        SparseMatrixCOO(),
        SparseMatrixCOO(),
        [], [], Inf, [], [], true)
end

function Base.empty!(assembly::BoundaryAssembly)
    empty!(assembly.C1)
    empty!(assembly.C2)
    empty!(assembly.D)
    empty!(assembly.g)
    assembly.changed = true
end


""" Construct a new boundary problem.

Examples
--------
Create Dirichlet boundary problem for vector-valued (dim=3) elasticity problem.

julia> bc1 = FieldProblem(DirichletProblem, "support", "displacement", 3)

"""
type BoundaryProblem{T}
    name :: ASCIIString
    parent_field_name :: ASCIIString
    parent_field_dim :: Int
    elements :: Vector{Element}
    assembly :: BoundaryAssembly
    properties :: T
end
function BoundaryProblem(problem::DataType,
                         name::ASCIIString,
                         parent_field_name::ASCIIString,
                         parent_field_dim::Int,
                         elements=[])
    BoundaryProblem{problem}(name, parent_field_name, parent_field_dim,
                                  elements, BoundaryAssembly(), problem())
end

function update!{P}(problem::BoundaryProblem{P}, solution::Vector{Float64})
    if length(solution) != length(problem.assembly.solution)
        resize!(problem.assembly.solution, length(solution))
        fill!(problem.assembly.solution, 0.0)
    end
    problem.assembly.previous_solution = copy(problem.assembly.solution)
    if get_formulation_type(P) == :incremental
        problem.assembly.solution += solution
    else
        problem.assembly.solution = solution
    end
    problem.assembly.solution_norm_change = norm(problem.assembly.solution - problem.assembly.previous_solution)
end

#=
function add_postprocessor!(problem::Union{FieldProblem, BoundaryProblem}, postprocessor_name::Symbol, args...; kwargs...)
    push!(problem.postprocessors, (postprocessor_name, args, kwargs))
end

function add_preprocessor!(problem::Union{FieldProblem, BoundaryProblem}, preprocessor_name::Symbol, args...; kwargs...)
    push!(problem.preprocessors, (preprocessor_name, args, kwargs))
end

=#

typealias Problem FieldProblem

typealias AllProblems Union{FieldProblem, BoundaryProblem}

function get_elements(problem::AllProblems)
    return problem.elements
end

""" Return the dimension of the unknown field of this problem. """
function get_unknown_field_dimension(problem::Problem)
    return problem.dim
end

""" Return the name of the unknown field of this problem. """
function get_unknown_field_name{P}(problem::Problem{P})
    return get_unknown_field_name(P)
end

function Base.push!(problem::AllProblems, element::Element)
    push!(problem.elements, element)
end

# TODO: better place for utility functions?

""" Calculate "nodal" vector from set of elements.

For example element 1 with dofs [1, 2, 3, 4] has [1, 1, 1, 1] and
element 2 with dofs [3, 4, 5, 6] has [2, 2, 2, 2] the result will
be sparse matrix with values [1, 1, 3, 3, 2, 2].

Parameters
----------
field_name
    name of field, e.g. "geometry"
field_dim
    degrees of freedom / node
elements
    elements used to calculate vector
time
"""
function calculate_nodal_vector(field_name::ASCIIString, field_dim::Int, elements::Vector{Element}, time::Real)
    A = SparseMatrixCOO()
    b = SparseMatrixCOO()
    for element in elements
        haskey(element, field_name) || continue
        gdofs = get_gdofs(element, 1)
        for ip in get_integration_points(element, Val{2})
            J = get_jacobian(element, ip, time)
            w = ip.weight*norm(J)
            f = element(field_name, ip, time)
            N = element(ip, time)
            add!(A, gdofs, gdofs, w*kron(N', N))
            for dim=1:field_dim
                add!(b, gdofs, w*f[dim]*N, dim)
            end
        end
    end
    A = sparse(A)
    b = sparse(b)
    nz = sort(unique(rowvals(A)))
    x = zeros(size(b)...)
    x[nz, :] = A[nz,nz] \ b[nz, :]
    return vec(transpose(x))
end

