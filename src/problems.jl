# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

abstract AbstractProblem
abstract FieldProblem <: AbstractProblem
abstract BoundaryProblem <: AbstractProblem
abstract MixedProblem <: AbstractProblem

"""
General linearized problem to solve
    K*u +   C1'*la  = f
    C2*u +  D*la    = g
"""
type Assembly
    # for field assembly
    M :: SparseMatrixCOO  # mass matrix
    K :: SparseMatrixCOO  # stiffness matrix
    f :: SparseMatrixCOO  # force vector
    # for boundary assembly
    C1 :: SparseMatrixCOO
    C2 :: SparseMatrixCOO
    D :: SparseMatrixCOO 
    g :: SparseMatrixCOO

    solution :: Vector{Float64}  # full solution vector when solving problem Ax = b
    previous_solution :: Vector{Float64}  # previous solution vector
    solution_norm_change :: Real  # for convergence studies
    prehooks :: Vector{Tuple{Symbol,Any,Any}}  # assign possible prehooks before assembly
    posthooks :: Vector{Tuple{Symbol,Any,Any}}  # assign possible posthooks after assembly
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
        [], [], Inf,
        [], [], true)
end

function Base.empty!(assembly::Assembly)
    empty!(assembly.M)
    empty!(assembly.K)
    empty!(assembly.f)
    empty!(assembly.C1)
    empty!(assembly.C2)
    empty!(assembly.D)
    empty!(assembly.g)
    assembly.changed = true
end

type Problem{P<:AbstractProblem}
    name :: ASCIIString              # descriptive name for problem
    dimension :: Int                 # degrees of freedom per node
    parent_field_name :: ASCIIString # (optional) name of parent field e.g. "displacement"
    elements :: Vector{Element}
    assembly :: Assembly
    properties :: P
end

""" Construct a new field problem.

Examples
--------
Create vector-valued (dim=3) elasticity problem:

julia> prob = Problem(Elasticity, "this is my problem", 3)

"""
function Problem{P<:FieldProblem}(::Type{P}, name, dimension, elements=[])
    Problem{P}(name, dimension, "none", elements, Assembly(), P())
end

""" Construct a new boundary problem.

Examples
--------
Create Dirichlet boundary problem for vector-valued (dim=3) elasticity problem.

julia> bc1 = Problem(Dirichlet, "support", 3, "displacement")

"""
function Problem{P<:BoundaryProblem}(::Type{P}, name, dimension, parent_field_name, elements=[])
    Problem{P}(name, dimension, parent_field_name, elements, Assembly(), P())
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

""" Update problem solution vector.
"""
function update!(problem::Problem, solution::Vector{Float64})
    assembly = get_assembly(problem)
    # resize & fill with zeros solution vector if length mismatch with current solution
    if length(solution) != length(assembly.solution)
        resize!(assembly.solution, length(solution))
        fill!(assembly.solution, 0.0)
    end
    assembly.previous_solution = copy(assembly.solution)
    if get_formulation_type(problem) == :incremental
        assembly.solution += solution
    else
        assembly.solution = solution
    end
    assembly.solution_norm_change = norm(assembly.solution - assembly.previous_solution)
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

function push!(problem::Problem, element)
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

