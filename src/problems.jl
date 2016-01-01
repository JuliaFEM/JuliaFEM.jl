# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

abstract AbstractProblem

type FieldProblem{T<:AbstractProblem}
    name :: ASCIIString
    dim :: Int
    elements :: Vector{Element}
    preprocessors :: Vector{Tuple{Symbol,Any,Any}}
    postprocessors :: Vector{Tuple{Symbol,Any,Any}}
end

type BoundaryProblem{T<:AbstractProblem}
    name :: ASCIIString
    parent_field_name :: ASCIIString
    parent_field_dim :: Int
    dim :: Int
    elements :: Vector{Element}
    preprocessors :: Vector{Tuple{Symbol,Any,Any}}
    postprocessors :: Vector{Tuple{Symbol,Any,Any}}
end

""" Construct new field problem.

Examples
--------
Create vector-valued (dim=3) elasticity problem:

julia> prob = FieldProblem(ElasticityProblem, "this is my problem", 3)

"""
function FieldProblem(problem_type::DataType, name::ASCIIString, dim::Int, elements=[], preprocessors=[], postprocessors=[])
    FieldProblem{problem_type}(name, dim, elements, preprocessors, postprocessors)
end


""" Construct new boundary problem.

Examples
--------
Create Dirichlet boundary problem for vector-valued (dim=3) elasticity problem.

julia> bc1 = FieldProblem(DirichletProblem, "support dy=0", "displacement", 3)

"""
function BoundaryProblem(problem_type::DataType, name::ASCIIString, parent_field_name::ASCIIString, parent_field_dim::Int, dim::Int=1, elements=[], preprocessors=[], postprocessors=[])
    BoundaryProblem{problem_type}(name, parent_field_name, parent_field_dim, dim, elements, preprocessors, postprocessors)
end


function add_postprocessor!(problem::Union{FieldProblem, BoundaryProblem}, postprocessor_name::Symbol, args...; kwargs...)
    push!(problem.postprocessors, (postprocessor_name, args, kwargs))
end

function add_preprocessor!(problem::Union{FieldProblem, BoundaryProblem}, postprocessor_name::Symbol, args...; kwargs...)
    push!(problem.preprocessors, (postprocessor_name, args, kwargs))
end

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
function get_unknown_field_name{P<:AbstractProblem}(problem::Problem{P})
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
#       info("gdofs = $gdofs")
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
