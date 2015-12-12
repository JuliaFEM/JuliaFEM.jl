# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

# Lagrange (Continous Galerkin) finite elements

abstract CG <: AbstractElement

"""
Given polynomial P and coordinates of reference element, calculate
Lagrange basis functions
"""
function calculate_lagrange_basis_coefficients(P, X)
    dim, nbasis = size(X)
    A = zeros(nbasis, nbasis)
    for i=1:nbasis
        A[i,:] = P(X[:, i])
    end
#   invA = inv(A)'
#   basis(xi) = (invA*P(xi))'
#   dbasisdxi(xi) = (ForwardDiff.jacobian((xi) -> invA*P(xi), xi, cache=autodiffcache))'
#   basis, dbasisdxi
#   info(inv(A))
#   info(P([0.0, 0.0]))
#   invA = inv(A)'
#   basis(xi) = invA*P(xi)
#   return basis
    return inv(A)'
end

function refcoords(X::Matrix)
    return Vector{Float64}[X[:,i] for i=1:size(X,2)]
end

"""
Create new Lagrange element

Examples
--------
>>> @create_lagrange_element(Seg2, "2 node linear segment", X, P)
"""
macro create_lagrange_element(element_name, element_description, X, P)
    eltype = esc(element_name)
    quote
        global get_basis, get_dbasis,
               get_reference_element_coordinates,
               get_reference_element_midpoint

        #basis, dbasis = calculate_lagrange_basis($P, $X)
        C = calculate_lagrange_basis_coefficients($P, $X)
        basis(xi) = C*$P(xi)
#       dbasis = ForwardDiff.jacobian(basis)

        abstract $eltype <: CG

        function get_basis(::Type{$eltype}, xi::Vector)
            return basis(xi)'
        end

        XX = refcoords($X)
        function get_reference_element_coordinates(::Type{$eltype})
            return XX
        end

        XXX = vec(mean($X, 2))
        function get_reference_element_midpoint(::Type{$eltype})
            return XXX
        end
#=
        function get_dbasis(::Type{$eltype}, xi::Vector{Float64})
            return dbasis(xi)'
        end
=#

        function $eltype(args...)
            return Element{$eltype}(args...)
        end

        function Base.size(::Type{$eltype})
            return Base.size($X)
        end

    end
end

# 1d Lagrange elements

@create_lagrange_element(Seg2, "2 node linear line element",
    [-1.0 1.0], (xi) -> [1.0, xi[1]])

@create_lagrange_element(Seg3, "3 node quadratic line element",
    [-1.0 1.0 0.0], (xi) -> [1.0, xi[1], xi[1]^2])

# 2d Lagrange elements

@create_lagrange_element(Tri3, "3 node bilinear triangle element",
    [0.0 1.0 0.0
     0.0 0.0 1.0],
    (xi) -> [1.0, xi[1], xi[2]])

#function get_reference_element_midpoint(::Type{Tri3})
#    return [1.0/3.0, 1.0/3.0]
#end

@create_lagrange_element(Tri6, "6 node quadratic triangle element",
    [0.0 1.0 0.0 0.5 0.5 0.0
     0.0 0.0 1.0 0.0 0.5 0.5],
    (xi) -> [1.0, xi[1], xi[2], xi[1]^2, xi[2]^2, xi[1]*xi[2]])

@create_lagrange_element(Quad4, "4 node bilinear quadrangle element",
    [-1.0  1.0 1.0 -1.0
     -1.0 -1.0 1.0  1.0],
    (xi) -> [1.0, xi[1], xi[2], xi[1]*xi[2]])

# 3d Lagrange elements

@create_lagrange_element(Tet4, "4 node tetrahedron",
    [0.0 1.0 0.0 0.0
     0.0 0.0 1.0 0.0
     0.0 0.0 0.0 1.0],
    (xi) -> [1.0, xi[1], xi[2], xi[3]])

@create_lagrange_element(Tet10, "10 node quadratic tetrahedron",
    [0.0 1.0 0.0 0.0 0.5 0.5 0.0 0.0 0.5 0.0
     0.0 0.0 1.0 0.0 0.0 0.5 0.5 0.0 0.0 0.5
     0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.5 0.5 0.5],
    (xi) -> [    1.0,   xi[1],       xi[2],       xi[3],     xi[1]^2,
             xi[2]^2, xi[3]^2, xi[1]*xi[2], xi[2]*xi[3], xi[3]*xi[1]])

