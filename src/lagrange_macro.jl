# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

# Lagrange (Continous Galerkin) finite elements shape functions generated using macro.

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
    return inv(A)
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
        global get_basis, length, size
        #=
               get_reference_element_coordinates,
               get_reference_element_midpoint
        =#

        type $eltype <: AbstractElement
        end

        A = calculate_lagrange_basis_coefficients($P, $X)
        #basis(xi) = C*$P(xi)

        function get_basis(element::Element{$eltype}, ip, time)
            return transpose($P(ip))*A
        end

        function size(element::Element{$eltype})
            return size($X)
        end

        function length(element::Element{$eltype})
            return size($X, 2)
        end

        #=
        XX = refcoords($X)
        function get_reference_element_coordinates(::Type{$eltype})
            return XX
        end

        XXX = vec(mean($X, 2))
        function get_reference_element_midpoint(::Type{$eltype})
            return XXX
        end

        function $eltype(args...)
            return Element{$eltype}(args...)
        end

        =#

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

@create_lagrange_element(Tri6, "6 node quadratic triangle element",
    [0.0 1.0 0.0 0.5 0.5 0.0
     0.0 0.0 1.0 0.0 0.5 0.5],
    (xi) -> [1.0, xi[1], xi[2], xi[1]^2, xi[2]^2, xi[1]*xi[2]])

@create_lagrange_element(Quad4, "4 node bilinear quadrangle element",
    [-1.0  1.0  1.0 -1.0
     -1.0 -1.0  1.0  1.0],
    (xi) -> [1.0, xi[1], xi[2], xi[1]*xi[2]])

@create_lagrange_element(Quad9, "9 node bilinear quadrangle element",
    [-1.0  1.0  1.0 -1.0  0.0  1.0  0.0 -1.0
     -1.0 -1.0  1.0  1.0 -1.0  0.0  1.0  0.0],
    (xi) -> [1.0,     xi[1],   xi[2],         xi[1]*xi[2],
             xi[1]^2, xi[2]^2, xi[1]^2*xi[2], xi[1]*xi[2]^2])

# 3d Lagrange elements

@create_lagrange_element(Hex8, "8 node hexahedra",
    [-1.0  1.0  1.0 -1.0 -1.0  1.0 1.0 -1.0
     -1.0 -1.0  1.0  1.0 -1.0 -1.0 1.0  1.0
     -1.0 -1.0 -1.0 -1.0  1.0  1.0 1.0  1.0],
    (xi) -> [1.0, xi[1], xi[2], xi[1]*xi[2], xi[3],
             xi[1]*xi[3], xi[2]*xi[3], xi[1]*xi[2]*xi[3]])

#=
@create_lagrange_element(Hex20, "20 node hexahedra",
    [
        -1.0  1.0  1.0 -1.0 -1.0  1.0  1.0 -1.0  0.0  1.0  0.0 -1.0 -1.0  1.0  1.0 -1.0  0.0  1.0  0.0 -1.0
        -1.0 -1.0  1.0  1.0 -1.0 -1.0  1.0  1.0 -1.0  0.0  1.0  0.0 -1.0 -1.0  1.0  1.0 -1.0  0.0  1.0  0.0
        -1.0 -1.0 -1.0 -1.0  1.0  1.0  1.0  1.0 -1.0 -1.0 -1.0 -1.0  0.0  0.0  0.0  0.0  1.0  1.0  1.0  1.0
    ],
    (xi) -> [1.0, xi[1], xi[2], xi[1]*xi[2], xi[3], xi[1]*xi[3], xi[2]*xi[3], xi[1]*xi[2]*xi[3]
             x[1]^2, 
    ])
=#

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

function get_reference_element_midpoint{E}(element::Element{E})
    get_reference_element_midpoint(E)
end
