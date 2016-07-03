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
        global get_basis, length, size, get_reference_coordinates
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

        XX = refcoords($X)
        function get_reference_coordinates(::Type{$eltype})
            return XX
        end

    end
end

# 3d Lagrange elements

function get_reference_element_midpoint{E}(element::Element{E})
    get_reference_element_midpoint(E)
end
