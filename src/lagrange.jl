# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

# Lagrange (Continous Galerkin) finite elements

abstract CG <: Element

"""
Create new element with element_name to family element_family

Examples
--------
>>> @create_element(Seg2, CG, "2 node linear segment")
"""
macro create_element(element_name, element_family, element_description)
#   Logging.debug("Creating element ", element_name, ": ", element_description, "\n")
    eltype = esc(element_name)
    elfam = esc(element_family)
    quote
        global get_element_description
        type $eltype <: $elfam
            connectivity :: Array{Int, 1}
            fields :: Dict{Any, Any}
        end
        $eltype(connectivity) = $eltype(connectivity, Dict{Any, Any}())
        get_element_description(el::Type{$eltype}) = $element_description
    end
end

"""
Given polynomial P and coordinates of reference element, calculate
Lagrange basis function and partial derivatives.
"""
function calculate_lagrange_basis(P, X)
    dim, nbasis = size(X)
    A = zeros(nbasis, nbasis)
    for i=1:nbasis
        A[i,:] = P(X[:, i])
    end
#   Logging.debug("Calculating inverse of A")
    invA = inv(A)'
    basis(xi) = invA*P(xi)
    dbasisdxi = ForwardDiff.jacobian(basis)
    basis, dbasisdxi
end

"""
Assign Lagrange basis for element.
"""
macro create_lagrange_basis(element_name, X, P)

#   Logging.debug("Creating Lagrange basis for element ", element_name, ". ")
    eltype = esc(element_name)

    quote

        global get_number_of_basis_functions, get_element_dimension
        global get_basis, get_dbasisdxi

        dim = size($X, 1)
        nbasis = size($X, 2)
#       Logging.debug("Number of basis functions: ", nbasis, ". ")
#       Logging.debug("Element dimension: ", dim)

        get_number_of_basis_functions(el::Type{$(esc(element_name))}) = nbasis
        get_number_of_basis_functions(el::$(esc(element_name))) = nbasis
        get_element_dimension(el::$(esc(element_name))) = dim

        basis, dbasisdxi = calculate_lagrange_basis($P, $X)
        get_basis(el::$eltype, xi) = basis(xi)
        get_dbasisdxi(el::$eltype, xi) = dbasisdxi(xi)
#       Logging.debug("Element ", $element_name, " created.")
    end

end

# 0d Lagrange element

@create_element(Point1, CG, "1 node point element")

# 1d Lagrange elements

@create_element(Seg2, CG, "2 node linear line element")
@create_lagrange_basis(Seg2, [-1.0 1.0], (xi) -> [1.0, xi[1]])

@create_element(Seg3, CG, "3 node quadratic line element")
@create_lagrange_basis(Seg3, [-1.0 1.0 0.0], (xi) -> [1.0, xi[1], xi[1]^2])

# 2d Lagrange elements

@create_element(Tri3, CG, "3 node bilinear triangle element")
@create_lagrange_basis(Tri3,
    [0.0 1.0 0.0
     0.0 0.0 1.0],
    (xi) -> [1.0, xi[1], xi[2]])

@create_element(Quad4, CG, "4 node bilinear quadrangle element")
@create_lagrange_basis(Quad4,
    [-1.0  1.0 1.0 -1.0
     -1.0 -1.0 1.0  1.0],
    (xi) -> [1.0, xi[1], xi[2], xi[1]*xi[2]])

# 3d Lagrange elements

@create_element(Tet10, CG, "10 node quadratic tetrahedron")
@create_lagrange_basis(Tet10,
    [0.0 1.0 0.0 0.0 0.5 0.5 0.0 0.0 0.5 0.0
     0.0 0.0 1.0 0.0 0.0 0.5 0.5 0.0 0.0 0.5
     0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.5 0.5 0.5],
    (xi) -> [    1.0,   xi[1],       xi[2],       xi[3],     xi[1]^2,
             xi[2]^2, xi[3]^2, xi[1]*xi[2], xi[2]*xi[3], xi[3]*xi[1]])

