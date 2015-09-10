# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using FactCheck
using ForwardDiff

abstract Element

#= ELEMENT DEFINITIONS

Each element must have

1. Connectivity information. How element is connected to other elements.
   This is typically node ids in Lagrange elements.
2. Ability to store fields, in array of shape dim × nnodes, where dim is
   dimension of field and nnodes is number of nodes of element. Note that
   this is always 2d array.
3. Default constructor which takes connectivity as argument.
4. Basis functions and derivative of basis functions.

These rules probably will change, but there's a test_element function which
tests element and that it obeys current rules. If test_element passes,
everything should be ok. I use Quad4 as an example element here.
Several functions are inherited from Element abstract type:

- get_connectivity
- get_number_of_basis_functions*
- get_element_dimension *
- get_basis *
- get_dbasisdxi*
- get_dbasisdX
- get_field
- set_field
- interpolate
- ...

Which should work if element is defined following some rules. Functions marked with asterisk * are the ones which must necessarily to implement by your own.

=#

# These must be implemented for your own element
get_number_of_basis_functions(el::Type{Element}) = nothing
get_element_dimension(el::Element) = nothing
get_basis(el::Element, xi) = nothing
get_dbasisdxi(el::Element, xi) = nothing

"""
Create new element with element_name to family element_family

Examples
--------
>>> @create_element(Seg2, CG, "2 node linear segment")
"""
macro create_element(element_name, element_family, element_description)
    print("Creating element ", element_name, ": ", element_description, "\n")
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

#=

Start of example

Example how to create new element. This is commented because I use code
generation for simple elements like Lagrage elements. Feel free to use
code generation but elements can be of course created manually too!

type Quad4 <: CG
    connectivity :: Array{Int, 1}
    fields :: Dict{Any, Any}
end

""" Default contructor. """
Quad4(connectivity) = Quad4(connectivity, Dict{Any, Any}())

""" Return number of basis functions of this element. """
get_number_of_basis_functions(el::Type{Quad4}) = 4

""" Return element dimension (length of xi vector). """
get_element_dimension(el::Type{Quad4}) = 2

""" Return basis functions for this element (xi dim = 2, functions = 4). """
function get_basis(el::Quad4, xi)
    [(1-xi[1])*(1-xi[2])/4
     (1+xi[1])*(1-xi[2])/4
     (1+xi[1])*(1+xi[2])/4
     (1-xi[1])*(1+xi[2])/4]
end

""" Return partial derivatives of basis functions. """
function get_dbasisdxi(el::Quad4, xi)
    [-(1-xi[2])/4.0    -(1-xi[1])/4.0
      (1-xi[2])/4.0    -(1+xi[1])/4.0
      (1+xi[2])/4.0     (1+xi[1])/4.0
     -(1+xi[2])/4.0     (1-xi[1])/4.0]
end

End of example.

=#

abstract CG <: Element # Lagrange (continous Galerkin) element family

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
    println("Calculating inverse of A")
    invA = inv(A)'
    basis(xi) = invA*P(xi)
    dbasisdxi = ForwardDiff.jacobian(basis)
    basis, dbasisdxi
end

"""
Assign Lagrange basis for element.
"""
macro create_lagrange_basis(element_name, X, P)
    
    print("Creating Lagrange basis for element ", element_name, ". ")
    eltype = esc(element_name)

    quote

        global get_number_of_basis_functions, get_element_dimension
        global get_basis, get_dbasisdxi

        dim = size($X, 1)
        nbasis = size($X, 2)
        print("Number of basis functions: ", nbasis, ". ")
        println("Element dimension: ", dim)

        get_number_of_basis_functions(el::Type{$(esc(element_name))}) = nbasis
        get_element_dimension(el::$(esc(element_name))) = dim

        basis, dbasisdxi = calculate_lagrange_basis($P, $X)
        get_basis(el::$eltype, xi) = basis(xi)
        get_dbasisdxi(el::$eltype, xi) = dbasisdxi(xi)
        println("Element ", $element_name, " created.")
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

# Common element routines


"""
Test routine for element.

Parameters
----------
eltype::Type{Element}
    Element to test

Raises
------
This uses FactCheck and throws exception if element is not passing.
"""
function test_element(eltype)
    Logging.info("Testing element $eltype")
    local el
    n = get_number_of_basis_functions(eltype)
    Logging.info("number of basis functions in this element: $n")
    @fact n --> not(nothing) """
    Unable to determine number of nodes for $eltype define a function
    'get_number_of_basis_functions' which returns the number of nodes
    for this element."""

    Logging.info("Initializing element")
    try
        el = eltype(collect(1:n))
    catch
        Logging.error("""
        Unable to create element with default constructor define function
        $eltype(connectivity) which initializes this element.""")
        return false
    end

    dim = get_element_dimension(el)
    Logging.info("Element dimension: $dim")
    @fact dim --> not(nothing) """
    Unable to get element dimension define function 'get_element_dimension'
    which return the dimension of this element (1, 2, 3)"""

    # try to interpolate some scalar field
    fld = collect(1:n)'
    Logging.info("Setting scalar field $fld to element.")
    set_field(el, "field1", fld)
    @fact get_field(el, "field1") --> fld

    try
        get_basis(el, zeros(dim))
    catch
        Logging.error("""
        Unable to evaluate basis, define function 'get_basis' for
        this element.""")
    end
    try
        get_dbasisdxi(el, zeros(dim))
    catch
        Logging.error("""
        Unable to evaluate partial derivatives of basis,
        define function 'get_dbasisdxi' for this element.""")
    end

    xi = zeros(dim)
    Logging.info("Interpolating scalar field at $xi")
    i = interpolate(el, "field1", zeros(dim))
    Logging.info("Value: $i")
    Logging.info("Element $eltype passed tests.")
end

"""
Get jacobian of element evaluated at point ξ on element.

Notes
-----
This function assumes that element has field :geometry defined.
"""
function get_jacobian(el::Element, xi)
    dbasisdxi = get_dbasisdxi(el, xi)
    X = get_field(el, :geometry)
    J = X*dbasisdxi
    return J
end

"""
Evaluate partial derivatives of basis function w.r.t
material description X, i.e. dbasis/dX
"""
function get_dbasisdX(el::Element, xi)
    dbasisdxi = get_dbasisdxi(el, xi)
    J = get_jacobian(el, xi)
    dbasisdxi*inv(J)
end

""" Set field variable. """
function set_field(el::Element, field_name, field_value)
    el.fields[field_name] = field_value
end

""" Get field variable. """
function get_field(el::Element, field_name)
    el.fields[field_name]
end

""" Evaluate some field in point ξ on element using basis functions. """
function interpolate(el::Element, field, xi)
    f = get_field(el, field)
    basis = get_basis(el, xi)
    dim, nnodes = size(f)
    result = zeros(dim)
    for i=1:nnodes
        result += basis[i]*f[:,i]
    end
    if dim == 1
        return result[1]
    else
        return result
    end
end
