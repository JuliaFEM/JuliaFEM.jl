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
- get_number_of_basis_functions
- get_element_dimension *
- get_basis *
- get_dbasisdxi
- get_dbasisdX
- get_field
- set_field
- interpolate
- ...

Which should work if element is defined following some rules. Functions marked with asterisk * are the ones which must necessarily to implement by your own.

=#

# These must be implemented for your own element
get_element_dimension(el::Element) = nothing
get_basis(el::Element, xi) = nothing
""" Return partial derivatives of basis using ForwardDiff """
function get_dbasisdxi(el::Element, xi)
    f(xi) = get_basis(el, xi)
    ForwardDiff.jacobian(f, xi)
end
function get_number_of_basis_functions(el::Type{Element})
    Logging.info("You really should define get_number_of_basis_functions")
    # this is hack, evaluate basis in some point and return length of vector.
    bf = get_basis([0.0, 0.0, 0.0])
    length(bf)
end

abstract CG <: Element # Lagrange (continous Galerkin) element family

"""
4 node bilinear quadrangle element

X = [-1.0  1.0  1.0 -1.0
     -1.0 -1.0  1.0  1.0]

P(xi) = [1.0, xi[1], xi[2], xi[1]*xi[2]]

dP(xi) = [0.0 1.0 0.0 xi[2]
          0.0 0.0 1.0 xi[1]]
"""
type Quad4 <: CG
    connectivity :: Array{Int, 1}
    fields :: Dict{Any, Any}
end

""" Default contructor. """
Quad4(connectivity) = Quad4(connectivity, Dict{Any, Any}())

""" Return number of basis functions of this element. """
get_number_of_basis_functions(el::Type{Quad4})

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
# TODO: create_lagrange_element(:Quad4, X, P, dP)

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
    local el
    n = get_number_of_nodes(eltype)
    Logging.info("number of connectivity points (nodes) in this element: $n")
    @fact n --> not(-1) """Unable to determine number of nodes for $eltype
    define a function 'get_number_of_nodes' which returns the number of nodes for this element."""

    Logging.info("Constructing element..")
    try
        el = eltype(collect(1:n))
    catch
        Logging.error("""Unable to create element with default constructor
        define function $eltype(connectivity) which initializes this element.
        """)
        return false
    end
    dim = get_element_dimension(el)
    Logging.info("Element dimension: $dim")
    @fact dim --> not(-1) """Unable to get element dimension
    define function 'get_element_dimension' which return the dimension of this element (1, 2, 3)"""

    # try to interpolate some scalar field
    fld = collect(1:n)'
    Logging.info("Setting scalar field $fld to element.")
    set_field(el, "field1", fld)
    @fact get_field(el, "field1") --> fld

    try
        get_basis(el, zeros(dim))
    catch
        Logging.error("""Unable to evaluate basis, define function 'get_basis' for this element.
        """)
    end
    try
        get_dbasisdxi(el, zeros(dim))
    catch
        Logging.error("""Unable to evaluate partial derivatives of basis, define function 'get_dbasisdxi' for this element.
        """)
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

#=
"""
Create new Lagrange element

FIXME: this is not working

LoadError: error compiling anonymous: type definition not allowed inside a local scope

It's the for loop which is causing problems. See
https://github.com/JuliaLang/julia/issues/10555

"""
function create_lagrange_element(element_name, X, P, dP)

    @eval begin

        nnodes, dim = size(X)
        A = zeros(nnodes, nnodes)
        for i=1:nnodes
            A[i,:] = P(X[i,:])
        end
        invA = inv(A)'

        type $element_name
            node_ids :: Array{Int, 1}
            fields :: Dict{ASCIIString, Any}
        end

        function $element_name(node_ids)
            fields = Dict{ASCIIString, Any}()
            $element_name(node_ids, fields)
        end

        function get_basis(el::$element_name, xi)
            invA*P(xi)
        end

        function get_dbasisdxi(el::$element_name, xi)
            invA*dP(xi)
        end

        $element_name

    end

end

=#

# 0d Lagrange elements

#=
"""
1 node point element
"""
type Point1 <: CG
    node_ids :: Array{Int, 1}
    fields :: Dict{ASCIIString, Any}
end
function Point1(node_ids)
    fields = Dict{ASCIIString, Any}()
    Point1(node_ids, fields)
end

# 1d Lagrange elements

"""
2 node linear line element
"""
type Seg2 <: CG
    node_ids :: Array{Int, 1}
    fields :: Dict{ASCIIString, Any}
end

# X = [-1.0 1.0]'
# P = (xi) -> [1.0 xi[1]]'
# dP = (xi) -> [0.0 1.0]'
# create_lagrange_element(:Seg2, X, P, dP)

"""
3 node quadratic line element
"""
type Seg2 <: CG
    node_ids :: Array{Int, 1}
    fields :: Dict{ASCIIString, Any}
end
#X = [-1.0 1.0 0.0]'
#P = (xi) -> [1.0 xi[1] xi[1]^2]'
#dP = (xi) -> [0.0 1.0 2*xi[1]]'
#create_lagrange_element(:Seg3, X, P, dP)

# 2d Lagrange elements
=#

# 3d Lagrange elements
#=
"""
10 node quadratic tethahedron
"""
type Tet10 <: CG
    node_ids :: Array{Int, 1}
    fields :: Dict{ASCIIString, Any}
end
=#
# X = [
#    0.0 0.0 0.0
#    1.0 0.0 0.0
#    0.0 1.0 0.0
#    0.0 0.0 1.0
#    0.5 0.0 0.0
#    0.5 0.5 0.0
#    0.0 0.5 0.0
#    0.0 0.0 0.5
#    0.5 0.0 0.5
#    0.0 0.5 0.5]
#    P(xi) = [
#        1
#        xi[1]
#        xi[2]
#        xi[3]
#        xi[1]^2
#        xi[2]^2
#        xi[3]^2
#        xi[1]*xi[2]
#        xi[2]*xi[3]
#        xi[3]*xi[1]]
#    dP(xi) = [
#        0 0 0
#        1 0 0
#        0 1 0
#        0 0 1
#        2*xi[1] 0 0
#        0       2*xi[2] 0
#        0       0       2*xi[3]
#        xi[2]   xi[1]   0
#        0       xi[3]   xi[2]
#        xi[3]   0       xi[1]
#    ]
#create_lagrange_element(:Tet10, X, P, dP)
