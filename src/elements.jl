# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using FactCheck

abstract Element

export get_number_of_nodes

get_number_of_nodes(el::Type{Element}) = -1
get_element_dimension(el::Element) = -1
get_dbasisdx(el::Element) = nothing
get_basis(el::Element) = nothing

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
Get jacobian of element evaluated at point xi
"""
function get_jacobian(el::Element, xi)
    dbasisdxi = get_dbasisdxi(el, xi)
    X = get_field(el, :geometry)
    #J = interpolate(X, dbasisdxi, xi)'
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

"""
Set field variable.
"""
function set_field(el::Element, field_name, field_value)
    el.fields[field_name] = field_value
end

"""
Get field variable.
"""
function get_field(el::Element, field_name)
    el.fields[field_name]
end

#"""
#Evaluate field in point xi using basis functions.
#"""
#function interpolate(el::Element, field::ASCIIString, xi::Array{Float64,1})
#    (get_basis(el, xi)'*get_field(el, field))'
#end

"""
Evaluate field in point xi using basis functions.
"""
function interpolate(el::Element, field::Union(ASCIIString, Symbol), xi::Array{Float64,1})
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

### Lagrange family ###

abstract CG <: Element # Lagrange element family

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

# 0d Lagrange elements

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

"""
4 node bilinear quadrangle element
"""
type Quad4 <: CG
    node_ids :: Array{Int, 1}
    fields :: Dict{ASCIIString, Any}
end
function Quad4(node_ids)
    fields = Dict{ASCIIString, Any}()
    Quad4(node_ids, fields)
end
function get_basis(el::Quad4, xi)
    [(1-xi[1])*(1-xi[2])/4
     (1+xi[1])*(1-xi[2])/4
     (1+xi[1])*(1+xi[2])/4
     (1-xi[1])*(1+xi[2])/4]
end
function get_dbasisdxi(el::Quad4, xi)
    [-(1-xi[2])/4.0    -(1-xi[1])/4.0
      (1-xi[2])/4.0    -(1+xi[1])/4.0
      (1+xi[2])/4.0     (1+xi[1])/4.0
     -(1+xi[2])/4.0     (1-xi[1])/4.0]
end
#X = [
#    -1.0 -1.0
#     1.0 -1.0
#     1.0  1.0
#    -1.0  1.0]
#P = (xi) -> [
#    1.0
#    xi[1]
#    xi[2]
#    xi[1]*xi[2]]
#dP = (xi) -> [
#      0.0   0.0
#      1.0   0.0
#      0.0   1.0
#    xi[2] xi[1]]
#create_lagrange_element(:Quad4, X, P, dP)

# 3d Lagrange elements

"""
10 node quadratic tethahedron
"""
type Tet10 <: CG
    node_ids :: Array{Int, 1}
    fields :: Dict{ASCIIString, Any}
end
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
