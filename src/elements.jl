# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

#=
Related notebooks
-----------------

2015-08-29-developing-juliafem.ipynb
=#

using JuliaFEM: interpolate
using FactCheck
using ForwardDiff


abstract Element

#= ELEMENT DEFINITIONS

TODO: rewrite instructions after notebook.

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
- get_dbasisdxi *
- get_dbasisdX
- get_field
- set_field
- interpolate
- ...

Which should work if element is defined following some rules. Functions marked with asterisk * are the ones which must necessarily to implement by your own.

Start of example
----------------

This is example how to create new element. This is commented because I use code
generation for simple elements like Lagrage elements. Feel free to use
code generation but elements can be of course created manually too!

abstract CG <: Element # create new element family "Continous Galerkin"

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

# These must be implemented for your own element
get_number_of_basis_functions(el::Type{Element}) = nothing
get_element_dimension(el::Type{Element}) = nothing

### LAGRANGE ELEMENTS ###
#include("lagrange.jl")

### HIERARCHICAL P-ELEMENTS ###
#include("hierarchical.jl")

### COMMON ELEMENT ROUTINES ###

"""
Test routine for element. If this passes, element interface is properly
defined.

Parameters
----------
eltype::Type{Element}
    Element to test

Raises
------
This uses FactCheck and throws exceptions if element is not passing all tests.
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

    dim = get_element_dimension(eltype)
    Logging.info("Element dimension: $dim")
    @fact dim --> not(nothing) """
    Unable to get element dimension define function 'get_element_dimension'
    which return the dimension of this element (1, 2, 3)"""

    # try to interpolate some scalar field
    fld = Field(0.0, collect(1:n))
    Logging.info("Creating new scalar field $fld")
    Logging.info("Pushing field to element.")
    new_fieldset!(el, "field1")
    add_field!(el, "field1", fld)
    fieldset = get_fieldset(el, "field1")
    @fact fieldset[1] --> fld

    mid = zeros(dim)
    try
        get_basis(el)(mid)
    catch
        Logging.error("""
        Unable to evaluate basis, define function 'get_basis' for
        this element.""")
    end
    try
        get_dbasisdxi(el)(mid)
    catch
        Logging.error("""
        Unable to evaluate partial derivatives of basis,
        define function 'get_dbasisdxi' for this element.""")
    end

    Logging.info("Interpolating scalar field at $mid")
    #f(field, xi, t) = el(xi)*el[field](t)
    #i = f(:field1, mid, 0.0)
    i = interpolate(el, "field1", mid, 0.0)
    Logging.info("Value: $i")
    Logging.info("Element $eltype passed tests.")
end

get_connectivity(el::Element) = el.connectivity

"""
Get basis functions of element.
"""
get_basis(el::Element) = el.basis
get_basis(el::Element, xi::Vector) = el.basis(xi)
Base.call(el::Element, xi::Vector) = el.basis(xi)

"""
Get partial derivatives of basis functions of element.
"""
get_dbasisdxi(el::Element) = el.basis.dbasisdxi
get_dbasisdxi(el::Element, xi::Vector) = el.basis.dbasisdxi(xi)

"""
Interpolate field on element.
"""
function interpolate(el::Element, field_name::Union{Symbol, ASCIIString}, xi::Vector, t::Number)
    fieldset = get_fieldset(el, symbol(field_name))
    field = interpolate(fieldset, t)
    basis = get_basis(el)
    interpolate(basis, field, xi)
end

"""
Interpolate derivative of field on element.
"""
function dinterpolate(el::Element, field_name::Union{Symbol, ASCIIString}, xi::Vector, t::Number)
    #get_dbasisdxi(el, xi)*el[field](t)
    fieldset = get_fieldset(el, symbol(field_name))
    field = interpolate(fieldset, t)
    basis = get_basis(el)
    dinterpolate(basis, field, xi)
end

"""
Get jacobian of element evaluated at point ξ on element in reference configuration.

Parameters
----------
el :: Element
xi :: Vector
geometry_field :: Any, optional
time :: Number

Returns
-------
Vector or Matrix
    depending on element type

"""
function get_jacobian(el::Element, xi, t, geometry_field=symbol("geometry"))
    dinterpolate(el, geometry_field, xi, t)
end

"""
Evaluate partial derivatives of basis, dbasis/dX
"""
function get_dbasisdX(el::Element, xi, t)
    dbasisdxi = get_dbasisdxi(el, xi)
    J = get_jacobian(el, xi, t)
    dbasisdxi*inv(J)
end


""" Create new empty set of fields for element. """
function new_fieldset!(el::Element, field_name::Union{Symbol, ASCIIString})
    el.fields[symbol(field_name)] = FieldSet()
end
function new_fieldset!(el::Element, field_name::Union{Symbol, ASCIIString}, field::Field)
    new_fieldset!(el, symbol(field_name))
    add_field!(el, symbol(field_name), field)
end


""" Add new field to fieldset of element. """
function add_field!(el::Element, field_name::Union{Symbol, ASCIIString}, field::Field)
    push!(el.fields[symbol(field_name)], field)
end


""" Get fieldset. """
function get_fieldset(el::Element, field_name::Union{Symbol, ASCIIString})
    el.fields[symbol(field_name)]
end
""" Get fieldset, convenient function. """
function Base.getindex(el::Element, field_name::Union{Symbol, ASCIIString})
    get_fieldset(el, field_name)
end




"""
calculate "local" normals in elements, in a way that
n = Nᵢnᵢ gives some reasonable results for ξ ∈ [-1, 1]
"""
function calculate_normals!(el::Element, t, field_name=symbol("normals"))
    new_field!(el, field_name, Vector)
    for xi in Vector[[-1.0], [1.0]]
        t = dinterpolate(el, :Geometry, xi)
        n = [0 -1; 1 0]*t
        n /= norm(n)
        push_field!(el, field_name, n)
    end
end

"""
Alter normal field such that normals of adjacent elements are averaged.
"""
function average_normals!(elements, normal_field=symbol("normals"))
    d = Dict()
    for el in elements
        c = get_connectivity(el)
        n = get_field(el, normal_field)
        for (ci, ni) in zip(c, n)
            d[ci] = haskey(d, ci) ? d[ci] + ni : ni
        end
    end
    for (ci, ni) in d
        d[ci] /= norm(d[ci])
    end
    for el in elements
        c = get_connectivity(el)
        new_normals = [d[ci] for ci in c]
        set_field(el, normal_field, new_normals)
    end
end


# FIXME: These two needs integration -- maybe not in elements.jl ..?
"""
Fit field s.t. || ∫ (Nᵢ(ξ)αᵢ - f(el, ξ)) dS || -> min!

Parameters
----------
f::Function
    Needs to take (el::Element, xi::Vector) as argument
fixed_coeffs::Int[]
    These coefficients are not changed during fitting -> constrained optimizatio
"""
function fit_field!(el::Element, field, f, fixed_coeffs=Int[])
    w = [
        128/225,
        (332+13*sqrt(70))/900,
        (332+13*sqrt(70))/900,
        (332-13*sqrt(70))/900,
        (332-13*sqrt(70))/900]
    xi = Vector[
        [0.0],
        [ 1/3*sqrt(5 - 2*sqrt(10/7))],
        [-1/3*sqrt(5 - 2*sqrt(10/7))], 
        [ 1/3*sqrt(5 + 2*sqrt(10/7))],
        [-1/3*sqrt(5 + 2*sqrt(10/7))]]
    n = get_number_of_basis_functions(el)
    fld = get_field(el, field)
    nfld = length(fld[1])
    #Logging.debug("dim of field $field: $nfld")

    M = zeros(n, n)
    b = zeros(n, nfld)
    for i=1:length(w)
        detJ = get_detJ(el, xi[i])
        N = get_basis(el, xi[i])
        M += w[i]*N*N'*detJ
        fi = f(el, xi[i])
        for j=1:nfld
            b[:, j] += w[i]*N*fi[j]*detJ
        end
    end

    coeffs = zeros(n)
    for j=1:nfld
        for k=1:n
            coeffs[k] = fld[k][j]
        end
        if length(fixed_coeffs) != 0
            # constrained problem, some coefficients are fixed
            N = Int[] # rest of coeffs
            S = Int[] # fixed coeffs
            for i = 1:n
                if i in fixed_coeffs
                    push!(S, i)
                else
                    push!(N, i)
                end
            end
            lhs = M[N,N]
            rhs = b[N,j] - M[N,S]*coeffs[S]
            coeffs[N] = lhs \ rhs
        else
            coeffs[:] = M \ b[:,j]
        end
        for k=1:n
            fld[k][j] = coeffs[k]
        end
    end
    set_field(el, field, fld)
    return
end


"""
Fit field s.t. || ∫ ∂/∂ξ(∑Nᵢ(ξ)αᵢ)f(el, ξ) dS || -> min!
"""
function fit_derivative_field!(el::Element, field, f, fixed_coeffs=Int[])
    w = [
        128/225,
        (332+13*sqrt(70))/900,
        (332+13*sqrt(70))/900,
        (332-13*sqrt(70))/900,
        (332-13*sqrt(70))/900]
    xi = Vector[
        [0.0],
        [ 1/3*sqrt(5 - 2*sqrt(10/7))],
        [-1/3*sqrt(5 - 2*sqrt(10/7))], 
        [ 1/3*sqrt(5 + 2*sqrt(10/7))],
        [-1/3*sqrt(5 + 2*sqrt(10/7))]]
    n = get_number_of_basis_functions(el)
    fld = get_field(el, field)
    nfld = length(fld[1])
    #Logging.debug("dim of field $field: $nfld")

    M = zeros(n, n)
    b = zeros(n, nfld)
    for i=1:length(w)
        detJ = get_detJ(el, xi[i])
        dNdxi = get_dbasisdxi(el, xi[i])
        dNdX = dNdxi / detJ
        M += w[i]*dNdX*dNdX'*detJ
        fi = f(el, xi[i])
        for j=1:nfld
            b[:, j] += w[i]*dNdX*fi[j]*detJ
        end
    end

    coeffs = zeros(n)
    for j=1:nfld
        for k=1:n
            coeffs[k] = fld[k][j]
        end
        if length(fixed_coeffs) != 0
            #Logging.info("constrained problem, some coefficients are fixed")
            N = Int[] # rest of coeffs
            S = Int[] # fixed coeffs
            for i = 1:n
                if i in fixed_coeffs
                    push!(S, i)
                else
                    push!(N, i)
                end
            end
            lhs = M[N,N]
            rhs = b[N,j] - M[N,S]*coeffs[S]
            coeffs[N] = lhs \ rhs
        else
            coeffs[:] = M \ b[:,j]
        end
        for k=1:n
            fld[k][j] = coeffs[k]
        end
    end
    set_field(el, field, fld)
    return
end


""" Find projection from slave nodes to master element. """
function calc_projection_slave_nodes_to_master_element(sel, mel)
    X1 = get_field(sel, :Geometry)
    N1 = get_field(sel, :Normals)
    X2(xi) = interpolate(mel, :Geometry, xi)
    dX2(xi) = dinterpolate(mel, :Geometry, xi)
    R(xi, k) = det([X2(xi) - X1[k] N1[k]]')
    dR(xi, k) = det([dX2(xi) N1[k]]')
    xi2 = Vector[[0.0], [0.0]]
    for k=1:2
        xi = xi2[k]
        for i=1:3
            dxi = -R(xi, k)/dR(xi, k)
            xi += dxi
            if abs(dxi) < 1.0e-9
                break
            end
        end
        xi2[k] = xi
    end
    clamp!(xi2, -1, 1)
    return xi2
end

""" Find projection from master nodes to slave element. """
function calc_projection_master_nodes_to_slave_element(sel, mel)
    X1(xi) = interpolate(sel, :Geometry, xi)
    dX1(xi) = dinterpolate(sel, :Geometry, xi)
    N1(xi) = interpolate(sel, :Normals, xi)
    dN1(xi) = dinterpolate(sel, :Normals, xi)
    X2 = get_field(mel, :Geometry)
    R(xi, k) = det([X1(xi) - X2[k] N1(xi)]')
    dR(xi, k) = det([dX1(xi) N1(xi)]') + det([X1(xi) - X2[k] dN1(xi)]')
    xi1 = Vector[[0.0], [0.0]]
    for k=1:2
        xi = xi1[k]
        for i=1:3
            dxi = -R(xi, k)/dR(xi, k)
            xi += dxi
            if abs(dxi) < 1.0e-9
                break
            end
        end
        xi1[k] = xi
    end
    clamp!(xi1, -1, 1)
    return xi1
end

function has_projection(sel, mel)
    xi1 = calc_projection_master_nodes_to_slave_element(sel, mel)
    l = abs(xi1[2]-xi1[1])[1]
    return l > 1.0e-9
end

"""
Calculate projection between 1d boundary elements
"""
function calc_projection(sel, mel)
    xi1 = calc_projection_master_nodes_to_slave_element(sel, mel)
    xi2 = calc_projection_slave_nodes_to_master_element(sel, mel)
    return xi1, xi2
end

