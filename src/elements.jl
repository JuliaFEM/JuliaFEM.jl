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
get_number_of_basis_functions(el::Element) = nothing
get_element_dimension(el::Element) = nothing
get_basis(el::Element, xi) = nothing
get_dbasisdxi(el::Element, xi) = nothing
get_connectivity(el::Element) = el.connectivity

### LAGRANGE ELEMENTS ###
include("lagrange.jl")

### HIERARCHICAL P-ELEMENTS ###
include("hierarchical.jl")

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
Get jacobian of element evaluated at point ξ on element in reference configuration.

Parameters
----------
el::Element
xi::Vector
geometry_field::Any, optional

Returns
-------
Vector or Matrix
    depending on element type

Notes
-----
Big "J" comes from reference (undeformed) configuration.
"""
function get_Jacobian(el::Element, xi, geometry_field=:Geometry)
    dinterpolate(el, geometry_field, xi)
end


"""
Get jacobian of element evaluated at point ξ on element in current configuration.

Notes
-----
Small "j" comes from current (deformed) configuration.
"""
function get_jacobian(el::Element, xi, geometry_field=:Geometry, displacement_field=:displacement)
    dbasisdxi = get_dbasisdxi(el, xi)
    X = get_field(el, geometry_field)
    u = get_field(el, displacement_field)
    j = (X+u)*dbasisdxi
    return j
end


"""
Evaluate partial derivatives of basis, dbasis/dX
"""
function get_dbasisdX(el::Element, xi)
    dbasisdxi = get_dbasisdxi(el, xi)
    J = get_Jacobian(el, xi)
    dbasisdxi*inv(J)
end


"""
Evaluate partial derivatives of basis, dbasis/dx
"""
function get_dbasisdx(el::Element, xi)
    dbasisdxi = get_dbasisdxi(el, xi)
    j = get_jacobian(el, xi)
    dbasisdxi*inv(j)
end


""" Set field variable. """
function set_field(el::Element, field_name, field_value)
    el.fields[field_name] = field_value
end


""" Create new empty field of some type. """
function new_field!(el::Element, field_name, field_type)
    el.fields[field_name] = field_type[]
end


""" Push to existing field. """
function push_field!(el::Element, field_name, field_value)
    push!(el.fields[field_name], field_value)
end


""" Get field variable. """
function get_field(el::Element, field_name)
    el.fields[field_name]
end


"""
Evaluate some field in point ξ on element using basis functions.

Parameters
----------
el :: Element
field :: Any
xi :: Vector

Returns
-------
Scalar, Vector, Tensor, depending on what is type of field to interpolate.

Notes
-----
This has another version which returns multiple values for set of coordinates {ξᵢ}.
dinterpolate returns derivatives.

Examples
--------
>>> field = [1.0, 2.0, 3.0, 4.0]
>>> set_field(el, :temperature, field)
>>> interpolate(el, :temperature, [0.0, 0.0])
15.0
"""
function interpolate(el::Element, field, xi::Number)
    interpolate(el, field, [xi])
end
function interpolate(el::Element, field, xi::Vector)
    field = get_field(el, field)
    sum(get_basis(el, xi) .* field)
end
function interpolate(el::Element, field, xis::Array{Vector, 1})
    field = get_field(el, field)
    interpolate_(xi) = sum(get_basis(el, xi) .* field)
    map(interpolate_, xis)
end

"""
"""
function dinterpolate(el::Element, field, xi::Number)
    dinterpolate(el, field, [xi])
end
function dinterpolate(el::Element, field, xi::Vector)
    fld = get_field(el, field)
    dbasis = get_dbasisdxi(el, xi)
    if isa(dbasis, Vector)
        return sum(dbasis .* fld)
    end
    return sum([fld[i]*dbasis[i,:] for i in 1:length(fld)])
end

"""
calculate "local" normals in elements, in a way that
n = Nᵢnᵢ gives some reasonable results for ξ ∈ [-1, 1]
"""
function calculate_normals!(el::Element, field_name=:Normals)
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
function average_normals!(elements, normal_field=:Normals)
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

