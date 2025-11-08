# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/FEMTruss.jl/blob/master/LICENSE

""" Truss implementation for JuliaFEM. """
# Truss element - consolidated from FEMTruss.jl



get_unknown_field_name,
get_formulation_type,
assemble!

"""
This truss fromulation is from
Concepts and applications of finite element analysis, forth edition
Chapter 2.4
Cook, Malkus, Plesha, Witt

# Features
- Nodal forces can be set using element `Poi1` with field
  `nodal force i`, where `i` is dof number.
- Displacements can be fixed using element `Poi1` with field
  `fixed displacement i`, where `i` is dof number.
- Temperature loads can be set by setting field `thermal expansion coefficient`
  and field `temperature difference`
"""
mutable struct Truss <: FieldProblem
end

function get_unknown_field_name(::Problem{Truss})
    return "displacement"
end

function get_formulation_type(::Problem{Truss})
    return :total
end

"""
get_truss_1d_and_Tg(element::Element{M,Seg2})
This function sets up the local truss 1d element needed for
the integration points to be used for the stiffness and forces
It also sets up the coordinate transformation matrix
Will need to change allocation strategy to pre-allocation later
"""
function get_truss_1d_and_Tg(element::Element{M,Seg2}, ndim) where M
    node_id1 = element.connectivity[1] #First node in element
    node_id2 = element.connectivity[2] #second node in elements
    pos = element("geometry")
    dl = pos[node_id2] - pos[node_id1]
    l = sqrt(dot(dl, dl))
    # Do the local element for calculations
    elem_1d = Element(Seg2, [1, 2])
    elem_1d.id = -1 # To signal it is local
    X = Dict{Int64,Vector{Float64}}(
        1 => [0.0],
        2 => [l])
    update!(elem_1d, "geometry", X)
    if ndim == 1
        T = eye(Float64, 2)
    elseif ndim == 2
        l_theta = dl[1] / l
        m_theta = dl[2] / l
        T = [l_theta m_theta 0 0; 0 0 l_theta m_theta]
    else # All three dimensions
        l_theta = dl[1] / l
        m_theta = dl[2] / l
        n_theta = dl[3] / l
        T = [l_theta m_theta n_theta 0 0 0; 0 0 0 l_theta m_theta n_theta]
    end
    return (elem_1d, T)
end

"""
   get_truss_1d_K(element::Element{M,Seg2}, nnodes, time)
   This function assembles the 1d truss stiffness matrix
   Will need to change allocation strategy to pre-allocation later
   Can discuss if we really need nnodes, since that is always 2 for trusses
"""
function get_truss_1d_K(elem_1d::Element{M,Seg2}, nnodes, time) where M
    K_loc = zeros(nnodes, nnodes)
    for ip in get_integration_points(elem_1d)
        dN = elem_1d(ip, time, Val{:Grad})
        detJ = elem_1d(ip, time, Val{:detJ})
        A = elem_1d("cross section area", ip, time)
        E = elem_1d("youngs modulus", ip, time)
        K_loc += ip.weight * E * A * dN' * dN * detJ
    end
    return K_loc
end

"""
    has_temperature_load(elem_1d::Element{M,Seg2})
    checks if we have a temperature load on the truss element
"""
function has_temperature_load(elem_1d::Element{M,Seg2}) where M
    return haskey(elem_1d, "temperature difference") && haskey(elem_1d, "thermal expansion coefficient")
end

"""
   get_truss_1d_f(element::Element{M,Seg2}, nnodes, time)
   This function assembles the 1d truss force vector
   Will need to change allocation strategy to pre-allocation later
   Can discuss if we really need nnodes, since that is always 2 for trusses
"""
function get_truss_1d_f(elem_1d::Element{M,Seg2}, nnodes, time) where M
    fe = zeros(nnodes)
    if has_temperature_load(elem_1d)
        for ip in get_integration_points(elem_1d)
            dt = elem_1d("temperature difference", ip, time)
            lambda = elem_1d("thermal expansion coefficient", ip, time)
            A = elem_1d("cross section area", ip, time)
            E = elem_1d("youngs modulus", ip, time)
            N = elem_1d(ip, time)
            detJ = elem_1d(ip, time, Val{:detJ})
            f = E * A * lambda * dt
            fe += ip.weight * N' * f # *detJ
        end
        fe[end] *= -1.0
    end
    # how do we generally set the right sides negative
    return fe
end

"""
   get_Kg_and_Tg(element::Element{M,Seg2}, nnodes, ndim, time)
   This function assembles the local stiffness uses global transformation matrix
   to make the global version of the local stiffnes matrix
   Will need to change allocation strategy to pre-allocation later
   Can discuss if we really need nnodes, since that is always 2 for trusses
"""

function get_Kg_and_Tg(element::Element{M,Seg2}, nnodes, ndim, time) where M
    elem_1d, T = get_truss_1d_and_Tg(element, ndim)
    # Update props for stiffness attributes
    update!(elem_1d, "youngs modulus", element("youngs modulus"))
    update!(elem_1d, "cross section area", element("cross section area"))
    K_loc = get_truss_1d_K(elem_1d, nnodes, time)
    ndofs = nnodes * ndim
    K = zeros(ndofs, ndofs)
    K = T' * K_loc * T

    return (K, T)
end

function get_F_local(element::Element{M,Seg2}, nnodes, ndim, time) where M
    elem_1d, T = get_truss_1d_and_Tg(element, ndim)
    # Update props for stiffness attributes
    update!(elem_1d, "youngs modulus", element("youngs modulus"))
    update!(elem_1d, "cross section area", element("cross section area"))
    if (has_temperature_load(element))
        update!(elem_1d, "thermal expansion coefficient", element("thermal expansion coefficient"))
        update!(elem_1d, "temperature difference", element("temperature difference"))
    end
    f = get_truss_1d_f(elem_1d, nnodes, time)
    return f
end

"""
    assemble!(assembly:Assembly, problem::Problem{Elasticity}, elements, time)
Start finite element assembly procedure for Elasticity problem.
Function groups elements to arrays by their type and assembles one element type
at time. This makes it possible to pre-allocate matrices common to same type
of elements.
"""
function assemble!(assembly::Assembly, problem::Problem{Truss},
    element::Element{M,Seg2}, time) where M
    #Require that the number of nodes = 2 ?
    nnodes = length(element)
    ndim = get_unknown_field_dimension(problem)
    elem_1d, T = get_truss_1d_and_Tg(element, ndim)
    # setuo the things needed for local stiffness matrix
    update!(elem_1d, "youngs modulus", element("youngs modulus"))
    update!(elem_1d, "cross section area", element("cross section area"))
    K_loc = get_truss_1d_K(elem_1d, nnodes, time)
    ndofs = nnodes * ndim
    K = zeros(ndofs, ndofs)
    K = T' * K_loc * T
    gdofs = get_gdofs(problem, element)
    add!(assembly.K, gdofs, gdofs, K)
    if has_temperature_load(element)
        update!(elem_1d, "thermal expansion coefficient", element("thermal expansion coefficient"))
        update!(elem_1d, "temperature difference", element("temperature difference"))
        f = get_truss_1d_f(elem_1d, nnodes, time) * -1.0 #Needs to be negated
        fg = T' * f
        add!(assembly.f, gdofs, fg)
    end
    #add!(assembly.f, gdofs, f)
end

assemble_elements!

function assemble_elements!(problem::Problem, assembly::Assembly,
    elements::Vector{Element{M,Poi1}}, time::Float64) where M
    dim = ndofs = get_unknown_field_dimension(problem)
    Ce = zeros(ndofs, ndofs)
    ge = zeros(ndofs)
    fe = zeros(ndofs)
    for element in elements
        fill!(Ce, 0.0)
        fill!(ge, 0.0)
        fill!(fe, 0.0)
        ip = (0.0,)
        for i = 1:dim
            if haskey(element, "fixed displacement $i")
                Ce[i, i] = 1.0
                ge[i] = element("fixed displacement $i", ip, time)
            end
            if haskey(element, "nodal force $i")
                fe[i] = element("nodal force $i", ip, time)
            end
        end
        gdofs = get_gdofs(problem, element)
        add!(assembly.C1, gdofs, gdofs, Ce)
        add!(assembly.C2, gdofs, gdofs, Ce)
        add!(assembly.g, gdofs, ge)
        add!(assembly.f, gdofs, fe)
    end
end

export Truss, get_Kg_and_Tg, get_F_local

