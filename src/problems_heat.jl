# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

""" Heat equations.

Field equation is:

    ρc∂u/∂t = ∇⋅(k∇u) + f

Weak form is: find u∈U such that ∀v in V

    ∫k∇u∇v dx = ∫fv dx + ∫gv ds,

where

    k = temperature thermal conductivity    defined on volume elements
    f = temperature load                    defined on volume elements
    g = temperature flux                    defined on boundary elements

Parameters
----------
temperature thermal conductivity
temperature load
temperature flux
thermal conductivity
heat source
heat flux
heat transfer coefficient
external temperature


Formulations
------------
1D, 2D, 3D

References
----------
https://en.wikipedia.org/wiki/Heat_equation
https://en.wikipedia.org/wiki/Heat_capacity
https://en.wikipedia.org/wiki/Heat_flux
https://en.wikipedia.org/wiki/Thermal_conduction
https://en.wikipedia.org/wiki/Thermal_conductivity
https://en.wikipedia.org/wiki/Thermal_diffusivity
https://en.wikipedia.org/wiki/Volumetric_heat_capacity
"""
type Heat <: FieldProblem
    formulation :: ASCIIString
    store_fields :: Vector{ASCIIString}
end

function Heat()
    return Heat("3D", [])
end

function get_unknown_field_name(problem::Problem{Heat})
    return "temperature"
end

function assemble!(assembly::Assembly, problem::Problem{Heat}, element::Element, time)
    formulation = Val{Symbol(problem.properties.formulation)}
    assemble!(assembly, problem, element, time, formulation)
end

# 3d heat problems

function assemble!{E}(assembly::Assembly, problem::Problem{Heat}, element::Element{E}, time, ::Type{Val{Symbol("3D")}})
    info("Unknown element type $E for 3d heat problem!")
end

typealias Heat3DVolumeElements Union{Tet4, Tet10, Hex8, Hex20, Hex27}
typealias Heat3DSurfaceElements Union{Tri3, Tri6, Quad4, Quad8, Quad9}

function assemble!{E<:Heat3DVolumeElements}(assembly::Assembly, problem::Problem{Heat}, element::Element{E}, time, ::Type{Val{Symbol("3D")}})
    gdofs = get_gdofs(problem, element)
    field_name = get_unknown_field_name(problem)
    nnodes = length(element)
    K = zeros(nnodes, nnodes)
    fq = zeros(nnodes)
    for ip in get_integration_points(element)
        detJ = element(ip, time, Val{:detJ})
        w = ip.weight*detJ
        N = element(ip, time)
        if haskey(element, "$field_name thermal conductivity")
            dN = element(ip, time, Val{:Grad})
            k = element("$field_name thermal conductivity", ip, time)
            K += w*k*dN'*dN
        end
        if haskey(element, "$field_name load")
            f = element("$field_name load", ip, time)
            fq += w*N'*f
        end
    end
    T = vec(element[field_name](time))
    fq -= K*T
    add!(assembly.K, gdofs, gdofs, K)
    add!(assembly.f, gdofs, fq)
end

function postprocess!{E}(assembly::Assembly, problem::Problem{Heat}, element::Element{E}, time)
    haskey(element, "temperature thermal conductivity") || return
    gdofs = get_gdofs(problem, element)
    field_name = get_unknown_field_name(problem)
    nnodes = length(element)
    Me = zeros(nnodes, nnodes)
    De = zeros(nnodes, nnodes)
    f = zeros(nnodes, 3)
    for ip in get_integration_points(element)
        detJ = element(ip, time, Val{:detJ})
        w = ip.weight*detJ
        N = element(ip, time)
        De += w*diagm(vec(N))
        Me += w*N'*N
    end
    Ae = De*inv(Me)
    for ip in get_integration_points(element)
        detJ = element(ip, time, Val{:detJ})
        w = ip.weight*detJ
        N = vec(element(ip, time))
        Phi = transpose(Ae*N)
        k = element("temperature thermal conductivity", ip, time)
        gradT = element(field_name, ip, time, Val{:Grad})
        q = -vec(k*gradT)
        update!(ip, "heat flux", time => q)
        f += w*Phi'*q'
    end
    add!(assembly.M, gdofs, gdofs, De)
    add!(assembly.f, gdofs, [1, 2, 3], f)
end

function assemble!{E<:Heat3DSurfaceElements}(assembly::Assembly, problem::Problem{Heat}, element::Element{E}, time, ::Type{Val{Symbol("3D")}})
    gdofs = get_gdofs(problem, element)
    field_name = get_unknown_field_name(problem)
    nnodes = length(element)
    K = zeros(nnodes, nnodes)
    fq = zeros(nnodes)
    for ip in get_integration_points(element, 1)
        detJ = element(ip, time, Val{:detJ})
        w = ip.weight*detJ
        N = element(ip, time)
        if haskey(element, "$field_name flux")
            q = element("$field_name flux", ip, time)
            fq += w*N'*q
        end
        if haskey(element, "$field_name heat transfer coefficient")
            h = element("$field_name heat transfer coefficient", ip, time)
            Tu = element("$field_name external temperature", ip, time)
            K += w*h*N'*N
            fq += w*N'*h*Tu
        end
    end
    T = vec(element[field_name](time))
    fq -= K*T
    add!(assembly.K, gdofs, gdofs, K)
    add!(assembly.f, gdofs, fq)
end

# 2d heat problems

function assemble!{E}(assembly::Assembly, problem::Problem{Heat}, element::Element{E}, time, ::Type{Val{Symbol("2D")}})
    info("Unknown element type $E for 2d heat problem!")
end

typealias Heat2DVolumeElements Union{Tri3, Tri6, Quad4}
typealias Heat2DSurfaceElements Union{Seg2, Seg3}

function assemble!{E<:Heat2DVolumeElements}(assembly::Assembly, problem::Problem{Heat}, element::Element{E}, time, ::Type{Val{Symbol("2D")}})
    gdofs = get_gdofs(problem, element)
    field_name = get_unknown_field_name(problem)
    nnodes = length(element)
    K = zeros(nnodes, nnodes)
    fq = zeros(nnodes)
    for ip in get_integration_points(element)
        detJ = element(ip, time, Val{:detJ})
        w = ip.weight*detJ
        N = element(ip, time)
        if haskey(element, "$field_name thermal conductivity")
            dN = element(ip, time, Val{:Grad})
            k = element("$field_name thermal conductivity", ip, time)
            K += w*k*dN'*dN
        end
        if haskey(element, "$field_name load")
            f = element("$field_name load", ip, time)
            fq += w*N'*f
        end
    end
    T = vec(element[field_name](time))
    fq -= K*T
    add!(assembly.K, gdofs, gdofs, K)
    add!(assembly.f, gdofs, fq)
end

function assemble!{E<:Heat2DSurfaceElements}(assembly::Assembly, problem::Problem{Heat}, element::Element{E}, time, ::Type{Val{Symbol("2D")}})
    gdofs = get_gdofs(problem, element)
    field_name = get_unknown_field_name(problem)
    nnodes = length(element)
    K = zeros(nnodes, nnodes)
    fq = zeros(nnodes)
    for ip in get_integration_points(element)
        detJ = element(ip, time, Val{:detJ})
        w = ip.weight*detJ
        N = element(ip, time)
        if haskey(element, "$field_name flux")
            g = element("$field_name flux", ip, time)
            fq += w*N'*g
        end
    end
    T = vec(element[field_name](time))
    fq -= K*T
    add!(assembly.K, gdofs, gdofs, K)
    add!(assembly.f, gdofs, fq)
end

