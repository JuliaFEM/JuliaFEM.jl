# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/HeatTransfer.jl/blob/master/LICENSE
#
# Heat transfer problem types - consolidated from HeatTransfer.jl

"""
    Heat

3D heat transfer analysis for JuliaFEM.

# Fields used in formulation

- `thermal conductivity`
- `heat source`
- `heat flux`
- `external temperature`
- `heat transfer coefficient`

# References

- https://en.wikipedia.org/wiki/Heat_equation
- https://en.wikipedia.org/wiki/Heat_capacity
- https://en.wikipedia.org/wiki/Heat_flux
- https://en.wikipedia.org/wiki/Thermal_conduction
- https://en.wikipedia.org/wiki/Thermal_conductivity
- https://en.wikipedia.org/wiki/Thermal_diffusivity
- https://en.wikipedia.org/wiki/Volumetric_heat_capacity
"""

struct PlaneHeat <: FieldProblem end
struct Heat <: FieldProblem end

get_unknown_field_name(::PlaneHeat) = "temperature"
get_unknown_field_name(::Heat) = "temperature"

function assemble_elements!(problem::Problem{P}, assembly::Assembly,
    elements::Vector{Element{M,B}}, time::Float64) where
{M,B,P<:Union{PlaneHeat,Heat}}

    bi = BasisInfo(B)
    ndofs = length(bi)
    Ke = zeros(ndofs, ndofs)
    fe = zeros(ndofs)

    for element in elements
        fill!(Ke, 0.0)
        fill!(fe, 0.0)
        for ip in get_integration_points(element)
            J, detJ, N, dN = element_info!(bi, element, ip, time)
            s = ip.weight * detJ
            k = element("thermal conductivity", ip, time)
            Ke += s * k * dN' * dN
            if haskey(element, "heat source")
                f = element("heat source", ip, time)
                fe += s * N' * f
            end
        end
        if haskey(element, "temperature")
            T = [element("temperature", time)...]
            fe -= Ke * T
        end
        gdofs = get_gdofs(problem, element)
        add!(assembly.K, gdofs, gdofs, Ke)
        add!(assembly.f, gdofs, fe)
    end

end

function assemble_elements!(problem::Problem{PlaneHeat}, assembly::Assembly,
    elements::Vector{Element{M,B}}, time::Float64) where
{M,B<:Union{Seg2,Seg3}}
    return assemble_boundary_elements!(problem, assembly, elements, time)
end

function assemble_elements!(problem::Problem{Heat}, assembly::Assembly,
    elements::Vector{Element{M,B}}, time::Float64) where
{M,B<:Union{Tri3,Quad4,Tri6,Quad8,Quad9}}
    return assemble_boundary_elements!(problem, assembly, elements, time)
end

function assemble_boundary_elements!(problem::Problem, assembly::Assembly,
    elements::Vector{Element{M,B}}, time::Float64) where {M,B}

    bi = BasisInfo(B)
    ndofs = length(bi)
    Ke = zeros(ndofs, ndofs)
    fe = zeros(ndofs)

    for element in elements
        fill!(fe, 0.0)
        fill!(Ke, 0.0)
        for ip in get_integration_points(element, 2)
            J, detJ, N, dN = element_info!(bi, element, ip, time)
            s = ip.weight * detJ
            if haskey(element, "heat flux")
                g = element("heat flux", ip, time)
                fe += s * N' * g
            end
            if haskey(element, "heat transfer coefficient") && haskey(element, "external temperature")
                h = element("heat transfer coefficient", ip, time)
                Tu = element("external temperature", ip, time)
                Ke += s * h * N' * N
                fe += s * N' * h * Tu
            end
        end
        if haskey(element, "temperature")
            T = [element("temperature", time)...]
            fe -= Ke * T
        end
        gdofs = get_gdofs(problem, element)
        add!(assembly.K, gdofs, gdofs, Ke)
        add!(assembly.f, gdofs, fe)
    end

end

export Heat, PlaneHeat
