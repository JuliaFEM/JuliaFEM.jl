# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

""" Here formulation is :total or :incremental meaning that we either give
constraint for total quantity u or it's increment Δu. For elasticity we are
using incremental formulation.
"""
type Dirichlet <: BoundaryProblem
    formulation :: Symbol
    dual_basis :: Bool
end

function Dirichlet()
    Dirichlet(:total, true)
end

function get_unknown_field_name(::Type{Dirichlet})
    return "reaction force"
end

function get_formulation_type(problem::Problem{Dirichlet})
    return problem.properties.formulation
end

function assemble!(assembly::Assembly, problem::Problem{Dirichlet}, element::Element, time::Real)

    @assert problem.properties.dual_basis

    # get dimension and name of PARENT field
    field_dim = get_unknown_field_dimension(problem)
    field_name = get_parent_field_name(problem)
    gdofs = get_gdofs(element, field_dim)

    De, Me, Ae = get_dualbasis(element, time)

    # left hand side
    for i=1:field_dim
        ldofs = gdofs[i:field_dim:end]
        if haskey(element, field_name*" $i")
            add!(assembly.C1, ldofs, ldofs, De)
            add!(assembly.C2, ldofs, ldofs, De)
        end
    end

    # right hand side
    for (w, xi) in get_integration_points(element, Val{3})
        J = element(xi, time, Val{:Jacobian})
        JT = transpose(J)
        if size(JT, 2) == 1  # plane problem
            w *= norm(JT)
        else
            w *= norm(cross(JT[:,1], JT[:,2]))
        end
        N = element(xi, time)

        for i=1:field_dim
            ldofs = gdofs[i:field_dim:end]
            if haskey(element, field_name*" $i")
                g = element(field_name*" $i", xi, time)
                if get_formulation_type(problem) == :incremental
                    # if having incremental formulation need to add previous
                    # displacement to rhs (solving increment Δu !
                    haskey(element, "displacement") || continue
                    g_prev = element(field_name, xi, time)
                    g -= g_prev[i]
                end
                add!(assembly.g, ldofs, w*g*Ae*N')
            end
        end

    end

end

#=

function assemble!(assembly::Assembly, problem::Problem{DirichletProblem}, element::Element, time::Real)

    # get dimension and name of PARENT field
    field_dim = problem.parent_field_dim
    field_name = problem.parent_field_name

    gdofs = get_gdofs(element, field_dim)
    for ip in get_integration_points(element, Val{2})
        w = ip.weight
        J = get_jacobian(element, ip, time)
        JT = transpose(J)
        if size(JT, 2) == 1  # plane problem
            w *= norm(JT)
        else
            w *= norm(cross(JT[:,1], JT[:,2]))
        end
        N = element(ip, time)
        A = w*N'*N

        if haskey(element, field_name)
            # add all dimensions at once if defined
            # element["blaa"] = 0.0
            # or
            # element["blaa"] = Vector{Float64}[[0.1, 0.2], [0.3, 0.4]]
            g = element(field_name, ip, time)
            if length(g) != length(N)
                g = g*ones(length(N))
            end
            for i=1:field_dim
                ldofs = gdofs[i:field_dim:end]
                add!(assembly.C1, ldofs, ldofs, A)
                add!(assembly.C2, ldofs, ldofs, A)
            end
            add!(assembly.g, gdofs, w*g*N)
        end

        for i=1:field_dim
            if haskey(element, field_name*" $i")
                g = element(field_name*" $i", ip, time)
                ldofs = gdofs[i:field_dim:end]
                add!(assembly.C1, ldofs, ldofs, A)
                add!(assembly.C2, ldofs, ldofs, A)
                add!(assembly.g, ldofs, w*g*N)
            end
        end
    end
end
=#
