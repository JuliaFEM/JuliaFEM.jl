# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

abstract DirichletProblem <: AbstractProblem

function DirichletProblem(parent_field_name, parent_field_dim, dim=1, elements=[])
    return BoundaryProblem{DirichletProblem}(parent_field_name, parent_field_dim, dim, elements)
end

function assemble!{E}(assembly::Assembly, problem::BoundaryProblem{DirichletProblem}, element::Element{E}, time::Number)

    # get dimension and name of PARENT field
    field_dim = problem.parent_field_dim
    field_name = problem.parent_field_name

    gdofs = get_gdofs(element, field_dim)
    for ip in get_integration_points(element)
        w = ip.weight * det(element, ip, time)
        N = element(ip, time)
        A = w*N'*N

        if haskey(element, field_name)
            # add all dimensions at once if defined element["blaa"] = 0.0
            for i=1:field_dim
                g = element(field_name, ip, time)
                ldofs = gdofs[i:field_dim:end]
                add!(assembly.stiffness_matrix, ldofs, ldofs, A)
                add!(assembly.force_vector, ldofs, w*g*N')
            end
        end
  
        for i=1:field_dim
            # add per dof if defined element["blaa 1"] = 1.0, element["blaa 2"] = 0.0 etc.
            if haskey(element, field_name*" $i")
                g = element(field_name*" $i", ip, time)
                ldofs = gdofs[i:field_dim:end]
                add!(assembly.stiffness_matrix, ldofs, ldofs, A)
                add!(assembly.force_vector, ldofs, w*g*N')
            end
        end
    end
end

