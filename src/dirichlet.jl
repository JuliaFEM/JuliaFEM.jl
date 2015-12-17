# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

abstract DirichletProblem <: AbstractProblem

function DirichletProblem(parent_field_name::ASCIIString, parent_field_dim::Int, dim::Int=1, elements=Element[])
    return BoundaryProblem{DirichletProblem}("dirichlet boundary", parent_field_name, parent_field_dim, dim, elements)
end
function DirichletProblem(problem_name::ASCIIString, parent_field_name::ASCIIString, parent_field_dim::Int, dim::Int=1, elements=Element[])
    return BoundaryProblem{DirichletProblem}(problem_name, parent_field_name, parent_field_dim, dim, elements)
end

function assemble!(assembly::Assembly, problem::BoundaryProblem{DirichletProblem}, element::Element, time::Number)

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

