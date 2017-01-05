# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

function optimize!(assembly::Assembly)
    optimize!(assembly.K)
    optimize!(assembly.Kg)
    optimize!(assembly.f)
    optimize!(assembly.fg)
    optimize!(assembly.C1)
    optimize!(assembly.C2)
    optimize!(assembly.D)
    optimize!(assembly.g)
    optimize!(assembly.c)
end

function append!(assembly::Assembly, sub_assembly::Assembly)
    append!(assembly.M, sub_assembly.M)
    append!(assembly.K, sub_assembly.K)
    append!(assembly.Kg, sub_assembly.Kg)
    append!(assembly.f, sub_assembly.f)
    append!(assembly.fg, sub_assembly.fg)
    append!(assembly.C1, sub_assembly.C1)
    append!(assembly.C2, sub_assembly.C2)
    append!(assembly.D, sub_assembly.D)
    append!(assembly.g, sub_assembly.g)
    append!(assembly.c, sub_assembly.c)
end

""" Calculate norm of assembly, i.e., norm of each block of matrix. """
function norm(assembly::Assembly, p=2)
    N1 = norm(assembly.M, p)
    N2 = norm(assembly.K, p)
    N3 = norm(assembly.Kg, p)
    N4 = norm(assembly.f, p)
    N5 = norm(assembly.fg, p)
    N6 = norm(assembly.C1, p)
    N7 = norm(assembly.C2, p)
    N8 = norm(assembly.D, p)
    N9 = norm(assembly.g, p)
    N10 = norm(assembly.c, p)
    return [N1, N2, N3, N4, N5, N6, N7, N8, N9, N10]
end

function isapprox(a1::Assembly, a2::Assembly)
    T = isapprox(a1.K, a2.K)
    T &= isapprox(a1.C1, a2.C1)
    T &= isapprox(a1.C2, a2.C2)
    T &= isapprox(a1.D, a2.D)
    T &= isapprox(a1.f, a2.f)
    T &= isapprox(a1.g, a2.g)
    return T
end

function assemble_prehook!
end

function assemble_posthook!
end

function assemble!(problem::Problem, time=0.0; auto_initialize=true)
    if !isempty(problem.assembly)
        warn("Assemble problem $(problem.name): problem.assembly is not empty and assembling, are you sure you know what are you doing?")
    end
    if isempty(problem.elements)
        warn("Assemble problem $(problem.name): problem.elements is empty, no elements in problem?")
    else
        first_element = first(problem.elements)
        unknown_field_name = get_unknown_field_name(problem)
        if !haskey(first_element, unknown_field_name)
            warn("Assemble problem $(problem.name): seems that problem is uninitialized.")
            if auto_initialize
                info("Initializing problem $(problem.name) at time $time automatically.")
                initialize!(problem, time)
            end
        end
    end
    if method_exists(assemble_prehook!, Tuple{typeof(problem), Float64})
        assemble_prehook!(problem, time)
    end
    for element in get_elements(problem)
        assemble!(problem.assembly, problem, element, time)
    end
    if method_exists(assemble_posthook!, Tuple{typeof(problem), Float64})
        assemble_posthook!(problem, time)
    end
    return true
end

function assemble!(problem::Problem, time::Real, ::Type{Val{:mass_matrix}}; density=0.0, dual_basis=false, dim=0)
    if !isempty(problem.assembly.M)
        warn("problem.assembly.M is not empty and assembling, are you sure you know what are you doing?")
    end
    if dim == 0
        dim = get_unknown_field_dimension(problem)
    end
    for element in get_elements(problem)
        if !haskey(element, "density") && density == 0.0
            error("Failed to assemble mass matrix, density not defined!")
        end
        nnodes = length(element)
        M = zeros(nnodes, nnodes)
        for ip in get_integration_points(element, 1)
            detJ = element(ip, time, Val{:detJ})
            N = element(ip, time)
            rho = haskey(element, "density") ? element("density", ip, time) : density
            M += ip.weight*rho*N'*N*detJ
        end
        gdofs = get_gdofs(problem, element)
        for j=1:dim
            ldofs = gdofs[j:dim:end]
            add!(problem.assembly.M, ldofs, ldofs, M)
        end
    end
end

