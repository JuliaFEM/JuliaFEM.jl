# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

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
    assemble!(get_assembly(problem), problem, get_elements(problem), time)
    if method_exists(assemble_posthook!, Tuple{typeof(problem), Float64})
        assemble_posthook!(problem, time)
    end
    return true
end

function assemble!(assembly::Assembly, problem::Problem, elements::Vector{Element}, time)
    warn("assemble!() this is default assemble operation, decreased performance can be expected without preallocation of memory!")
    for element in elements
        assemble!(assembly, problem, element, time)
    end
    return nothing
end

function assemble_mass_matrix!(problem::Problem, time)
    if !isempty(problem.assembly.M)
        info("Mass matrix for $(problem.name) is already assembled, skipping assemble routine")
        return
    end
    elements = get_elements(problem)
    for (element_type, elements) in group_by_element_type(get_elements(problem))
        assemble_mass_matrix!(problem::Problem, elements, time)
    end
    return
end

function assemble_mass_matrix!{Basis}(problem::Problem, elements::Vector{Element{Basis}}, time)
    nnodes = length(Basis)
    dim = get_unknown_field_dimension(problem)
    M = zeros(nnodes, nnodes)
    N = zeros(1, nnodes)
    NtN = zeros(nnodes, nnodes)
    ldofs = zeros(Int, nnodes)
    for element in elements
        fill!(M, 0.0)
        for ip in get_integration_points(element, 2)
            detJ = element(ip, time, Val{:detJ})
            rho = element("density", time)
            w = ip.weight*rho*detJ
            eval_basis!(Basis, N, ip)
            N = element(ip, time)
            At_mul_B!(NtN, N, N)
            scale!(NtN, w)
            for i=1:nnodes^2
                M[i] += NtN[i]
            end
        end
        for (i, j) in enumerate(get_connectivity(element))
            @inbounds ldofs[i] = (j-1)*dim
        end
        for i=1:dim
            add!(problem.assembly.M, ldofs+i, ldofs+i, M)
        end
    end
    return
end

