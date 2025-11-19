# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/FEMBase.jl/blob/master/LICENSE

function isapprox(a1::Assembly, a2::Assembly)
    T = isapprox(a1.K, a2.K)
    T &= isapprox(a1.C1, a2.C1)
    T &= isapprox(a1.C2, a2.C2)
    T &= isapprox(a1.D, a2.D)
    T &= isapprox(a1.f, a2.f)
    T &= isapprox(a1.g, a2.g)
    return T
end

function assemble_prehook!(::Problem, ::T) where T<:Number end

function assemble_posthook!(::Problem, ::T) where T<:Number end

"""
    assemble_elements!(problem, assembly, elements, time)

Assemble elements for problem.

This should be overridden with own `assemble_elements!`-implementation.
"""
function assemble_elements!(problem::Problem, assembly::Assembly,
    elements::Vector{T}, time) where T<:AbstractElement{E} where E
    elements2 = convert(Vector{Element}, elements)
    assemble!(assembly, problem, elements2, time)
end

function assemble!(problem::Problem, time)

    assemble_prehook!(problem, time)
    elements = get_elements(problem)
    assembly = get_assembly(problem)

    if !isempty(assembly)
        @warn("Problem assembly is not empty before assembling. This is probably " *
              "causing unexpected results. To remove old assembly, use " *
              "`empty!(problem.assembly)`", typeof(problem), problem.name)
        assemble_posthook!(problem, time)
        return nothing
    end

    if isempty(elements)
        @warn("There is no elements defined in problem. Before assembling a " *
              "problem, elements must be added using " *
              "`add_elements!(problem, elements)`.", typeof(problem), problem.name)
        assemble_posthook!(problem, time)
        return nothing
    end

    first_element = first(elements)
    unknown_field_name = get_unknown_field_name(problem)
    if !haskey(first_element, unknown_field_name)
        #=
        warn("Assembling elements for problem $(problem.name): seems that ",
             "problem is uninitialized. To initialize problem, use ",
             "`initialize!(problem, time)`.")
        info("Initializing problem $(problem.name) at time $time automatically.")
        =#
        initialize!(problem, time)
    end

    for (element_type, elements) in group_by_element_type(elements)
        assemble_elements!(problem, assembly, elements, time)
    end
    assemble_posthook!(problem, time)
    return nothing
end

function assemble!(problem::Problem)
    @warn("assemble!(problem) will be deprecated. Use assemble!(problem, time)")
    assemble!(problem, 0.0)
end

function assemble_mass_matrix!(problem::Problem, time::Float64)
    if !isempty(problem.assembly.M)
        @info("Mass matrix for is already assembled, not assembling.",
            problem.name)
        return nothing
    end
    elements = get_elements(problem)
    for (element_type, elements) in group_by_element_type(get_elements(problem))
        assemble_mass_matrix!(problem::Problem, elements, time)
    end
    return nothing
end

function assemble_mass_matrix!(problem::Problem, elements::Vector{E}, time) where E<:AbstractElement{M_,B} where {M_,B}
    nnodes = length(first(elements))
    dim = get_unknown_field_dimension(problem)
    M = zeros(nnodes, nnodes)
    N = zeros(1, nnodes)
    NtN = zeros(nnodes, nnodes)
    ldofs = zeros(Int, nnodes)
    for element in elements
        fill!(M, 0.0)
        for ip in get_integration_points(element, 2)
            detJ = element(ip, time, Val{:detJ})
            rho = element("density", ip, time)
            w = ip.weight * rho * detJ
            eval_basis!(B, N, ip)
            N = element(ip, time)
            mul!(NtN, transpose(N), N)
            rmul!(NtN, w)
            for i = 1:nnodes^2
                M[i] += NtN[i]
            end
        end
        for (i, j) in enumerate(get_connectivity(element))
            @inbounds ldofs[i] = (j - 1) * dim
        end
        for i = 1:dim
            add!(problem.assembly.M, ldofs .+ i, ldofs .+ i, M)
        end
    end
    return
end

# TODO (Phase 1B): Re-enable after resolving Tet10 topology vs Tet10Basis name conflict
# This specialized method assumes Tet10 <: AbstractBasis, but Tet10 is now a topology type.
# Need to refactor to use Tet10Basis or accept AbstractElement{M, T, Tet10Basis}
#=
"""
    assemble_mass_matrix!(problem, elements::Vector{Element{Tet10}}, time)

Assemble Tet10 mass matrices using special method. If Tet10 has constant metric
if can be integrated analytically to gain performance.
"""
function assemble_mass_matrix!(problem::Problem, elements::Vector{E}, time) where E<:AbstractElement{M_, Tet10} where M_
    nnodes = length(Tet10)
    dim = get_unknown_field_dimension(problem)
    M = zeros(nnodes, nnodes)
    N = zeros(1, nnodes)
    NtN = zeros(nnodes, nnodes)
    ldofs = zeros(Int, nnodes)

    M_CM = 1.0/2520.0 * [
         6  1  1  1 -4 -6 -4 -4 -6 -6
         1  6  1  1 -4 -4 -6 -6 -4 -6
         1  1  6  1 -6 -4 -4 -6 -6 -4
         1  1  1  6 -6 -6 -6 -4 -4 -4
        -4 -4 -6 -6 32 16 16 16 16  8
        -6 -4 -4 -6 16 32 16  8 16 16
        -4 -6 -4 -6 16 16 32 16  8 16
        -4 -6 -6 -4 16  8 16 32 16 16
        -6 -4 -6 -4 16 16  8 16 32 16
        -6 -6 -4 -4  8 16 16 16 16 32]

    function is_CM(::AbstractElement{M, Tet10}, X; rtol=1.0e-6) where M
        isapprox(X[5],  1/2*(X[1]+X[2]); rtol=rtol) || return false
        isapprox(X[6],  1/2*(X[2]+X[3]); rtol=rtol) || return false
        isapprox(X[7],  1/2*(X[3]+X[1]); rtol=rtol) || return false
        isapprox(X[8],  1/2*(X[1]+X[4]); rtol=rtol) || return false
        isapprox(X[9],  1/2*(X[2]+X[4]); rtol=rtol) || return false
        isapprox(X[10], 1/2*(X[3]+X[4]); rtol=rtol) || return false
        return true
    end


    n_CM = 0
    for element in elements
        for (i, j) in enumerate(get_connectivity(element))
            @inbounds ldofs[i] = (j-1)*dim
        end

        X = element("geometry", time)
        rho = element("density", time)
        if is_CM(element, X) && length(rho) == 1
            ip = (1.0/3.0, 1.0/3.0, 1.0/3.0)
            detJ = element(ip, time, Val{:detJ})
            rho = element("density", ip, time)
            CM_s = detJ*rho
            n_CM += 1
            for i=1:dim
                add!(problem.assembly.M, ldofs .+ i, ldofs .+ i, CM_s * M_CM)
            end
        else
            fill!(M, 0.0)
            for ip in get_integration_points(element, 2)
                detJ = element(ip, time, Val{:detJ})
                rho = element("density", ip, time)
                w = ip.weight*rho*detJ
                eval_basis!(Tet10, N, ip)
                N = element(ip, time)
                mul!(NtN, transpose(N), N)
                rmul!(NtN, w)
                for i=1:nnodes^2
                    M[i] += NtN[i]
                end
            end
            for i=1:dim
                add!(problem.assembly.M, ldofs .+ i, ldofs .+ i, M)
            end
        end
    end
    @info("$n_CM of $(length(elements)) was constant metric.")
    return
end
=#
