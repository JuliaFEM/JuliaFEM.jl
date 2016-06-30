# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

# Functions to handle global assembly of problem

type CAssembly
    interior_dofs :: Vector{Int}
    boundary_dofs :: Vector{Int}
    F :: Union{Factorization, Matrix}
    Kc :: SparseMatrixCSC
    fc :: SparseMatrixCSC
    Kib :: SparseMatrixCSC
    fi :: SparseMatrixCSC
end

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

""" Calculate reduced stiffness matrix.
mindofs: if dofs < mindofs, do not reduce
"""
function reduce(assembly::Assembly, boundary_dofs_::Vector{Int}, mindofs=100000)
    all_dofs = unique(assembly.stiffness_matrix.I)
    boundary_dofs = intersect(all_dofs, boundary_dofs_)
    interior_dofs = setdiff(all_dofs, boundary_dofs_)

    K = sparse(assembly.stiffness_matrix)
    f = sparse(assembly.force_vector)

    dim = size(K, 1)

    # empty assembly to release memory for factorization
    empty!(assembly.stiffness_matrix)
    empty!(assembly.force_vector)
    gc()

    if dim < mindofs
        # no need to do any reduction of matrix size at all, just \ it.
        return CAssembly([], all_dofs, Matrix{Float64}(), K, f, spzeros(0, 0), spzeros(0,1))
    end

    # check that matrix is symmetric
    s = maximum(abs(1/2*(K + K') - K))
    @assert s < 1.0e-6
    K = 1/2*(K + K')

    Kib = K[interior_dofs, boundary_dofs]
    Kbb = K[boundary_dofs, boundary_dofs]
    fi = f[interior_dofs]
    fb = f[boundary_dofs]

    F = cholfact(K[interior_dofs, interior_dofs])
    K = 0
    gc()


    if dim < 100000
        # for small problems we don't need to care about memory usage
        Kd = Kib' * (F \ Kib)
    else
        # for larger problems calculate schur complement in pieces
        nb = length(boundary_dofs)
        p = nb > 10 ? round(Int, nb/10) : nb
        Kd = zeros(nb, nb)
        for bi in 1:nb
            mod(bi, p) == 0 && info("Reduction: ", round(Int, bi/nb*100), " % done")
            C = full(F \ Kib[:, bi])
            for bj in bi:nb
                d = Kib[:, bj]
                @inbounds Kd[bj,bi] = dot(C[rowvals(d)], nonzeros(d))
            end
        end
        Kd += tril(Kd, -1)'
    end
    Kc = spzeros(dim, dim)
    Kc[boundary_dofs, boundary_dofs] = Kbb - Kd

#=  # this is slightly faster but uses more memory
    chunks = round(Int, dim/3000)
    info("Reduction is done in $chunks chunks.")
    nb = length(boundary_dofs)
    kk = round(Int, collect(linspace(0, nb, chunks+1)))
    sl = [kk[j]+1:kk[j+1] for j=1:length(kk)-1]
    Kd = zeros(Float64, nb, nb)
    #Kd = SharedArray(Float64, nb, nb)
    for (k,sli) in enumerate(sl)
        b1 = boundary_dofs[sli]
        Sc = F \ Kib[:,sli]
        for slj in sl
            b2 = boundary_dofs[slj]
            #Kc[b2,b1] = Kbb[slj,sli] - Kib[:,slj]'*Sc
            Kd[slj, sli] = Kib[:,slj]'*Sc
        end
        info("Reduction: ", round(k/chunks*100, 0), " % done")
    end

    Kc = spzeros(dim, dim)
    Kc[boundary_dofs, boundary_dofs] = Kbb - Kd
=#

    fc = spzeros(dim, 1)
    fc[boundary_dofs] = fb - Kib' * (F \ fi)

    return CAssembly(interior_dofs, boundary_dofs, F, Kc, fc, Kib, fi)
end

function reconstruct!(ca::CAssembly, x::SparseMatrixCSC)
    if isa(ca.F, Factorization)
        x[ca.interior_dofs] = ca.F \ (ca.fi - ca.Kib*x[ca.boundary_dofs])
    else  # normal inverse of matrix
        x[ca.interior_dofs] = ca.F * (ca.fi - ca.Kib*x[ca.boundary_dofs])
    end
end

function Base.(:+)(ass1::Assembly, ass2::Assembly)
    mass_matrix = ass1.mass_matrix + ass2.mass_matrix
    stiffness_matrix = ass1.stiffness_matrix + ass2.stiffness_matrix
    force_vector = ass1.force_vector + ass2.force_vector
    return Assembly(mass_matrix, stiffness_matrix, force_vector)
end
