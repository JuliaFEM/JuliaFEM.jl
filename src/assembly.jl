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
    optimize!(assembly.mass_matrix)
    optimize!(assembly.stiffness_matrix)
    optimize!(assembly.force_vector)
end

function append!(assembly::Assembly, sub_assembly::Assembly)
    append!(assembly.mass_matrix, sub_assembly.mass_matrix)
    append!(assembly.stiffness_matrix, sub_assembly.stiffness_matrix)
    append!(assembly.force_vector, sub_assembly.force_vector)
end

function append!(assembly::BoundaryAssembly, sub_assembly::BoundaryAssembly)
    append!(assembly.C1, sub_assembly.C1)
    append!(assembly.C2, sub_assembly.C2)
    append!(assembly.D, sub_assembly.D)
    append!(assembly.g, sub_assembly.g)
end

function assemble!(problem::Union{FieldProblem, BoundaryProblem}, time::Float64; empty_assembly::Bool=true)
    !problem.assembly.changed && return
    empty_assembly && empty!(problem.assembly)
    for element in get_elements(problem)
        assemble!(problem.assembly, problem, element, time)
    end
    problem.assembly.changed = true
end

function assemble(problem::Union{FieldProblem, BoundaryProblem}, time::Real)
    assemble!(problem, time; empty_assembly=true)
    return problem.assembly
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
