# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

# PETSc interface for solver

using PETSc

import JuliaFEM.Core: solve

"""
Parameters
----------
preconditioner : "jacobi"
ksp_type: "bcgs", "gmres"?
"""
function solve(K, f, C, g, ::Type{Val{:PETSc_GMRES}}; preconditioner=nothing)
    t0 = time()
    dim = size(K, 1)

    # make sure C is square
    boundary_dofs = unique(rowvals(C))
    boundary_dofs2 = unique(rowvals(C'))
    @assert length(boundary_dofs) == length(boundary_dofs2)
    @assert setdiff(Set(boundary_dofs), Set(boundary_dofs2)) == Set()
    all_dofs = unique(rowvals(K))
    interior_dofs = setdiff(all_dofs, boundary_dofs)
    info("PETSc: all dofs = $(length(all_dofs))")
    info("PETSc: interior dofs = $(length(interior_dofs))")
    info("PETSc: boundary dofs = $(length(boundary_dofs))")
    # solve displacement on known boundary
    LUF = lufact(C[boundary_dofs, boundary_dofs])
    u = zeros(dim)
    u[boundary_dofs] = LUF \ full(g[boundary_dofs])
    info("PETSc: displacement on boundary solved.")
    normub = norm(u[boundary_dofs])
    if isapprox(normub, 0.0)
        info("PETSc: homogeneous dirichlet boundary")
    end

    # interior domain and lagrange multipliers

    t = time()
    # this is completely unnecessary step and will be removed in future.
    # -->
    info("PETSc: creating matrices in PETSc format.")
    ninterior_dofs = length(interior_dofs)

    # nz, see https://github.com/JuliaParallel/PETSc.jl/issues/52
    d = Dict{Int64, Int64}()
    for i in rowvals(K)
        haskey(d, i) ? (d[i] += 1) : (d[i] = 1)
    end
    nz = maximum(values(d))

    A = PETSc.Mat(Float64, ninterior_dofs, ninterior_dofs; nz=nz)
    info("PETSc: $ninterior_dofs interior dofs, assembling to PETSc Mat")
    for (i, j, v) in zip(findnz(K[interior_dofs, interior_dofs])...)
        A[i, j] = v
    end

    fi = f[interior_dofs]

    b = PETSc.Vec(Float64, ninterior_dofs, PETSc.C.VECMPI)
    for (i, j, v) in zip(findnz(sparse(f[interior_dofs]))...)
        b[i] = v
    end

    info("PETSc: initialization of matrices in ", time()-t, " seconds")
    # <--

    kspg = PETSc.KSP(A, ksp_monitor="")

    # apply preconditioner if defined
    if !isa(preconditioner, Void)
        info("PETSc: preconditioner: $preconditioner")
        pc = PETSc.PC(Float64, comm=PETSc.comm(kspg), pc_type=preconditioner)
        PETSc.chk(PETSc.C.PCSetOperators(pc.p, A.p, A.p))
        kspg = PETSc.KSP(pc, ksp_monitor="")
    end

    info("PETSc: performing ksp GMRES solve")
    x = kspg \ b
    info("PETSc: finished ksp solve")
    info("PETSc: ksp info:\n",petscview(kspg))
    for (i, d) in enumerate(interior_dofs)
        u[d] = x[i]
    end

    la = zeros(dim)
    Kib = K[interior_dofs, boundary_dofs]
    Kbb = K[boundary_dofs, boundary_dofs]
    la[boundary_dofs] = LUF \ full(Kib'*u[interior_dofs] - Kbb*u[boundary_dofs])

    info("PETSc: solved in ", time()-t0, " seconds. norm = ", norm(u))
    return u, la
end

info("PETSc interface loaded.")
