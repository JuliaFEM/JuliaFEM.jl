# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
Solve linear system using LDLt factorization (SuiteSparse). This version
requires that final system is symmetric and positive definite, so boundary
conditions are first eliminated before solution.
"""
function solve_linear_system!(ls::LinearSystem, ::Type{Val{1}})

    K, C1, C2, D, f, g, du, la = ls.K, ls.C1, ls.C2, ls.D, ls.f, ls.g, ls.du, ls.la

    nnz(D) == 0 || return false

    A = get_nonzero_rows(K)
    B = get_nonzero_rows(C2)
    B2 = get_nonzero_columns(C2)
    B == B2 || return false
    I = setdiff(A, B)
    
    debug("# A = $(length(A))")
    debug("# B = $(length(B))")
    debug("# I = $(length(I))")

    if length(B) == 0
        warn("No rows in C2, forget to set Dirichlet boundary conditions to model?")
    else
        @timeit to "solve boundary system using LU factorization" du[B] = lufact(C2[B,B2]) \ full(g[B])
    end

    @timeit to "solve interior domain using LDLt factorization" du[I] = ldltfact(K[I,I]) \ full(f[I] - K[I,B]*du[B])
    @timeit to "solve Lagrange multipliers using LU factorization" la[B] = lufact(C1[B2,B]) \ full(f[B] - K[B,I]*du[I] - K[B,B]*du[B])

    return true
end

"""
Solve linear system using LU factorization (UMFPACK). This version solves
directly the saddle point problem without elimination of boundary conditions.
It is assumed that C1 == C2 and D = 0, so problem is symmetric and zero rows
cand be removed from total system before solution. This kind of system arises
in e.g. mesh tie problem
"""
function solve_linear_system!(ls::LinearSystem, ::Type{Val{2}})

    K, C1, C2, D, f, g, du, la = ls.K, ls.C1, ls.C2, ls.D, ls.f, ls.g, ls.du, ls.la

    C1 == C2 || return false
    length(D) == 0 || return false

    A = [K C1'; C2  D]
    b = [f; g]

    nz1 = get_nonzero_rows(A)
    nz2 = get_nonzero_columns(A)
    nz1 == nz2 || return false
    
    x = lufact(A[nz1,nz2]) \ full(b[nz1])
    x = sparsevec(nz, x)

    ndofs = length(f)
    ls.du = x[1:ndofs]
    ls.la = x[ndofs+1:end]

    return true
end

"""
Solve linear system using LU factorization (UMFPACK). This version solves
directly the saddle point problem without elimination of boundary conditions.
If matrix has zero rows, diagonal term is added to that matrix is invertible.
"""
function solve_linear_system!(la::LinearSystem, ::Type{Val{3}})

    A = [K C1'; C2  D]
    b = [f; g]

    nz = ones(length(b))
    nz[get_nonzero_rows(A)] = 0.0
    A += spdiagm(nz)

    x = lufact(A) \ full(b)

    ndofs = length(f)
    ls.du = x[1:ndofs]
    ls.la = x[ndofs+1:end]

    return true
end
