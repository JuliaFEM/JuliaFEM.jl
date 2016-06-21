# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
Examples
--------

julia> problems = get_problems()
julia> solver = Solver(Modal)
julia> push!(solver, problems...)
julia> call(solver)

"""
type Modal <: AbstractSolver
    geometric_stiffness :: Bool
    eigvals :: Vector
    eigvecs :: Matrix
    nev :: Int
    which :: Symbol
end

function Modal(nev=10, which=:SM)
    solver = Modal(false, Vector(), Matrix(), nev, which)
end

function call(solver::Solver{Modal}; debug=false)
    # assemble all field problems
    info("Assembling problems ...")
    tic()
    for problem in get_field_problems(solver)
        assemble!(problem, solver.time)
        assemble!(problem, solver.time, Val{:mass_matrix})
    end
    for problem in get_boundary_problems(solver)
        assemble!(problem, solver.time)
    end
    t1 = round(toq(), 2)
    info("Assembled in $t1 seconds.")
    M, K, Kg, f = get_field_assembly(solver; with_mass_matrix=true)
    Kb, C1, C2, D, fb, g = get_boundary_assembly(solver)
    K = K + Kb
    f = f + fb
    if solver.properties.geometric_stiffness
        K += Kg
    end
    
    @assert nnz(D) == 0
    @assert C1 == C2

    tic()
    P, h = create_projection(C1, g)
    K_red = P'*K*P
    M_red = P'*M*P
    t1 = round(toq(), 2)
    info("Eliminated dirichlet boundaries in $t1 seconds.")

    nz = get_nonzero_rows(K_red)
    ndofs = solver.ndofs
    props = solver.properties
    info("Calculate $(props.nev) eigenvalues...")
    if debug && length(nz) < 100
        info("Stiffness matrix:")
        dump(round(full(K[nz, nz])))
        info("Mass matrix:")
        dump(round(full(M[nz, nz])))
    end

    tic()
    om2, X = eigs(K_red[nz,nz], M_red[nz,nz]; nev=props.nev, which=props.which)
    props.eigvals = om2
    props.eigvecs = zeros(ndofs, length(om2))
    v = zeros(ndofs)
    for i=1:length(om2)
        fill!(v, 0.0)
        v[nz] = X[:,i]
        props.eigvecs[:,i] = P*v + g
    end
    t1 = round(toq(), 2)
    info("Eigenvalues computed in $t1 seconds. Eigenvalues: $om2")
    return true
end

