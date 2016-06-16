# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

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
    t1 = round(toq(), 2)
    info("Assembled in $t1 seconds.")
    M, K, Kg, f = get_field_assembly(solver; with_mass_matrix=true)
    if solver.properties.geometric_stiffness
        K += Kg
    end
    for problem in get_boundary_problems(solver)
        assemble!(problem, solver.time)
        # FIXME: Check for tie contacts and rhs. Here we just
        # remove all fixed dofs giving funny results if problem
        # is having MPCs or non-homogeneous Dirichlet conditions
#       eliminate!(M, K, Kg, f, problem)
        fixed_dofs = get_nonzero_rows(problem.assembly.C2)
        K[fixed_dofs, :] = 0
        K[:, fixed_dofs] = 0
        M[fixed_dofs, :] = 0
        M[:, fixed_dofs] = 0
        f[fixed_dofs, :] = 0
    end
    fd = get_nonzero_rows(K)
    ndofs = solver.ndofs
    props = solver.properties
    info("Calculate $(props.nev) eigenvalues...")
    if debug
        info("Stiffness matrix:")
        dump(round(full(K[fd, fd])))
        info("Mass matrix:")
        dump(round(full(M[fd, fd])))
    end
    tic()
    om2, X = eigs(K[fd, fd], M[fd, fd]; nev=props.nev, which=props.which)
    props.eigvals = om2
    props.eigvecs = zeros(ndofs, length(om2))
    props.eigvecs[fd, :] = X
    t1 = round(toq(), 2)
    info("Eigenvalues computed in $t1 seconds. Eigenvalues: $om2")
    return true
end

