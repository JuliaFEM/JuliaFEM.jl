# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
Examples
--------

julia> problems = get_problems()
julia> solver = Solver(Modal)
julia> push!(solver, problems...)
julia> solver()

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

function call(solver::Solver{Modal}; show_info=true, debug=false)
    show_info && info(repeat("-", 80))
    show_info && info("Starting natural frequency solver")
    show_info && info("Increment time t=$(round(solver.time, 3))")
    show_info && info(repeat("-", 80))
    initialize!(solver)
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
    M, K, Kg, f = get_field_assembly(solver)
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
    K_red = 1/2*(K_red + K_red')
    M_red = 1/2*(M_red + M_red')
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
    om2 = nothing
    X = nothing
    try
        om2, X = eigs(K_red[nz,nz], M_red[nz,nz]; nev=props.nev, which=props.which)
    catch
        info("failed to calculate eigenvalues")
        info("K sym?", issym(K_red[nz,nz]))
        info("M sym?", issym(M_red[nz,nz]))
        info("K posdef?", isposdef(K_red[nz,nz]))
        info("M posdef?", isposdef(M_red[nz,nz]))
        k1 = maximum(abs(K_red[nz,nz] - K_red[nz,nz]'))
        m1 = maximum(abs(M_red[nz,nz] - M_red[nz,nz]'))
        info("K skewness ", k1)
        info("M skewness ", m1)
        rethrow()
    end
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

    for i=1:length(om2)
        u = props.eigvecs[:,i]
        field_dim = get_unknown_field_dimension(solver)
        field_name = get_unknown_field_name(solver)
        nnodes = round(Int, length(u)/field_dim)
        solution = reshape(u, field_dim, nnodes)
        for problem in get_problems(solver)
            local_sol = Dict{Int64, Vector{Float64}}()
            for node_id in get_connectivity(problem)
                local_sol[node_id] = solution[:, node_id]
            end
            freq = real(sqrt(om2[i])/(2.0*pi))
            update!(problem, field_name, freq => local_sol)
        end
    end
    
    return true
end

