# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

abstract AbstractSolver

type Solver{S<:AbstractSolver}
    name :: AbstractString       # some descriptive name for problem
    time :: Float64              # current time
    problems :: Vector{Problem}
    norms :: Vector{Tuple}       # solution norms for convergence studies
    ndofs :: Int                 # number of degrees of freedom in problem
    xdmf :: Nullable{Xdmf}       # input/output handle
    initialized :: Bool
    u :: Vector{Float64}
    la :: Vector{Float64}
    properties :: S
end


function Solver{S<:AbstractSolver}(::Type{S}, name="solver", properties...)
    variant = S(properties...)
    solver = Solver{S}(name, 0.0, [], [], 0, nothing, false, [], [], variant)
    return solver
end

function Solver{S<:AbstractSolver}(::Type{S}, problems::Problem...)
    solver = Solver(S, "$(S)Solver")
    push!(solver.problems, problems...)
    return solver
end

function get_problems(solver::Solver)
    return solver.problems
end

function push!(solver::Solver, problem)
    push!(solver.problems, problem)
end

function getindex(solver::Solver, problem_name)
    for problem in get_problems(solver)
        if problem.name == problem_name
            return problem
        end
    end
    throw(KeyError(problem_name))
end

# one-liner helpers to identify problem types

is_field_problem(problem) = false
is_field_problem{P<:FieldProblem}(problem::Problem{P}) = true
is_boundary_problem(problem) = false
is_boundary_problem{P<:BoundaryProblem}(problem::Problem{P}) = true
get_field_problems(solver::Solver) = filter(is_field_problem, get_problems(solver))
get_boundary_problems(solver::Solver) = filter(is_boundary_problem, get_problems(solver))

"""
Posthook for field assembly. By default, do nothing.
This can be used to make some modifications for assembly
after all elements are assembled.

Examples
--------
function field_assembly_posthook!(solver::Solver,
                                  K::SparseMatrixCSC,
                                  Kg::SparseMatrixCSC,
                                  f::SparseMatrixCSC,
                                  fg::SpareMatrixCSC)
    info("doing stuff, size(K) = ", size(K))
end
"""
function field_assembly_posthook!
end

"""Return one combined field assembly for a set of field problems.

Parameters
----------
solver :: Solver

Returns
-------
M, K, Kg, f, fg :: SparseMatrixCSC

Notes
-----
If several field problems exists, they are simply summed together, so
problems must have unique node ids.

"""
function get_field_assembly(solver::Solver; show_info=true)
    problems = get_field_problems(solver)

    M = SparseMatrixCOO()
    K = SparseMatrixCOO()
    Kg = SparseMatrixCOO()
    f = SparseMatrixCOO()
    fg = SparseMatrixCOO()

    for problem in problems
        append!(M, problem.assembly.M)
        append!(K, problem.assembly.K)
        append!(Kg, problem.assembly.Kg)
        append!(f, problem.assembly.f)
        append!(fg, problem.assembly.fg)
    end

    if solver.ndofs == 0
        solver.ndofs = size(K, 1)
        show_info && info("automatically determined problem dimension, ndofs = $(solver.ndofs)")
    end

    M = sparse(M, solver.ndofs, solver.ndofs)
    K = sparse(K, solver.ndofs, solver.ndofs)
    if nnz(K) == 0
        warn("Field assembly seems to be empty. Check that elements are pushed to problem and formulation is correct.")
    end
    Kg = sparse(Kg, solver.ndofs, solver.ndofs)
    f = sparse(f, solver.ndofs, 1)
    fg = sparse(fg, solver.ndofs, 1)

    # run any posthook for assembly if defined
    args = Tuple{Solver, SparseMatrixCSC, SparseMatrixCSC, SparseMatrixCSC, SparseMatrixCSC}
    if method_exists(field_assembly_posthook!, args)
        field_assembly_posthook!(solver, K, Kg, fg, fg)
    end

    return M, K, Kg, f, fg
end

""" Posthook for boundary assembly. By default, do nothing. """
function boundary_assembly_posthook!
end

""" Return one combined boundary assembly for a set of boundary problems.

Returns
-------
C1, C2, D, g :: SparseMatrixCSC

Notes
-----
When some dof is constrained by multiple boundary problems an algorithm is
launched what tries to do it's best to solve issue. It's far from perfect
but is able to handle some basic situations occurring in corner nodes and
crosspoints.

"""
function get_boundary_assembly(solver::Solver)
    ndofs = solver.ndofs
    @assert ndofs != 0
    K = spzeros(ndofs, ndofs)
    C1 = spzeros(ndofs, ndofs)
    C2 = spzeros(ndofs, ndofs)
    D = spzeros(ndofs, ndofs)
    f = spzeros(ndofs, 1)
    g = spzeros(ndofs, 1)
    for problem in get_boundary_problems(solver)
        assembly = problem.assembly
        K_ = sparse(assembly.K, ndofs, ndofs)
        C1_ = sparse(assembly.C1, ndofs, ndofs)
        C2_ = sparse(assembly.C2, ndofs, ndofs)
        D_ = sparse(assembly.D, ndofs, ndofs)
        f_ = sparse(assembly.f, ndofs, 1)
        g_ = sparse(assembly.g, ndofs, 1)
        # check for overconstraint situation and handle it if possible
        already_constrained = get_nonzero_rows(C2)
        new_constraints = get_nonzero_rows(C2_)
        overconstrained_dofs = intersect(already_constrained, new_constraints)
        if length(overconstrained_dofs) != 0
            overconstrained_dofs = sort(overconstrained_dofs)
            overconstrained_nodes = find_nodes_by_dofs(problem, overconstrained_dofs)
            handle_overconstraint_error!(problem, overconstrained_nodes,
                overconstrained_dofs, C1, C1_, C2, C2_, D, D_, g, g_)
        end
        K += K_
        C1 += C1_
        C2 += C2_
        D += D_
        f += f_
        g += g_
    end
    return K, C1, C2, D, f, g
end

function resize!(A::SparseMatrixCSC, m::Int64, n::Int64)
    (n == A.n) && (m == A.m) && return
    @assert n >= A.n
    @assert m >= A.m
    append!(A.colptr, A.colptr[end]*ones(Int, m-A.m))
    A.n = n
    A.m = m
end

"""
Given C and g, construct new basis such that v = P*u + g

Parameters
----------
S  set of linearly independent dofs.
"""
function create_projection(C::SparseMatrixCSC, g; S=nothing, tol=1.0e-12)
    n, m = size(C)
    @assert n == m
    if S == nothing
        S = get_nonzero_rows(C)
    end
    # FIXME: this creates dense matrices
    # efficiency / memory usage is a question
    M = get_nonzero_columns(C)
    F = qrfact(C[S,:])
    P = spzeros(n,m)
    P[:,M] = sparse(F \ full(C[S,M]))
    h = sparse(F \ full(g[S]))
    resize!(P, n, m)
    resize!(h, n, 1)
    P = speye(n) - P
    SparseMatrix.droptol!(P, tol)
    return P, h
end

""" Assume C is invertible. """
function create_projection(C, g, ::Type{Val{:invertible}})
    nz1, nz2 = get_nonzeros(C)
    P = spzeros(size(C)...)
    for j=1:size(C,1)
        j in nz1 && continue
        P[j,j] = 1.0
    end
    v = lufact(C[nz1,nz2]) \ full(g[nz1])
    return P, v
end



"""
Solve linear system using LDLt factorization (SuiteSparse). This version
requires that final system is symmetric and positive definite, so boundary
conditions are first eliminated before solution.
"""
function solve!(solver::Solver, K, C1, C2, D, f, g, u, la, ::Type{Val{1}}; debug=false)

    nnz(D) == 0 || return false
    nz = get_nonzero_rows(C2)
    B = get_nonzero_rows(C2')
    # C2^-1 exists or this doesn't work
    length(nz) == length(B) || return false

    A = get_nonzero_rows(K)
    I = setdiff(A, B)

    if debug
        info("# nz = $(length(nz))")
        info("# A = $(length(A))")
        info("# B = $(length(B))")
        info("# I = $(length(I))")
    end

    # solver boundary dofs
    try
        u[B] = lufact(C2[nz,B]) \ full(g[nz])
    catch
        info("solver #1 failed to solve boundary dofs (you should not see this message).")
        info("# nz = $(length(nz))")
        info("# A = $(length(A))")
        info("# B = $(length(B))")
        info("# I = $(length(I))")
        info("nz = $nz")
        info("B = $B")
        rethrow()
    end

    # solve interior domain using LDLt factorization
    F = ldltfact(K[I,I])
    u[I] = F \ (f[I] - K[I,B]*u[B])
    # solve lambda
    la[B] = lufact(C1[B,nz]) \ full(f[B] - K[B,I]*u[I] - K[B,B]*u[B])

    return true
end

"""
Solve linear system using LU factorization (UMFPACK). This version solves
directly the saddle point problem without elimination of boundary conditions.
"""
function solve!(solver::Solver, K, C1, C2, D, f, g, u, la, ::Type{Val{2}})
    # construct global system Ax = b and solve using lufact (UMFPACK)
    A = [K C1'; C2  D]
    b = [f; g]
    x = lufact(A) \ full(b)
    u[:] = x[1:solver.ndofs]
    la[:] = x[solver.ndofs+1:end]
    return true
end

""" Default linear system solver for solver. """
function solve!(solver::Solver; empty_assemblies_before_solution=true, show_info=true)
    show_info && info("Solving problems ...")
    t0 = Base.time()

    # assemble field & boundary problems
    # TODO: return same kind of set for both assembly types
    # M1, K1, Kg1, f1, fg1, C11, C21, D1, g1 = get_field_assembly(solver)
    # M2, K2, Kg2, f2, fg2, C12, C22, D2, g2 = get_boundary_assembly(solver)

    M, K, Kg, f, fg = get_field_assembly(solver)
    Kb, C1, C2, D, fb, g = get_boundary_assembly(solver)
    K = K + Kg + Kb
    f = f + fg + fb
    K = 1/2*(K + K')
    M = 1/2*(M + M')

    nz = ones(solver.ndofs)
    nz[get_nonzero_rows(C2)] = 0.0
    nz[get_nonzero_rows(D)] = 0.0
    D += spdiagm(nz)

    # free up some memory before solution
    for problem in get_problems(solver)
        if empty_assemblies_before_solution
            empty!(problem.assembly)
        else
            optimize!(problem.assembly)
        end
        gc()
    end

    ndofs = solver.ndofs
    u = zeros(ndofs)
    la = zeros(ndofs)
    status = false
    i = 0
    for i in [1, 2]
        status = solve!(solver, K, C1, C2, D, f, g, u, la, Val{i})
        status && break
    end
    status || error("Failed to solve linear system!")
    t1 = round(Base.time()-t0, 2)
    norms = (norm(u), norm(la))
    show_info && info("Solved problems in $t1 seconds using solver $i. Solution norms = $norms.")
    push!(solver.norms, norms)

    solver.u = u
    solver.la = la

    return
end

""" Default assembler for solver. """
function assemble!(solver::Solver; show_info=true, timing=true, with_mass_matrix=false)
    show_info && info("Assembling problems ...")

    function do_assemble(problem)
        t00 = Base.time()
        empty!(problem.assembly)
        assemble!(problem, solver.time)
        if with_mass_matrix && is_field_problem(problem)
            assemble!(problem, solver.time, Val{:mass_matrix})
        end
        t11 = Base.time()
        return t11-t00
    end

    t0 = Base.time()
    assembly_times = map(do_assemble, solver.problems)
    nproblems = length(assembly_times)

    ndofs = 0
    for problem in solver.problems
        Ks = size(problem.assembly.K, 2)
        Cs = size(problem.assembly.C1, 2)
        ndofs = max(ndofs, Ks, Cs)
    end

    solver.ndofs = ndofs
    t1 = round(Base.time()-t0, 2)
    show_info && info("Assembled $nproblems problems in $t1 seconds. ndofs = $ndofs.")
    if timing
        info("Assembly times:")
        for (i, problem) in enumerate(solver.problems)
            pn = problem.name
            pt = round(assembly_times[i], 2)
            info("$i $pn $pt")
        end
    end
end

function get_unknown_fields(solver::Solver)
    fields = Dict()
    for problem in get_field_problems(solver)
        field_name = get_unknown_field_name(problem)
        field_dim = get_unknown_field_dimension(problem)
        fields[field_name] = field_dim
    end
    return fields
end

function get_unknown_field_name(solver::Solver)
    fields = get_unknown_fields(solver)
    return join(sort(collect(keys(fields))), ", ")
end

function get_unknown_field_dimension(solver::Solver)
    fields = get_unknown_fields(solver)
    return sum(values(fields))
end

""" Default initializer for solver. """
function initialize!(solver::Solver; show_info=true)
    if solver.initialized
        show_info && info("initialize!(): solver already initialized")
        return
    end
    show_info && info("Initializing solver ...")
    problems = get_problems(solver)
    length(problems) != 0 || error("Empty solver, add problems to solver using push!")
    t0 = Base.time()
    field_problems = get_field_problems(solver)
    length(field_problems) != 0 || warn("No field problem found from solver, add some..?")
    field_name = get_unknown_field_name(solver)
    field_dim = get_unknown_field_dimension(solver)
    info("initialize!(): looks we are solving $field_name, $field_dim dofs/node")
    nodes = Set{Int64}()
    for problem in problems
        initialize!(problem, solver.time)
        for element in get_elements(problem)
            conn = get_connectivity(element)
            push!(nodes, conn...)
        end
    end
    nnodes = length(nodes)
    info("Total number of nodes in problems: $nnodes")
    maxdof = maximum(nnodes)*field_dim
    info("# of max dof (=size of solution vector) is $maxdof")
    solver.u = zeros(maxdof)
    solver.la = zeros(maxdof)
    # TODO: this could be used to initialize elements too...
    # TODO: cannot initialize to zero always, construct vector from elements.
    for problem in problems
        problem.assembly.u = zeros(maxdof)
        problem.assembly.la = zeros(maxdof)
        # initialize(problem, ....)
    end
    t1 = round(Base.time()-t0, 2)
    show_info && info("Initialized solver in $t1 seconds.")
    solver.initialized = true
end

function get_all_elements(solver::Solver)
    elements = [get_elements(problem) for problem in get_problems(solver)]
    return [elements...;]
end

function get_element_type{E}(element::Element{E})
    return E
end

function get_element_id{E}(element::Element{E})
    return element.id
end

function is_element_type{E}(element::Element{E}, element_type)
    return is(E, element_type)
end

function filter_by_element_type(element_type, elements)
    return filter(element -> is_element_type(element, element_type), elements)
end

function call(solver::Solver, field_name::AbstractString, time::Float64)
    fields = [problem(field_name, time) for problem in get_problems(solver)]
    return merge(fields...)
end

function get_temporal_collection(xdmf::Xdmf)
    domain = find_element(xdmf.xml, "Domain")
    grid = nothing
    if domain == nothing
        info("Xdmf: creating new temporal collection")
        domain = new_child(xdmf.xml, "Domain")
        grid = new_child(domain, "Grid")
        set_attribute(grid, "CollectionType", "Temporal")
        set_attribute(grid, "GridType", "Collection")
    end
    grid = find_element(domain, "Grid")
    return grid
end

""" Default update for solver. """
function update!{S}(solver::Solver{S}; show_info=true)
    u = solver.u
    la = solver.la

    show_info && info("Updating problems ...")
    t0 = Base.time()

    for problem in solver.problems
        assembly = get_assembly(problem)
        elements = get_elements(problem)
        # update solution, first for assembly (u,la) ...
        update!(problem, assembly, u, la)
        # .. and then from assembly (u,la) to elements
        update!(problem, assembly, elements, solver.time)
    end

    # if io is attached to solver, update hdf / xml also
    if !isnull(solver.xdmf)
        update_xdmf!(solver)
    end

    t1 = round(Base.time()-t0, 2)
    show_info && info("Updated problems in $t1 seconds.")
end

function update_xdmf!{S}(solver::Solver{S}; show_info=true)
    xdmf = get(solver.xdmf)
    temporal_collection = get_temporal_collection(xdmf)

    frame = new_element("Grid")
    new_child(frame, "Time", Dict("Value" => solver.time))

    # save geometry
    X = solver("geometry", solver.time)
    node_ids = sort(collect(keys(X)))
    geometry = hcat([X[nid] for nid in node_ids]...)
    ndim, nnodes = size(geometry)
    geom_type = ndim == 2 ? "XY" : "XYZ"
    dataitem = new_dataitem(xdmf, "/Node IDs", node_ids)
    geom = new_child(frame, "Geometry", Dict("Type" => geom_type))
    dataitem = new_dataitem(xdmf, "/Geometry", geometry)
    add_child(geom, dataitem)

    # save topology
    all_elements = get_all_elements(solver)
    nelements = length(all_elements)
    element_types = unique(map(get_element_type, all_elements))

    xdmf_element_mapping = Dict(
        "Seg2" => "Polyline",
        "Tri3" => "Triangle",
        "Quad4" => "Quadrilateral",
        "Tet4" => "Tetrahedron",
        "Pyramid5" => "Pyramid",
        "Wedge6" => "Wedge",
        "Hex8" => "Hexahedron",
        "Seg3" => "Edge_3",
        "Tri6" => "Tri_6",
        "Quad8" => "Quad_8",
        "Tet10" => "Tet_10",
        "Pyramid13" => "Pyramid_13",
        "Wedge15" => "Wedge_15",
        "Hex20" => "Hex_20")

    for element_type in element_types
        elements = filter_by_element_type(element_type, all_elements)
        sort!(elements, by=get_element_id)
        element_ids = map(get_element_id, elements)
        element_conn = map(get_connectivity, elements)
        element_conn = transpose(hcat(element_conn...)) - 1
        element_code = split(string(element_type), ".")[end]
        dataitem = new_dataitem(xdmf, "/Topology/$element_code/Element IDs", element_ids)
        dataitem = new_dataitem(xdmf, "/Topology/$element_code/Connectivity", element_conn)
        topology = new_child(frame, "Topology")
        set_attribute(topology, "TopologyType", xdmf_element_mapping[element_code])
        set_attribute(topology, "NumberOfElements", length(elements))
        add_child(topology, dataitem)
    end

    # save solved fields
    unknown_field_name = get_unknown_field_name(solver)
    U = solver(unknown_field_name, solver.time)
    node_ids2 = sort(collect(keys(U)))

    @assert node_ids == node_ids2
    ndim = length(U[first(node_ids)])
    field_type = ndim == 1 ? "Scalar" : "Vector"
    field_center = "Node"
    if ndim == 2
        for nid in node_ids
            U[nid] = [U[nid]; 0.0]
        end
        ndim = 3
    end
    U = hcat([U[nid] for nid in node_ids]...)
    unknown_field_name = ucfirst(unknown_field_name)
    time = solver.time
    path = ""
    if S == Nonlinear
        iteration = solver.properties.iteration
        path = "/Results/Time $time/Iteration $iteration/Nodal Fields/$unknown_field_name"
    elseif S == Linear
        path = "/Results/Time $time/Nodal Fields/$unknown_field_name"
    end
    dataitem = new_dataitem(xdmf, path, U)
    attribute = new_child(frame, "Attribute")
    set_attribute(attribute, "Name", unknown_field_name)
    set_attribute(attribute, "Center", field_center)
    set_attribute(attribute, "AttributeType", field_type)
    add_child(attribute, dataitem)
    if (S == Linear) || ((S == Nonlinear) && has_converged(solver))
        add_child(temporal_collection, frame)
    end
    save!(xdmf)
end


### Nonlinear quasistatic solver

type Nonlinear <: AbstractSolver
    iteration :: Int        # iteration counter
    min_iterations :: Int64 # minimum number of iterations
    max_iterations :: Int64 # maximum number of iterations
    convergence_tolerance :: Float64
    error_if_no_convergence :: Bool # throw error if no convergence
end

function Nonlinear()
    solver = Nonlinear(0, 1, 10, 5.0e-5, true)
    return solver
end

""" Check convergence of problems.

Notes
-----
Default convergence criteria is obtained by checking each sub-problem convergence.
"""
function has_converged(solver::Solver{Nonlinear}; show_info=false,
                       check_convergence_for_boundary_problems=false)
    properties = solver.properties
    converged = true
    eps = properties.convergence_tolerance
    for problem in solver.problems
        has_converged = true
        if is_field_problem(problem)
            has_converged = problem.assembly.u_norm_change < eps
            if isapprox(norm(problem.assembly.u), 0.0)
                # trivial solution
                has_converged = true
            end
            show_info && info("Details for problem $(problem.name)")
            show_info && info("Norm: $(norm(problem.assembly.u))")
            show_info && info("Norm change: $(problem.assembly.u_norm_change)")
            show_info && info("Has converged? $(has_converged)")
        end
        if is_boundary_problem(problem) && check_convergence_for_boundary_problems
            has_converged = problem.assembly.la_norm_change/norm(problem.assembly.la) < eps
            show_info && info("Details for problem $(problem.name)")
            show_info && info("Norm: $(norm(problem.assembly.la))")
            show_info && info("Norm change: $(problem.assembly.la_norm_change)")
            show_info && info("Has converged? $(has_converged)")
        end
        converged &= has_converged
    end
    return converged
end

type NonlinearConvergenceError <: Exception
    solver :: Solver
end

function Base.showerror(io::IO, exception::NonlinearConvergenceError)
    max_iters = exception.solver.properties.max_iterations
    print(io, "nonlinear iteration did not converge in $max_iters iterations!")
end

""" Default solver for quasistatic nonlinear problems. """
function call(solver::Solver{Nonlinear}; show_info=true)

    properties = solver.properties

    # 1. initialize each problem so that we can start nonlinear iterations
    initialize!(solver)

    # 2. start non-linear iterations
    for properties.iteration=1:properties.max_iterations
        show_info && info(repeat("-", 80))
        show_info && info("Starting nonlinear iteration #$(properties.iteration)")
        show_info && info("Increment time t=$(round(solver.time, 3))")
        show_info && info(repeat("-", 80))

        # 2.1 update linearized assemblies
        assemble!(solver)
        # 2.2 call solver for linearized system
        solve!(solver)
        # 2.3 update solution back to elements
        update!(solver)

        # 2.4 check convergence
        if has_converged(solver)
            info("Converged in $(properties.iteration) iterations.")
            properties.iteration >= properties.min_iterations && return true
            info("Convergence criteria met, but iteration < min_iterations, continuing...")
        end
    end

    # 3. did not converge
    properties.error_if_no_convergence && throw(NonlinearConvergenceError(solver))
end

""" Convenience function to call nonlinear solver. """
function NonlinearSolver(problems...)
    solver = Solver(Nonlinear, "default nonlinear solver")
    if length(problems) != 0
        push!(solver, problems...)
    end
    return solver
end
function NonlinearSolver(name::AbstractString, problems::Problem...)
    solver = NonlinearSolver(problems...)
    solver.name = name
    return solver
end


### Linear quasistatic solver

""" Quasistatic solver for linear problems.

Notes
-----
Main differences in this solver, compared to nonlinear solver are:
1. system of problems is assumed to converge in one step
2. reassembly of problem is done only if it's manually requested using empty!(problem.assembly)

"""
type Linear <: AbstractSolver
end

function assemble!(solver::Solver{Linear}; show_info=true)
    show_info && info("Assembling problems ...")
    tic()
    nproblems = 0
    ndofs = 0
    for problem in get_problems(solver)
        if isempty(problem.assembly)
            assemble!(problem, solver.time)
            nproblems += 1
        else
            show_info && info("$(problem.name) already assembled, skipping.")
        end
        ndofs = max(ndofs, size(problem.assembly.K, 2))
    end
    solver.ndofs = ndofs
    t1 = round(toq(), 2)
    show_info && info("Assembled $nproblems problems in $t1 seconds. ndofs = $ndofs.")
end

function call(solver::Solver{Linear}; show_info=true)
    t0 = Base.time()
    show_info && info(repeat("-", 80))
    show_info && info("Starting linear solver")
    show_info && info("Increment time t=$(round(solver.time, 3))")
    show_info && info(repeat("-", 80))
    initialize!(solver)
    assemble!(solver)
    solve!(solver)
    update!(solver)
    t1 = round(Base.time()-t0, 2)
    show_info && info("Linear solver ready in $t1 seconds.")
end

""" Convenience function to call linear solver. """
function LinearSolver(problems::Problem...)
    solver = Solver(Linear, "default linear solver")
    if length(problems) != 0
        push!(solver, problems...)
    end
    return solver
end
function LinearSolver(name::AbstractString, problems::Problem...)
    solver = LinearSolver(problems...)
    solver.name = name
    return solver
end

### End of linear quasistatic solver

### Postprocessor

type Postprocessor <: AbstractSolver
    assembly :: Assembly
    F :: Union{Factorization, Void}
end

function Postprocessor()
    Postprocessor(Assembly(), nothing)
end

function assemble!(solver::Solver{Postprocessor}; show_info=true)
    show_info && info("Assembling problems ...")
    tic()
    nproblems = 0
    ndofs = 0
    assembly = solver.properties.assembly
    empty!(assembly)
    for problem in get_problems(solver)
        for element in get_elements(problem)
            postprocess!(assembly, problem, element, solver.time)
        end
        nproblems += 1
        ndofs = max(ndofs, size(problem.assembly.K, 2))
    end
    solver.ndofs = ndofs
    t1 = round(toq(), 2)
    show_info && info("Assembled $nproblems problems in $t1 seconds. ndofs = $ndofs.")
end

function call(solver::Solver{Postprocessor}; show_info=true)
    t0 = Base.time()
    show_info && info(repeat("-", 80))
    show_info && info("Starting postprocessor")
    show_info && info("Increment time t=$(round(solver.time, 3))")
    show_info && info(repeat("-", 80))
    initialize!(solver)
    assemble!(solver)
    assembly = solver.properties.assembly
    M = sparse(assembly.M)
    f = sparse(assembly.f)
    F = cholfact(M)
    q = F \ f
    t1 = round(Base.time()-t0, 2)
    show_info && info("Postprocess of results ready in $t1 seconds.")
    return q
end

""" Convenience function to call postprocessor. """
function Postprocessor(problems::Problem...)
    solver = Solver(Postprocessor, "default postprocessor")
    if length(problems) != 0
        push!(solver, problems...)
    end
    return solver
end

function Postprocessor(name::AbstractString, problems::Problem...)
    solver = Postprocessor(problems...)
    solver.name = name
    return solver
end
