# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

""" Modal solver to solve generalized eigenvalue problems Ku = Muλ

Examples
--------

julia> problems = get_problems()
julia> solver = Solver(Modal)
julia> push!(solver, problems...)
julia> solver()
"""
type Modal <: AbstractSolver
    time :: Float64
    geometric_stiffness :: Bool
    eigvals :: Vector
    eigvecs :: Matrix
    nev :: Int
    which :: Symbol
    bc_invertible :: Bool
    P :: Vector{SparseMatrixCSC}
    symmetric :: Bool
    empty_assemblies_before_solution :: Bool
    dense :: Bool
    info_matrices :: Bool
    sigma :: Float64
end

function Modal(nev=10, which=:SM)
    solver = Modal(0.0, false, [], Matrix{Float64}(0,0), nev, which,
                   false, [], true, true, false, false, 0.0)
end

""" Eliminate Dirichlet boundary condition from matrices K, M. """
function FEMBase.eliminate_boundary_conditions!(K_red::SparseMatrixCSC,
                                                M_red::SparseMatrixCSC,
                                                problem::Problem{Dirichlet}, ndim::Int)
    K = sparse(problem.assembly.K, ndim, ndim)
    C1 = sparse(problem.assembly.C1, ndim, ndim)
    C2 = sparse(problem.assembly.C2, ndim, ndim)
    D = sparse(problem.assembly.D, ndim, ndim)
    f = sparse(problem.assembly.f, ndim, 1)
    g = sparse(problem.assembly.g, ndim, 1)
    Kg = sparse(problem.assembly.Kg, ndim, ndim)
    fg = sparse(problem.assembly.fg, ndim, ndim)
    # only homogenenous boundary condition u=0 is implemented at the moment.
    @assert nnz(K) == 0
    @assert nnz(D) == 0
    @assert nnz(Kg) == 0
    @assert nnz(fg) == 0
    @assert nnz(f) == 0
    @assert nnz(g) == 0
    @assert C1 == C2
    @assert isdiag(C1)
    fixed_dofs = get_nonzero_rows(C1)
    P = ones(ndim)
    P[fixed_dofs] = 0.0
    Q = spdiagm(P)
    K_red[:,:] = Q' * K_red * Q
    M_red[:,:] = Q' * M_red * Q
    return
end

""" Given data vector, return slave displacements. """
function calc_projection(problem::Problem{Mortar}, ndim::Int)

    C1 = sparse(problem.assembly.C1, ndim, ndim)
    C2 = sparse(problem.assembly.C2, ndim, ndim)

    @assert nnz(sparse(problem.assembly.K)) == 0
    @assert nnz(sparse(problem.assembly.D)) == 0
    @assert nnz(sparse(problem.assembly.Kg)) == 0
    @assert nnz(sparse(problem.assembly.fg)) == 0
    @assert nnz(sparse(problem.assembly.f)) == 0
    @assert nnz(sparse(problem.assembly.g)) == 0

    @assert C1 == C2
    #@assert problem.properties.dual_basis == true
    @assert problem.properties.adjust == false

    S = get_nonzero_rows(C2)
    M = setdiff(get_nonzero_columns(C2), S)

    # Construct matrix P = D^-1*M
    D_ = C2[S,S]
    M_ = -C2[S,M]
    P = nothing

    if !isdiag(D_)
        warn("D is not diagonal, is dual basis used? This might take a long time.")
        P = ldltfact(1/2*(D_ + D_')) \ M_
    else
        P = D_ \ M_
    end
    info("Matrix P ready.")

    return S, M, P
end


""" Eliminate mesh tie constraints from matrices K, M. """
function FEMBase.eliminate_boundary_conditions!(K_red::SparseMatrixCSC,
                                                M_red::SparseMatrixCSC,
                                                problem::Union{Problem{Mortar}, Problem{Mortar2D}},
                                                ndim::Int)

    C1 = sparse(problem.assembly.C1, ndim, ndim)
    C2 = sparse(problem.assembly.C2, ndim, ndim)

    @assert nnz(sparse(problem.assembly.K)) == 0
    @assert nnz(sparse(problem.assembly.D)) == 0
    @assert nnz(sparse(problem.assembly.Kg)) == 0
    @assert nnz(sparse(problem.assembly.fg)) == 0
    @assert nnz(sparse(problem.assembly.f)) == 0
    @assert nnz(sparse(problem.assembly.g)) == 0

    @assert C1 == C2
    #@assert problem.properties.dual_basis == true
    @assert problem.properties.adjust == false

    info("Eliminating mesh tie constraint $(problem.name) using static condensation")

    S = get_nonzero_rows(C2)
    M = setdiff(get_nonzero_columns(C2), S)
    info("# slave dofs = $(length(S)), # master dofs = $(length(M))")

    D_ = C2[S,S]
    M_ = -C2[S,M]

    P = nothing
    if !isdiag(D_)
        warn("D is not diagonal, is dual basis used? This might take a long time.")
        P = ldltfact(1/2*(D_ + D_')) \ M_
    else
        P = D_ \ M_
    end
    Id = ones(ndim)
    Q = spdiagm(Id)
    Q[M,S] += P'

    K_red[:,:] = Q*K_red*Q'
    K_red[S,:] = 0.0
    K_red[:,S] = 0.0

    M_red[:,:] = Q*M_red*Q'
    M_red[S,:] = 0.0
    M_red[:,S] = 0.0

    return true
end

function solve!(solver::Solver{Modal}, time::Float64)
    problems = get_problems(solver)
    properties = solver.properties
    info(repeat("-", 80))
    info("Starting natural frequency solver")
    info("Increment time t=$(round(time, 3))")
    info(repeat("-", 80))

    @timeit "assemble matrices" begin
        assemble!(solver, time; with_mass_matrix=true)
        M, K, Kg, f = get_field_assembly(solver)
        if properties.geometric_stiffness
            K += Kg
        end
    end

    dim = size(K, 1)
    ndofs = size(K, 1)

    nboundary_problems = length(get_boundary_problems(solver))

    K_red = K
    M_red = M

    @timeit "eliminate boundary conditions" begin
        if length(properties.P) > 0
            info("Using custom P to make transform K_red = P'*K*P and M_red = P'*M*P")
            for P in properties.P
                K_red = P'*K_red*P
                M_red = P'*M_red*P
            end
        elseif nboundary_problems != 0
            info("Eliminate boundary conditions from system.")
            for boundary_problem in get_boundary_problems(solver)
                eliminate_boundary_conditions!(K_red, M_red, boundary_problem, dim)
            end
        else
            info("No boundary Dirichlet boundary conditions found for system.")
        end
    end

    # free up some memory before solution
    if properties.empty_assemblies_before_solution
        for problem in get_field_problems(solver)
            empty!(problem.assembly)
        end
        gc()
    end

    SparseArrays.droptol!(K_red, 1.0e-9)
    SparseArrays.droptol!(M_red, 1.0e-9)
    nz = get_nonzero_rows(K_red)
    K_red = K_red[nz,nz]
    M_red = M_red[nz,nz]

    sigma = 0.0
    if properties.sigma != 0.0
        info("Adding diagonal term $(properties.sigma) to stiffness matrix")
        sigma = properties.sigma
    end

    props = solver.properties

    info("Calculate $(props.nev) eigenvalues...")

    tic()

    if properties.symmetric
        K_red = 1/2*(K_red + transpose(K_red))
        M_red = 1/2*(M_red + transpose(M_red))
    end

    if properties.info_matrices
        info("is K symmetric? ", issymmetric(K_red))
        info("is M symmetric? ", issymmetric(M_red))
        info("is K positive definite? ", isposdef(K_red))
        info("is M positive definite? ", isposdef(M_red))
    end

    if properties.dense
        K_red = full(K_red)
        M_red = full(M_red)
    end


    om2 = nothing
    X = nothing
    passed = false

    try
        @timeit "solve eigenvalue problem using `eigs`" begin
            om2, X = eigs(K_red + sigma*I, M_red; nev=props.nev, which=props.which)
        end
        passed = true
    catch
        info("failed to calculate eigenvalues for problem. Maybe stiffness matrix is not positive definite, checking...")
        info("is K symmetric? ", issymmetric(K_red))
        info("is M symmetric? ", issymmetric(M_red))
        info("is K positive definite? ", isposdef(K_red))
        info("is M positive definite? ", isposdef(M_red))
        info("Probably the reason is that stiffness matrix is not positive definite and Cholesky factorization is failing.")
        info("To work around this problem, use arguments `sigma = <some small value>` when calling solver, i.e.")
        info("solver(; sigma=1.0e-9")
        info("Be aware that using sigma shifts eigenvalues up and a bit different results can be expected.")
        if size(K_red, 1) < 2000
            info("stiffness matrix is small, using dense eigenvalue solver to check eigenvalues ...")
            om2 = eigvals(full(K_red))
            info("squared eigenvalues om2 = $om2")
        end
        if properties.sigma != 0.0
            info("sigma is manually set and did not work, giving up, try increase sigma.")
            rethrow()
        end
    end

    if !passed
        sigma = 1.0e-9
        info("Calculation of eigenvalues failed, trying again using sigma value sigma=$sigma")
        try
            om2, X = eigs(K_red + sigma*I, M_red; nev=props.nev, which=props.which)
            passed = true
        catch
            info("Failed to calculate eigenvalues with sigma=$sigma, manually set sigma to something larger and try again.")
            rethrow()
        end
    end

    t1 = round(toq(), 2)
    info("Eigenvalues computed in $t1 seconds. Squared eigenvalues: $om2")

    props.eigvals = om2
    neigvals = length(om2)
    props.eigvecs = zeros(ndofs, neigvals)
    for i=1:neigvals
        props.eigvecs[nz,i] = X[:,i]
        for problem in get_boundary_problems(solver)
            isa(problem, Problem{Mortar}) || continue
            S, M, P = calc_projection(problem, dim)
            # FIXME: store projection to boundary problem, i.e.
            # update!(problem, "master-slave projection", time => P)
            # us = P*um
            props.eigvecs[S,i] = P*props.eigvecs[M,i]
        end
    end

    @timeit "save results to Xdmf" update_xdmf!(solver)

    return true

end

function update_xdmf!(solver::Solver{Modal})

    results_writers = get_results_writers(solver)
    if length(results_writers) == 0
        info("Xdmf is not attached to solver, not writing output to a file.")
        info("To write results to Xdmf file, attach Xdmf to Solver, i.e.")
        info("add_results_writer!(solver, Xdmf(\"results\"))")
        return
    end
    if maximum(abs.(imag(solver.properties.eigvals))) > 1.0e-9
        info("Writing imaginary eigenvalues for Xdmf not supported.")
        return
    end

    xdmf = first(results_writers)

    @timeit "fetch geometry" X_ = solver("geometry", solver.properties.time)
    node_ids = keys(X_)
    @timeit "create node permutation" P = Dict(j=>i for (i, j) in enumerate(node_ids))
    nnodes = length(X_)
    ndim = length(X_[first(node_ids)])
    @timeit "create ncoords array" begin
        X = zeros(ndim, nnodes)
        for j in node_ids
            X[:, P[j]] = X_[j]
        end
    end
    geom_type = (ndim == 2 ? "XY" : "XYZ")
    all_elements = get_all_elements(solver)
    element_types = unique(map(get_element_type, all_elements))
    nelements = length(all_elements)

    elcon_arrays  = Dict()
    @timeit "create topology arrays" for element_type in element_types
        elements = collect(filter_by_element_type(element_type, all_elements))
        nelements = length(elements)
        eldim = length(element_type)
        element_conn = zeros(Int, eldim, nelements)
        for (i, element) in enumerate(elements)
            for (j, conn) in enumerate(get_connectivity(element))
                element_conn[j,i] = P[conn]-1
            end
        end
        elcon_arrays[element_type] = element_conn
    end

    xdmf_element_mapping = Dict(
        "Poi1" => "Polyvertex",
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

    # save modes

    temporal_collection = get_temporal_collection(xdmf)
    unknown_field_name = ucfirst(get_unknown_field_name(solver))
    frames = []

    @timeit "save modes" for (j, eigval) in enumerate(real(solver.properties.eigvals))
        if eigval < 0.0
            warn("negative real eigenvalue found, om2=$eigval, setting to zero.")
            eigval = 0.0
        end
        freq = sqrt(eigval)/(2.0*pi)
        path = "/Results/Natural Frequency Analysis/$unknown_field_name/Mode $j"
        info("Creating frequency frame f=$(round(freq, 3)), path=$path")

        frame = new_element("Grid")
        time = new_child(frame, "Time")
        set_attribute(time, "Value", freq)

        geometry = new_element("Geometry")
        set_attribute(geometry, "Type", geom_type)
        data_node_ids = new_dataitem(xdmf, "/Node IDs", collect(node_ids))
        data_geometry = new_dataitem(xdmf, "/Geometry", X)
        add_child(geometry, data_geometry)
        add_child(frame, geometry)

        for element_type in element_types
            timeit("save topology of element type $element_type") do
                elements = collect(filter_by_element_type(element_type, all_elements))
                nelements = length(elements)
                element_ids = map(get_element_id, elements)
                element_conn = elcon_arrays[element_type]
                #@timeit "creaet element_conn" element_conn = map(element -> [nid_mapping[j]-1 for j in get_connectivity(element)], elements)
                #@timeit "hcat elcon" element_conn = hcat(element_conn...)
                element_code = split(string(element_type), ".")[end]
                dataitem = new_dataitem(xdmf, "/Topology/$element_code/Element IDs", element_ids)
                dataitem = new_dataitem(xdmf, "/Topology/$element_code/Connectivity", element_conn)
                topology = new_element("Topology")
                set_attribute(topology, "TopologyType", xdmf_element_mapping[element_code])
                set_attribute(topology, "NumberOfElements", nelements)
                add_child(topology, dataitem)
                add_child(frame, topology)
            end
        end

        @timeit "store eigenmode" begin
            mode_ = solver.properties.eigvecs[:,j]
            mode_ = reshape(mode_, ndim, round(Int, length(mode_)/ndim))
            @timeit "create mode array" begin
                mode = zeros(ndim, nnodes)
                for i in node_ids
                    mode[:,P[i]] = mode_[:,i]
                end
            end
            field_type = ndim == 1 ? "Scalar" : "Vector"
            field_center = "Node"
            attribute = new_child(frame, "Attribute")
            set_attribute(attribute, "Name", unknown_field_name)
            set_attribute(attribute, "Center", field_center)
            set_attribute(attribute, "AttributeType", field_type)
            dataitem = new_dataitem(xdmf, path, mode)
            add_child(attribute, dataitem)
            add_child(frame, attribute)
            add_child(temporal_collection, frame)
        end

    end

    save!(xdmf)
end
