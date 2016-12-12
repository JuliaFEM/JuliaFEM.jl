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

""" Eliminate Dirichlet boundary condition from matrices K, M. """
function eliminate_boundary_conditions!(K_red::SparseMatrixCSC,
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
    nz = get_nonzero_rows(C1)
    info("bc $(problem.name): $(length(nz)) nonzeros, $nz")
    K_red[nz,:] = 0.0
    K_red[:,nz] = 0.0
    M_red[nz,:] = 0.0
    M_red[:,nz] = 0.0
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

    # determine master and slave dofs
    dim = get_unknown_field_dimension(problem)
    M = Set{Int64}()
    S = Set{Int64}()
    slave_elements = get_slave_elements(problem)
    master_elements = setdiff(get_elements(problem), slave_elements)

    for element in slave_elements
        for j in get_connectivity(element)
            for i=1:dim
                push!(S, dim*(j-1)+i)
            end
        end
    end

    for element in master_elements
        for j in get_connectivity(element)
            for i=1:dim
                push!(M, dim*(j-1)+i)
            end
        end
    end

    S = sort(collect(S))
    M = sort(collect(M))

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
function eliminate_boundary_conditions!(K_red::SparseMatrixCSC,
                                        M_red::SparseMatrixCSC,
                                        problem::Problem{Mortar}, ndim::Int)

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

    # determine master and slave dofs
    dim = get_unknown_field_dimension(problem)
    M = Set{Int64}()
    S = Set{Int64}()
    slave_elements = get_slave_elements(problem)
    master_elements = setdiff(get_elements(problem), slave_elements)

    for element in slave_elements
        for j in get_connectivity(element)
            for i=1:dim
                push!(S, dim*(j-1)+i)
            end
        end
    end

    for element in master_elements
        for j in get_connectivity(element)
            for i=1:dim
                push!(M, dim*(j-1)+i)
            end
        end
    end

    S = sort(collect(S))
    M = sort(collect(M))
    N = setdiff(get_nonzero_rows(K_red), union(S, M))
    info("#S = $(length(S)), #M = $(length(M)), #N = $(length(N))")

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
    #@assert isdiag(D_)
    info("Matrix P ready.")
    Id = ones(ndim)
    #Id[S] = 0
    #Id[M] = 0
    Q = spdiagm(Id)
    Q[M,S] += P'
    info("Matrix Q ready.")

    # testing
    #K_red_orig = copy(K_red)    
    #K_red[N,M] += K_red[N,S]*P
    #K_red[M,N] += P'*K_red[S,N]
    #K_red[M,M] += P'*K_red[S,S]*P
    #K_red[S,:] = 0.0
    #K_red[:,S] = 0.0
    
    info("K transform")
    K_red[:,:] = Q*K_red*Q'
    K_red[S,:] = 0.0
    K_red[:,S] = 0.0

    #K_res = K_red - K_red_2
    #SparseArrays.droptol!(K_res, 1.0e-9)

    info("K transform ready")
    #info("Create matrices, M")
    #info("Sum matricse, M")
    #M_red_orig = copy(M_red)
    #M_red[N,M] += M_red[N,S]*P
    #M_red[M,N] += P'*M_red[S,N]
    #M_red[M,M] += P'*M_red[S,S]*P
    #M_red[S,:] = 0.0
    #M_red[:,S] = 0.0
    info("M transform")
    M_red[:,:] = Q*M_red*Q'
    M_red[S,:] = 0.0
    M_red[:,S] = 0.0
    info("M transform ready")
 
    #M_res = M_red - M_red_2
    #SparseArrays.droptol!(M_res, 1.0e-9)
    #info("Diff")
    #println(K_res)
    #println(M_res)

    return true
end

function (solver::Solver{Modal})(; show_info=true, debug=false,
              bc_invertible=false, P=nothing, symmetric=true,
              empty_assemblies_before_solution=true, dense=false)
    show_info && info(repeat("-", 80))
    show_info && info("Starting natural frequency solver")
    show_info && info("Increment time t=$(round(solver.time, 3))")
    show_info && info(repeat("-", 80))
    initialize!(solver)
    assemble!(solver; with_mass_matrix=true)
    M, K, Kg, f = get_field_assembly(solver)
    if solver.properties.geometric_stiffness
        K += Kg
    end

    dim = size(K, 1)
    tic()

    nboundary_problems = length(get_boundary_problems(solver))

    K_red = K
    M_red = M
    if !(P == nothing)
        info("using custom P")
        K_red = P'*K_red*P
        M_red = P'*M_red*P
    else
        info("Eliminate boundary conditions from system.")
        for boundary_problem in get_boundary_problems(solver)
            eliminate_boundary_conditions!(K_red, M_red, boundary_problem, dim)
        end
    end

    t1 = round(toq(), 2)
    info("Eliminated boundary conditions in $t1 seconds.")

    # free up some memory before solution
    if empty_assemblies_before_solution
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
    ndofs = solver.ndofs
    props = solver.properties

    info("Calculate $(props.nev) eigenvalues...")

    if debug && length(nz) < 100
        info("Stiffness matrix:")
        dump(round(full(K_red)))
        info("Mass matrix:")
        dump(round(full(M_red)))
    end

    tic()
 
    om2 = nothing
    X = nothing

    if symmetric
        K_red = 1/2*(K_red + transpose(K_red))
        M_red = 1/2*(M_red + transpose(M_red))
    end

    info("is K symmetric? ", issymmetric(K_red))
    info("is M symmetric? ", issymmetric(M_red))
    info("is K positive definite? ", isposdef(K_red))
    info("is M positive definite? ", isposdef(M_red))

    if dense
        K_red = full(K_red)
        M_red = full(M_red)
    end

    try
        om2, X = eigs(K_red, M_red; nev=props.nev, which=props.which)
    catch
        info("failed to calculate eigenvalues")
        info("reduced system")
        info("is K symmetric? ", issym(K_red))
        info("is M symmetric? ", issym(M_red))
        info("is K positive definite? ", isposdef(K_red))
        info("is M positive definite? ", isposdef(M_red))
	    dump(full(K_red[1:10,1:10]))
        if size(K_red, 1) < 2000
            om2 = eigvals(full(K_red))
            info("om2 = $om2")
        end
        rethrow()
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

    if !isnull(solver.xdmf)
        update_xdmf!(solver)
    end
    return true

end

function update_xdmf!(solver::Solver{Modal}; show_info=true)

    if maximum(abs(imag(solver.properties.eigvals))) > 1.0e-9
        error("Writing imaginary eigenvalues for Xdmf not supported.")
    end

    xdmf = get(solver.xdmf)

    # geometry
    X_ = solver("geometry", solver.time)
    node_ids = sort(collect(keys(X_)))
    X = hcat([X_[nid] for nid in node_ids]...)
    ndim, nnodes = size(X)
    geom_type = (ndim == 2 ? "XY" : "XYZ")

    # topology
    nid_mapping = Dict(j=>i for (i, j) in enumerate(node_ids)) 
    all_elements = get_all_elements(solver)
    nelements = length(all_elements)
    debug("Saving topology: $nelements elements total.")
    element_types = unique(map(get_element_type, all_elements))

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
    for (j, eigval) in enumerate(real(solver.properties.eigvals))
        if eigval < 0.0
            warn("negative real eigenvalue found, om2=$eigval, setting to zero.")
            eigval = 0.0
        end
        freq = sqrt(eigval)/(2.0*pi)
        path = "/Results/Natural Frequency Analysis/$unknown_field_name/Mode $j"
        info("Creating frequency frame f=$(round(freq, 3)), path=$path")

        frame = new_element("Grid")
        new_child(frame, "Time", Dict("Value" => freq))

        # add geometry
        geometry = new_element("Geometry", Dict("Type" => geom_type))
        data_node_ids = new_dataitem(xdmf, "/Node IDs", node_ids)
        data_geometry = new_dataitem(xdmf, "/Geometry", X)
        add_child(geometry, data_geometry)
        add_child(frame, geometry)

        # add topology
        for element_type in element_types
            elements = filter_by_element_type(element_type, all_elements)
            nelements = length(elements)
            debug("Xdmf save: $nelements elements of type $element_type")
            sort!(elements, by=get_element_id)
            element_ids = map(get_element_id, elements)
            element_conn = map(element -> [nid_mapping[j]-1 for j in get_connectivity(element)], elements)
            element_conn = hcat(element_conn...)
            element_code = split(string(element_type), ".")[end]
            dataitem = new_dataitem(xdmf, "/Topology/$element_code/Element IDs", element_ids)
            dataitem = new_dataitem(xdmf, "/Topology/$element_code/Connectivity", element_conn)
            topology = new_element("Topology")
            set_attribute(topology, "TopologyType", xdmf_element_mapping[element_code])
            set_attribute(topology, "NumberOfElements", length(elements))
            add_child(topology, dataitem)
            add_child(frame, topology)
        end

        mode = zeros(X)
        mode_ = solver.properties.eigvecs[:,j]
        mode_ = reshape(mode_, ndim, round(Int, length(mode_)/ndim))
        for nid in node_ids
            loc = nid_mapping[nid]
            mode[:,loc] = mode_[:,nid]
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
    
    info("Saving Xdmf")
    save!(xdmf)
end
