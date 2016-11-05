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
function eliminate_boundary_conditions!(K_red, M_red, problem::Problem{Dirichlet}, ndim::Int)
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
    @assert problem.properties.dual_basis == true
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
    if !isdiag(D_)
        info("D is not diagonal, is dual basis used?")
        println(D_)
    end
    @assert isdiag(D_)
    P = D_ \ M_
    info("Projection P ready.")

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
    @assert problem.properties.dual_basis == true
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
    if !isdiag(D_)
        info("D is not diagonal, is dual basis used?")
        println(D_)
    end
    @assert isdiag(D_)
    P = D_ \ M_
    info("Projection P ready.")

    K_red[N,M] += K_red[N,S]*P
    K_red[M,N] += P'*K_red[S,N]
    K_red[M,M] += P'*K_red[S,S]*P
    K_red[S,:] = 0.0
    K_red[:,S] = 0.0
    
    M_red[N,M] += M_red[N,S]*P
    M_red[M,N] += P'*M_red[S,N]
    M_red[M,M] += P'*M_red[S,S]*P
    M_red[S,:] = 0.0
    M_red[:,S] = 0.0

    return true
end

function call(solver::Solver{Modal}; show_info=true, debug=false,
              bc_invertible=false, P=nothing, symmetric=true,
              empty_assemblies_before_solution=true, dense=false,
	      real_eigenvalues=true, positive_eigenvalues=true)
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

    SparseArrays.droptol!(K_red, 1.0e-12)
    SparseArrays.droptol!(M_red, 1.0e-12)
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

    info("is K symmetric? ", issym(K_red))
    info("is M symmetric? ", issym(M_red))
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
        #k1 = maximum(abs(K_red[nz,nz] - K_red[nz,nz]'))
        #m1 = maximum(abs(M_red[nz,nz] - M_red[nz,nz]'))
        #info("K 'skewness' (max(abs(K - K'))) = ", k1)
        #info("M 'skewness' (max(abs(M - M'))) = ", m1)
        #info("original matrix")
        #info("is K symmetric? ", issym(K[nz,nz]))
        #info("is M symmetric? ", issym(M[nz,nz]))
        #info("is K positive definite? ", isposdef(K[nz,nz]))
        #info("is M positive definite? ", isposdef(M[nz,nz]))
        #k1 = maximum(abs(K[nz,nz] - K[nz,nz]'))
        #m1 = maximum(abs(M[nz,nz] - M[nz,nz]'))
        #info("K 'skewness' (max(abs(K - K'))) = ", k1)
        #info("M 'skewness' (max(abs(M - M'))) = ", m1)
        rethrow()
    end
    
    t1 = round(toq(), 2)
    info("Eigenvalues computed in $t1 seconds. Squared eigenvalues: $om2")

    if real_eigenvalues
        om2 = real(om2)
    end

    if positive_eigenvalues
        om2[om2 .< 0.0] = 0.0
    end
 
    om = sqrt(om2)

    props.eigvals = om
    props.eigvecs = zeros(ndofs, length(om))
    for i=1:length(om)
        props.eigvecs[nz,i] = X[:,i]
        for problem in get_boundary_problems(solver)
            isa(problem, Problem{Mortar}) || continue
            S, M, P = calc_projection(problem, dim)
	    # us = P*um
            props.eigvecs[S,i] = P*props.eigvecs[M,i]
        end
    end


#=
    for i=1:length(om)
        freq = real(om[i]/(2.0*pi))
        u = props.eigvecs[:,i]
        field_dim = get_unknown_field_dimension(solver)
        field_name = get_unknown_field_name(solver)
        if field_dim != 1
            nnodes = round(Int, length(u)/field_dim)
            u = reshape(u, field_dim, nnodes)
            u = Vector{Float64}[u[:,i] for i in 1:nnodes]
        end
        for problem in get_problems(solver)
            for element in get_elements(problem)
                connectivity = get_connectivity(element)
                update!(element, field_name, freq => u[connectivity])
            end
        end
    end
=#

    update_xdmf!(solver)
    return true

end

function update_xdmf!(solver::Solver{Modal}; show_info=true)
    xdmf = get(solver.xdmf)
    temporal_collection = get_temporal_collection(xdmf)

    # 1. save geometry
    X_ = solver("geometry", solver.time)
    node_ids = sort(collect(keys(X_)))
    X = hcat([X_[nid] for nid in node_ids]...)
    ndim, nnodes = size(X)
    geom_type = (ndim == 2 ? "XY" : "XYZ")
    data_node_ids = new_dataitem(xdmf, "/Node IDs", node_ids)
    data_geometry = new_dataitem(xdmf, "/Geometry", X)
    geometry = new_element("Geometry", Dict("Type" => geom_type))
    add_child(geometry, data_geometry)

    # 2. save topology

    nid_mapping = Dict([j => i for (i, j) in enumerate(node_ids)])
    
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

    topology = []
    for element_type in element_types
        elements = filter_by_element_type(element_type, all_elements)
        sort!(elements, by=get_element_id)
        #elements = elements[1:5]
        element_ids = map(get_element_id, elements)
        #element_conn = map(get_connectivity, elements)
        #trans = element -> [nid_mapping[j] for j in get_connectivity(element)]
        #element_conn = map(trans, element_conn)
        #info("first element, connectivity = $(get_connectivity(first(elements)))")
        #info("first element, coordinates = $([X_[j] for j in get_connectivity(first(elements))])")
        element_conn = map(element -> [nid_mapping[j]-1 for j in get_connectivity(element)], elements)
        #info("conn2 = $element_conn")
        #G1 = vec(first(elements)("geometry", solver.time))
        #G1 = reshape(G1, 3, 4)
        #G2 = X[:, first(element_conn)+1]
        #info("first element, coordinates from geometry field = $G1")
        #info("first element, coordinates from array = $G2")
        element_conn = hcat(element_conn...)
        #info(element_conn)
        element_code = split(string(element_type), ".")[end]
        dataitem = new_dataitem(xdmf, "/Topology/$element_code/Element IDs", element_ids)
        dataitem = new_dataitem(xdmf, "/Topology/$element_code/Connectivity", element_conn)
        topology_ = new_element("Topology")
        set_attribute(topology_, "TopologyType", xdmf_element_mapping[element_code])
        set_attribute(topology_, "NumberOfElements", length(elements))
        add_child(topology_, dataitem)
        push!(topology, topology_)
        break
    end

    # save modes

    freqs = real(solver.properties.eigvals/(2.0*pi))
    for (j, freq) in enumerate(freqs)
        info("Saving frequency $(round(freq, 3))")
        frame = new_element("Grid")
        new_child(frame, "Time", Dict("Value" => freq))
        add_child(frame, geometry)
        for topo in topology
            add_child(frame, topo)
        end
        add_child(temporal_collection, frame)

        mode = zeros(X)
	mode_ = solver.properties.eigvecs[:,j]
        mode_ = reshape(mode_, ndim, round(Int, length(mode_)/ndim))
        for nid in node_ids
            loc = nid_mapping[nid]
            mode[:,loc] = mode_[:,nid]
        end

        field_type = ndim == 1 ? "Scalar" : "Vector"
        field_center = "Node"
        unknown_field_name = get_unknown_field_name(solver)
        unknown_field_name = ucfirst(unknown_field_name)
	freqn = freqs[j]
        path = "/Results/Frequency $freqn/Nodal Fields/$unknown_field_name"
        dataitem = new_dataitem(xdmf, path, mode)
        attribute = new_child(frame, "Attribute")
        set_attribute(attribute, "Name", unknown_field_name)
        set_attribute(attribute, "Center", field_center)
        set_attribute(attribute, "AttributeType", field_type)
        add_child(attribute, dataitem)
        add_child(frame, attribute)

        continue

        # save solved fields
        unknown_field_name = get_unknown_field_name(solver)
        U = solver(unknown_field_name, freq)
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
        path = "/Results/Frequency $freq/Nodal Fields/$unknown_field_name"
        dataitem = new_dataitem(xdmf, path, U)
        attribute = new_child(frame, "Attribute")
        set_attribute(attribute, "Name", unknown_field_name)
        set_attribute(attribute, "Center", field_center)
        set_attribute(attribute, "AttributeType", field_type)
        add_child(attribute, dataitem)
        
    end

    info("Saving Xdmf")
    save!(xdmf)
end
