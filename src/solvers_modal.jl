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

function call(solver::Solver{Modal}; show_info=true, debug=false,
              bc_invertible=false, P=nothing,
              empty_assemblies_before_solution=true)
    show_info && info(repeat("-", 80))
    show_info && info("Starting natural frequency solver")
    show_info && info("Increment time t=$(round(solver.time, 3))")
    show_info && info(repeat("-", 80))
    initialize!(solver)
    assemble!(solver; with_mass_matrix=true)
    M, K, Kg, f = get_field_assembly(solver)
    Kb, C1, C2, D, fb, g = get_boundary_assembly(solver)
    K = K + Kb
    f = f + fb
    if solver.properties.geometric_stiffness
        K += Kg
    end

    # free up some memory before solution
    if empty_assemblies_before_solution
        for problem in get_problems(solver)
            empty!(problem.assembly)
        end
        gc()
    end

    @assert nnz(D) == 0
    @assert C1 == C2

    tic()

    nboundary_problems = length(get_boundary_problems(solver))

    if !(P == nothing)
        info("using custom P")
        K_red = P'*K*P
        M_red = P'*M*P
    elseif nboundary_problems != 0
        if bc_invertible
            info("Invertible C, calculating P")
            P, h = create_projection(C1, g, Val{:invertible})
        else
            info("Contacts, calculate P")
            P, h = create_projection(C1, g)
        end
        K_red = P'*K*P
        M_red = P'*M*P
    else
        info("No dirichlet boundaryes, P = I")
        P = speye(size(K, 1))
        K_red = K
        M_red = M
    end

    t1 = round(toq(), 2)
    info("Eliminated dirichlet boundaries in $t1 seconds.")

    # make sure matrices are symmetric
    info("Making matrices symmetric")
    tic()
    K_red = 1/2*(K_red + K_red')
    M_red = 1/2*(M_red + M_red')
    t1 = round(toq(), 2)
    info("Finished in $t1 seconds.")

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
        info("reduced system")
        info("is K symmetric? ", issym(K_red[nz,nz]))
        info("is M symmetric? ", issym(M_red[nz,nz]))
        info("is K positive definite? ", isposdef(K_red[nz,nz]))
        info("is M positive definite? ", isposdef(M_red[nz,nz]))
        k1 = maximum(abs(K_red[nz,nz] - K_red[nz,nz]'))
        m1 = maximum(abs(M_red[nz,nz] - M_red[nz,nz]'))
        info("K 'skewness' (max(abs(K - K'))) = ", k1)
        info("M 'skewness' (max(abs(M - M'))) = ", m1)

        info("original matrix")
        info("is K symmetric? ", issym(K[nz,nz]))
        info("is M symmetric? ", issym(M[nz,nz]))
        info("is K positive definite? ", isposdef(K[nz,nz]))
        info("is M positive definite? ", isposdef(M[nz,nz]))
        k1 = maximum(abs(K[nz,nz] - K[nz,nz]'))
        m1 = maximum(abs(M[nz,nz] - M[nz,nz]'))
        info("K 'skewness' (max(abs(K - K'))) = ", k1)
        info("M 'skewness' (max(abs(M - M'))) = ", m1)

        rethrow()
    end
    t1 = round(toq(), 2)
    info("Eigenvalues computed in $t1 seconds. Eigenvalues: $om2")

    tic()
    props.eigvals = om2
    props.eigvecs = zeros(ndofs, length(om2))
    v = zeros(ndofs)
    for i=1:length(om2)
        fill!(v, 0.0)
        v[nz] = X[:,i]
        props.eigvecs[:,i] = P*v + g
    end
    t1 = round(toq(), 2)

#=    
    for i=1:length(om2)
        freq = real(sqrt(om2[i])/(2.0*pi))
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

    return true
end

function update_xdmf!(solver::Solver{Modal}; show_info=true)
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

