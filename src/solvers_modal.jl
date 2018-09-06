# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using SparseArrays, Arpack

mutable struct Modal <: AbstractSolver
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
    solver = Modal(0.0, false, [], Matrix{Float64}(undef,0,0), nev, which,
                   false, [], true, true, false, false, 0.0)
end

"""
    calc_projection(problem, ndim)

A helper function to calculate P = D^-1*M
"""
function calc_projection(problem::T) where
        {T<:Union{Problem{Mortar}, Problem{Mortar2D}}}

    C1 = sparse(problem.assembly.C1)
    C2 = sparse(problem.assembly.C2)
    @assert C1 == C2
    @assert problem.properties.adjust == false

    s = get_nonzero_rows(C2)
    m = setdiff(get_nonzero_columns(C2), s)

    # Construct matrix P = D^-1*M
    D = C2[s,s]
    M = -C2[s,m]

    if !isdiag(D)
        @warn("Mortar matrix D is not diagonal. This might take a long time.")
        P = ldlt(1/2*(D + D')) \ M
    else
        P = D \ M
    end

    return s, m, P
end

function FEMBase.eliminate_boundary_conditions!(problem::P, K, M, f) where {P}
    isempty(problem.assembly.C2) && return nothing
    C1 = sparse(problem.assembly.C1)
    C2 = sparse(problem.assembly.C2)
    C1 == C2 || error("Cannot eliminate boundary condition $P: C1 != C2.")
    isdiag(C1) || error("Cannot eliminate boundary condition $P: C is not diagonal")
    @info("Eliminating boundary condition $(problem.name) from global system.")
    fixed_dofs = get_nonzero_rows(C1)
    K[fixed_dofs,:] .= 0.0
    K[:,fixed_dofs] .= 0.0
    M[fixed_dofs,:] .= 0.0
    M[:,fixed_dofs] .= 0.0
    dropzeros!(K)
    dropzeros!(M)
    return nothing
end

"""
    eliminate_boundary_conditions!(problem, K, M, f)

Eliminate Mortar boundary condition from matrices K, M and force vector f.
"""
function FEMBase.eliminate_boundary_conditions!(problem::T, K, M, f) where
                            {T <: Union{Problem{Mortar}, Problem{Mortar2D}}}
    @info("Eliminating mesh tie constraint $(problem.name) using static condensation")
    s, m, P = calc_projection(problem)
    ndim = size(K, 1)
    Id = ones(ndim)
    Id[s] .= 0.0
    Q = sparse(Diagonal(Id))
    Q[s,m] += P
    K[:,:] .= Q'*K*Q
    M[:,:] .= Q'*M*Q
    return nothing
end

function FEMBase.run!(solver::Solver{Modal})
    time = solver.properties.time
    problems = get_problems(solver)
    properties = solver.properties

    @info("Starting natural frequency solver at time $time")

    @timeit "assemble matrices" begin
        assemble!(solver, time; with_mass_matrix=true)
        M, K, Kg, f = get_field_assembly(solver)
        if properties.geometric_stiffness
            K += Kg
        end
    end

    dim = size(K, 1)
    ndofs = size(K, 1)

    for P in properties.P
        @info("Using P to make transformation K_red = P'*K*P and M_red = P'*M*P")
        K[:,:] .= P'*K*P
        M[:,:] .= P'*M*P
    end

    for problem in get_problems(solver)
        eliminate_boundary_conditions!(problem, K, M, f)
    end

    # free up some memory before solution
    if properties.empty_assemblies_before_solution
        for problem in get_field_problems(solver)
            empty!(problem.assembly)
        end
    end

    SparseArrays.droptol!(K, 1.0e-9)
    SparseArrays.droptol!(M, 1.0e-9)
    nz = get_nonzero_rows(K)
    K = K[nz,nz]
    M = M[nz,nz]

    sigma = 0.0
    if properties.sigma != 0.0
        @info("Adding diagonal term $(properties.sigma) to stiffness matrix")
        sigma = properties.sigma
    end

    props = solver.properties

    @debug("Calculating $(props.nev) eigenvalues...")

    if properties.symmetric
        K = Symmetric(K)
        M = Symmetric(M)
    end

    if properties.info_matrices
        @info("is K symmetric? ", issymmetric(K))
        @info("is M symmetric? ", issymmetric(M))
        @info("is K positive definite? ", isposdef(K))
        @info("is M positive definite? ", isposdef(M))
    end

    if properties.dense
        K = Matrix(K)
        M = Matrix(M)
    end

    om2 = nothing
    X = nothing
    passed = false

    try
        @timeit "solve eigenvalue problem using `eigs`" begin
            om2, X = eigs(K + sigma*I, M; nev=props.nev, which=props.which)
        end
        passed = true
    catch
        @info("Failed to calculate eigenvalues for problem.",
              issymmetric(K), issymmetric(M), isposdef(K), isposdef(M))
        if !isapprox(properties.sigma, 0.0)
            @info("Stiffness matrix is not positive definite and Cholesky " *
                  "factorization is failing. Model is not supported enough " *
                  "with boundary conditions. To work around this problem, " *
                  "use `problem.properties.sigma = <some small value>` " *
                  "To add artificial stiffness to model. (Or add boundary " *
                  "conditions.)")
            rethrow()
        end
    end

    if !passed
        sigma = props.sigma = 1.0e-9
        @info("Calculation of eigenvalues failed. Stiffness matrix is not " *
              "positive definite and Cholesky factorization is failing. Trying " *
              "again by adjusting problem.properties.sigma to $sigma.")
        try
            om2, X = eigs(K + sigma*I, M; nev=props.nev, which=props.which)
            passed = true
        catch
            @info("Failed to calculate eigenvalues with sigma value $sigma. " *
                  "Manually set sigma to something larger and try again.")
            rethrow()
        end
    end

    @info("Squared eigenvalues: $om2.")

    props.eigvals = om2
    neigvals = length(om2)
    props.eigvecs = zeros(ndofs, neigvals)
    for i=1:neigvals
        props.eigvecs[nz,i] = X[:,i]
        for problem in get_boundary_problems(solver)
            isa(problem, Problem{Mortar}) || continue
            s, m, P = calc_projection(problem)
            # FIXME: store projection to boundary problem, i.e.
            # update!(problem, "master-slave projection", time => P)
            # us = P*um
            props.eigvecs[s,i] = P*props.eigvecs[m,i]
        end
    end

    @timeit "save results to Xdmf" update_xdmf!(solver)

    return nothing

end

function update_xdmf!(solver::Solver{Modal})

    results_writers = get_results_writers(solver)
    if length(results_writers) == 0
        @info("Xdmf is not attached to solver, not writing output to a file.")
        @info("To write results to Xdmf file, attach Xdmf to Solver, i.e.")
        @info("add_results_writer!(solver, Xdmf(\"results\"))")
        return
    end
    if maximum(abs.(imag(solver.properties.eigvals))) > 1.0e-9
        @info("Writing imaginary eigenvalues for Xdmf not supported.")
        return
    end

    xdmf = first(results_writers)

    @timeit "fetch geometry" X_ = solver("geometry", solver.properties.time)
    node_ids = keys(X_)
    @timeit "create node permutation" P = Dict(j=>i for (i, j) in enumerate(node_ids))
    nnodes = length(X_)
    ndofs = round(Int, size(solver.properties.eigvecs, 1)/nnodes)
    ndim = length(X_[first(node_ids)])
    @info("Number of nodes: $nnodes. ",
          "Number of dofs/node: $ndofs. ",
          "Dimension of geometry: $ndim.")
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
            @warn("negative real eigenvalue found, om2=$eigval, setting to zero.")
            eigval = 0.0
        end
        freq = sqrt(eigval)/(2.0*pi)
        path = "/Results/Natural Frequency Analysis/$unknown_field_name/Mode $j"
        @info("Creating frequency frame f=$(round(freq; digits=3)), path=$path")

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
            mode_ = reshape(solver.properties.eigvecs[:,j], ndofs, nnodes)
            @timeit "create mode array" begin
                mode = zeros(ndofs, nnodes)
                for nid in node_ids
                    mode[:,P[nid]] = mode_[:,nid]
                end
            end
            if ndofs == 1
                field_type = "Scalar"
            elseif ndofs == 2 # extend to 3d
                field_type = "Vector"
                mode = vcat(mode, zeros(1, nnodes))
            elseif ndofs == 3
                field_type = "Vector"
            elseif ndofs == 6 # has rotation dofs, drop them
                field_type = "Vector"
                mode = mode[1:3, :]
            else
                error("Number of dofs / node = $ndofs, I don't know how to store results to Xdmf!")
            end
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

function solve!(solver::Solver{Modal}, time::Float64)
    @info("solve!(analysis, time) is deprecated. Use run!(analysis) instead.")
    solver.properties.time = time
    run!(solver)
end
