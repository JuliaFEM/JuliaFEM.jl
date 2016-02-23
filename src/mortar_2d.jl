# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

# Mortar projection calculation for 2d, in initial configuration X

""" Find projection from slave nodes to master element, i.e. find xi2 from
master element corresponding to the xi1.
"""
function project_from_slave_to_master{S,M}(slave::Element{S}, master::Element{M}, xi1::Vector, time::Real; max_iterations=5, tol=1.0e-9)

    # slave side geometry and normal direction at xi1
    X1 = slave("geometry", xi1, time)
    N1 = slave("normal-tangential coordinates", xi1, time)[:,1]

    # master side geometry at xi2
    master_basis(xi2) = get_basis(M, [xi2])
    master_dbasis(xi2) = get_dbasis(M, [xi2])
    master_geometry = master("geometry")(time)

    function X2(xi2)
        N = master_basis(xi2)
        return sum([N[i]*master_geometry[i] for i=1:length(N)])
    end

    function dX2(xi2)
        dN = master_dbasis(xi2)
        return sum([dN[i]*master_geometry[i] for i=1:length(dN)])
    end

    # equation to solve
    R(xi2) = det([X2(xi2)-X1  N1]')
    dR(xi2) = det([dX2(xi2)  N1]')

    # solve using Newton iterations
    xi2 = 0.0
    for i=1:max_iterations
        dxi2 = -R(xi2) / dR(xi2)
        xi2 += dxi2
        if norm(dxi2) < tol
            return Float64[xi2]
        end
    end

    println("slave element geometry")
    dump(slave("geometry", time).data)
    println("master element geometry")
    dump(master("geometry", time).data)
    error("find projection from slave to master: did not converge")
end

""" Find projection from master surface to slave point, i.e. find xi1 from slave
element corresponding to the xi2. """
function project_from_master_to_slave{S,M}(slave::Element{S}, master::Element{M}, xi2::Vector, time::Real; max_iterations=5, tol=1.0e-9)

    # slave side geometry and normal direction at xi1

    slave_geometry = slave("geometry")(time)
    slave_normals = slave("normal-tangential coordinates")(time)
    slave_basis(xi) = get_basis(S, [xi])
    slave_dbasis(xi) = get_dbasis(S, [xi])

    function X1(xi1)
        N = slave_basis(xi1)
        return sum([N[i]*slave_geometry[i] for i=1:length(N)])
    end

    function dX1(xi1)
        dN = slave_dbasis(xi1)
        return sum([dN[i]*slave_geometry[i] for i=1:length(dN)])
    end

    function N1(xi1)
        N = slave_basis(xi1)
        return sum([N[i]*slave_normals[i] for i=1:length(N)])[:,1]
    end

    function dN1(xi1)
        dN = slave_dbasis(xi1)
        return sum([dN[i]*slave_normals[i] for i=1:length(dN)])[:,1]
    end

    # master side geometry at xi2
    X2 = master("geometry", xi2, time)

    # equation to solve
    R(xi1) = det([X1(xi1)-X2  N1(xi1)]')
    dR(xi1) = det([dX1(xi1) N1(xi1)]') + det([X1(xi1)-X2 dN1(xi1)]')

    # go!
    xi1 = 0.0
    for i=1:max_iterations
        dxi1 = -R(xi1) / dR(xi1)
        xi1 += dxi1
        if norm(dxi1) < tol
            return Float64[xi1]
        end
    end

    println("slave element geometry")
    dump(slave("geometry", time).data)
    println("master element geometry")
    dump(master("geometry", time).data)
    error("find projection from master to slave: did not converge")
end

# for deformed state

""" Find projection from slave nodes to master element, i.e. find xi2 from
master element corresponding to the xi1.
"""
function project_from_slave_to_master{S,M}(slave::Element{S}, master::Element{M}, xi1::Vector, time::Real, ::Type{Val{:deformed}}; max_iterations=5, tol=1.0e-9)

    # slave side geometry and normal direction at xi1
    x1 = slave("geometry", xi1, time)
    if haskey(slave, "displacement")
        x1 += slave("displacement", xi1, time)
    end
    N1 = slave("normal-tangential coordinates", xi1, time)[:,1]

    # master side geometry at xi2
    master_basis(xi2) = get_basis(M, [xi2])
    master_dbasis(xi2) = get_dbasis(M, [xi2])
    master_geometry = master("geometry")(time)
    if haskey(master, "displacement")
        master_geometry += master("displacement")(time)
    end

    function x2(xi2)
        N = master_basis(xi2)
        return sum([N[i]*master_geometry[i] for i=1:length(N)])
    end

    function dx2(xi2)
        dN = master_dbasis(xi2)
        return sum([dN[i]*master_geometry[i] for i=1:length(dN)])
    end

    # equation to solve
    R(xi2) = det([x2(xi2)-x1  N1]')
    dR(xi2) = det([dx2(xi2)  N1]')

    # solve using Newton iterations
    xi2 = 0.0
    for i=1:max_iterations
        dxi2 = -R(xi2) / dR(xi2)
        xi2 += dxi2
        if norm(dxi2) < tol
            return Float64[xi2]
        end
    end

    println("slave element geometry")
    dump(slave("geometry", time).data)
    println("master element geometry")
    dump(master("geometry", time).data)
    error("find projection from slave to master: did not converge")
end

""" Find projection from master surface to slave point, i.e. find xi1 from slave
element corresponding to the xi2. """
function project_from_master_to_slave{S,M}(slave::Element{S}, master::Element{M}, xi2::Vector, time::Real, ::Type{Val{:deformed}}; max_iterations=5, tol=1.0e-9)

    # slave side geometry and normal direction at xi1

    slave_geometry = slave("geometry")(time)
    if haskey(slave, "displacement")
        slave_geometry += slave("displacement")(time)
    end
    slave_normals = slave("normal-tangential coordinates")(time)
    slave_basis(xi) = get_basis(S, [xi])
    slave_dbasis(xi) = get_dbasis(S, [xi])

    function x1(xi1)
        N = slave_basis(xi1)
        return sum([N[i]*slave_geometry[i] for i=1:length(N)])
    end

    function dx1(xi1)
        dN = slave_dbasis(xi1)
        return sum([dN[i]*slave_geometry[i] for i=1:length(dN)])
    end

    function N1(xi1)
        N = slave_basis(xi1)
        return sum([N[i]*slave_normals[i] for i=1:length(N)])[:,1]
    end

    function dN1(xi1)
        dN = slave_dbasis(xi1)
        return sum([dN[i]*slave_normals[i] for i=1:length(dN)])[:,1]
    end

    # master side geometry at xi2
    x2 = master("geometry", xi2, time)
    if haskey(master, "displacement")
        x2 += master("displacement", xi2, time)
    end

    # equation to solve
    R(xi1) = det([x1(xi1)-x2  N1(xi1)]')
    dR(xi1) = det([dx1(xi1) N1(xi1)]') + det([x1(xi1)-x2 dN1(xi1)]')

    # go!
    xi1 = 0.0
    for i=1:max_iterations
        dxi1 = -R(xi1) / dR(xi1)
        xi1 += dxi1
        if norm(dxi1) < tol
            return Float64[xi1]
        end
    end

    println("slave element geometry")
    dump(slave("geometry", time).data)
    println("master element geometry")
    dump(master("geometry", time).data)
    error("find projection from master to slave: did not converge")
end

# Mortar assembly 2d

# quadratic not tested yet
typealias MortarElements2D Union{Seg2}

function assemble!{E<:MortarElements2D}(assembly::Assembly, problem::Problem{Mortar},
                                        slave_element::Element{E}, time::Real)
    # for finite deformation we need to use incremental formulation
    assemble!(assembly, problem, slave_element, time, Val{problem.properties.formulation})
end

""" Assemble 2d mortar contribution. Mortar matrices are assembled at initial
configuration X, so this works for tie contact and small sliding contact. """
function assemble!{E<:MortarElements2D}(assembly::Assembly, problem::Problem{Mortar},
                                        slave_element::Element{E}, time::Real, ::Type{Val{:total}})

    # slave element must have a set of master elements
    haskey(slave_element, "master elements") || return
    props = problem.properties

    # get dimension and name of PARENT field
    field_dim = problem.dimension
    field_name = problem.parent_field_name

    slave_dofs = get_gdofs(slave_element, field_dim)
    nnodes = size(slave_element, 2)

    # slave side quantities: rotation matrix, geometry, displacement, reaction force
    Q = slave_element("normal-tangential coordinates", time)
    Z = zeros(nnodes, nnodes)
    if nnodes == 2
        Q2 = [Q[1] Z; Z Q[2]]
    elseif nnodes == 3
        Q2 = [Q[1] Z Z; Z Q[2] Z; Z Z Q[3]]
    end
    X1 = vec(slave_element("geometry", time))
    u1 = zeros(2*nnodes)
    if haskey(slave_element, "displacement")
        u1 = vec(slave_element("displacement", time))
    end
    x1 = X1 + u1
    la = zeros(2*nnodes)
    if haskey(slave_element, "reaction force")
        la = vec(slave_element("reaction force", time))
    end
    la = Q2'*la

    G = zeros(2*nnodes)
    g = zeros(2*nnodes)
    local_assembly = Assembly()

    for master_element in slave_element["master elements"]

        X2 = vec(master_element("geometry", time))
        u2 = zeros(2*nnodes)
        if haskey(master_element, "displacement")
            u2 = vec(master_element("displacement", time))
        end
        x2 = X2 + u2

        # if distance between elements is "far enough" cannot expect contact
        if props.contact && (props.minimum_distance < Inf)
            slave_midpoint = Float64[mean(x1[1:field_dim:2]), mean(x1[2:field_dim:2])]
            master_midpoint = Float64[mean(x2[1:field_dim:2]), mean(x2[2:field_dim:2])]
            if norm(slave_midpoint - master_midpoint) > props.minimum_distance
                continue
            end
        end

        master_dofs = get_gdofs(master_element, field_dim)
        xi1a = project_from_master_to_slave(slave_element, master_element, [-1.0], time)
        xi1b = project_from_master_to_slave(slave_element, master_element, [ 1.0], time)
        xi1 = clamp([xi1a xi1b], -1.0, 1.0)
        l = 1/2*abs(xi1[2]-xi1[1])
        isapprox(l, 0.0) && continue # no contribution

        # Calculate slave side projection matrix D
        Ae = zeros(nnodes, nnodes)
        De = zeros(nnodes, nnodes)
        Me = zeros(nnodes, nnodes)
        if problem.properties.dual_basis # Construct dual basis
            for ip in get_integration_points(slave_element, Val{5})
                J = get_jacobian(slave_element, ip, time)
                w = ip.weight*norm(J)*l
                xi = 1/2*(1-ip.xi)*xi1[1] + 1/2*(1+ip.xi)*xi1[2]
                N = slave_element(xi, time)
                De += w*diagm(vec(N))
                Me += w*N'*N
            end
            Ae = De*inv(Me)
        else # Standard Lagrange basis
            for ip in get_integration_points(slave_element, Val{5})
                J = get_jacobian(slave_element, ip, time)
                w = ip.weight*norm(J)*l
                xi = 1/2*(1-ip.xi)*xi1[1] + 1/2*(1+ip.xi)*xi1[2]
                N = slave_element(xi, time)
                De += w*N'*N
            end
            Ae = eye(nnodes)
        end

        C1S2 = zeros(2*nnodes, 2*nnodes)
        C1M2 = zeros(2*nnodes, 2*nnodes)

        # Slave side already done; it's De
        for i=1:field_dim
            C1S2[i:field_dim:end,i:field_dim:end] += De
        end

        # Calculate master side projection matrix M
        for ip in get_integration_points(slave_element, Val{5})
            J = get_jacobian(slave_element, ip, time)
            w = ip.weight*norm(J)*l
            # integration point on slave side segment
            xi_slave = 1/2*(1-ip.xi)*xi1[1] + 1/2*(1+ip.xi)*xi1[2]
            # projected integration point to master side element
            xi_master = project_from_slave_to_master(slave_element, master_element, xi_slave, time)
            N1 = slave_element(xi_slave, time)
            N2 = master_element(xi_master, time)
            M = w*kron(Ae*N1', N2)
            for i=1:field_dim
                C1M2[i:field_dim:end,i:field_dim:end] += M
            end
        end

        # Calculate normal-tangential constraints and weighted gap
        C2S2 = Q2'*C1S2
        C2M2 = Q2'*C1M2
        G += -(C2S2*X1 - C2M2*X2)
        g += -(C2S2*x1 - C2M2*x2)

        # Add contributions
        add!(local_assembly.C1, slave_dofs, slave_dofs, C1S2)
        add!(local_assembly.C1, slave_dofs, master_dofs, -C1M2)
        add!(local_assembly.C2, slave_dofs, slave_dofs, C2S2)
        add!(local_assembly.C2, slave_dofs, master_dofs, -C2M2)

    end # all master elements are done

    # if only equality constraints, i.e., mesh tying problem, we're done for this element.
    if !props.contact
        append!(assembly, local_assembly)
        return
    end

    add!(local_assembly.g, slave_dofs, G)

    lan = la[1:field_dim:end]
    lat = la[2:field_dim:end]
    gn = g[1:field_dim:end]
    gt = g[2:field_dim:end]

    # normal condition
    cn = 1.0 # complemementarity parameter
    Cn = lan - max(0, lan - cn*gn)
    inactive_nodes = find(lan - cn*gn .<= 0)
    active_nodes = find(lan - cn*gn .> 0)

    # if all nodes inactive, nothing to contribute.
    if length(active_nodes) == 0
        return
    end

    # manipulate local assembly (remove rows from it based on active set)
    # before adding it to global assembly
    C1 = sparse(local_assembly.C1)
    C2 = sparse(local_assembly.C2)
    D = spzeros(size(C2)...)
    g = sparse(local_assembly.g)

    node_ids = get_connectivity(slave_element)

    # normal constraint: remove inactive nodes
    for j in node_ids[inactive_nodes]
        if length(props.always_in_contact) != 0
            j in props.always_in_contact && continue
        end
        gdofs = [2*(j-1)+1, 2*(j-1)+2]
        # λⱼ = 0 ∀ j ∈ S
        C1[gdofs,:] = 0
        C2[gdofs,:] = 0
        D[gdofs,:] = 0
        g[gdofs,:] = 0
    end

    for (i, j) in enumerate(node_ids[active_nodes])
        gdofs = [2*(j-1)+1, 2*(j-1)+2]
        #D[gdofs[2],gdofs] = C2[gdofs[2],gdofs]
        D[gdofs[2],gdofs] = Q[i][:,2]
        C2[gdofs[2],:] = 0
        g[gdofs[2],:] = 0
    end

    local_assembly.C1 = C1
    local_assembly.C2 = C2
    local_assembly.D = D
    local_assembly.g = g
    append!(assembly, local_assembly)

    if props.store_debug_info
        slave_element["g"] = g
        slave_element["c"] = c
        slave_element["C1"] = C1
        slave_element["C2"] = C2
        slave_element["D"] = D
        slave_element["active nodes"] = active_nodes
    end

end

function assemble!{E<:MortarElements2D}(assembly::Assembly, problem::Problem{Mortar},
                                        slave_element::Element{E}, time::Real, ::Type{Val{:incremental}})

    # slave element must have a set of master elements
    haskey(slave_element, "master elements") || return
    props = problem.properties

    # get dimension and name of PARENT field
    field_dim = problem.dimension
    field_name = problem.parent_field_name

    slave_dofs = get_gdofs(slave_element, field_dim)
    nnodes = size(slave_element, 2)

    # slave side quantities: rotation matrix, geometry, displacement, reaction force
    Q = slave_element("normal-tangential coordinates", time)
    Z = zeros(nnodes, nnodes)
    if nnodes == 2
        Q2 = [Q[1] Z; Z Q[2]]
    elseif nnodes == 3
        Q2 = [Q[1] Z Z; Z Q[2] Z; Z Z Q[3]]
    end
    X1 = vec(slave_element("geometry", time))
    u1 = zeros(2*nnodes)
    if haskey(slave_element, "displacement")
        u1 = vec(slave_element("displacement", time))
    end
    x1 = X1 + u1
    la = zeros(2*nnodes)
    if haskey(slave_element, "reaction force")
        la = vec(slave_element("reaction force", time))
    end
    la = Q2'*la

    G = zeros(2*nnodes)
    g = zeros(2*nnodes)
    local_assembly = Assembly()

    has_contribution = false

    for master_element in slave_element["master elements"]

        X2 = vec(master_element("geometry", time))
        u2 = zeros(2*nnodes)
        if haskey(master_element, "displacement")
            u2 = vec(master_element("displacement", time))
        end
        x2 = X2 + u2

        # if distance between elements is "far enough" cannot expect contact
        if props.contact && (props.minimum_distance < Inf)
            slave_midpoint = Float64[mean(x1[1:field_dim:2]), mean(x1[2:field_dim:2])]
            master_midpoint = Float64[mean(x2[1:field_dim:2]), mean(x2[2:field_dim:2])]
            if norm(slave_midpoint - master_midpoint) > props.minimum_distance
                continue
            end
        end

        master_dofs = get_gdofs(master_element, field_dim)
        xi1a = project_from_master_to_slave(slave_element, master_element, [-1.0], time, Val{:deformed})
        xi1b = project_from_master_to_slave(slave_element, master_element, [ 1.0], time, Val{:deformed})
        xi1 = clamp([xi1a xi1b], -1.0, 1.0)
        l = 1/2*abs(xi1[2]-xi1[1])
        isapprox(l, 0.0) && continue # no contribution

        # Calculate slave side projection matrix D
        Ae = zeros(nnodes, nnodes)
        De = zeros(nnodes, nnodes)
        Me = zeros(nnodes, nnodes)
        if problem.properties.dual_basis # Construct dual basis
            for ip in get_integration_points(slave_element, Val{5})
                J = get_jacobian(slave_element, ip, time, Val{:deformed})
                w = ip.weight*norm(J)*l
                xi = 1/2*(1-ip.xi)*xi1[1] + 1/2*(1+ip.xi)*xi1[2]
                N = slave_element(xi, time)
                De += w*diagm(vec(N))
                Me += w*N'*N
            end
            Ae = De*inv(Me)
        else # Standard Lagrange basis
            for ip in get_integration_points(slave_element, Val{5})
                J = get_jacobian(slave_element, ip, time, Val{:deformed})
                w = ip.weight*norm(J)*l
                xi = 1/2*(1-ip.xi)*xi1[1] + 1/2*(1+ip.xi)*xi1[2]
                N = slave_element(xi, time)
                De += w*N'*N
            end
            Ae = eye(nnodes)
        end

        C1S2 = zeros(2*nnodes, 2*nnodes)
        C1M2 = zeros(2*nnodes, 2*nnodes)

        # Slave side already done; it's De
        for i=1:field_dim
            C1S2[i:field_dim:end,i:field_dim:end] += De
        end

        # Calculate master side projection matrix M
        for ip in get_integration_points(slave_element, Val{5})
            J = get_jacobian(slave_element, ip, time, Val{:deformed})
            w = ip.weight*norm(J)*l
            # integration point on slave side segment
            xi_slave = 1/2*(1-ip.xi)*xi1[1] + 1/2*(1+ip.xi)*xi1[2]
            # projected integration point to master side element
            xi_master = project_from_slave_to_master(slave_element, master_element,
                                                     xi_slave, time, Val{:deformed})
            N1 = slave_element(xi_slave, time)
            N2 = master_element(xi_master, time)
            M = w*kron(Ae*N1', N2)
            for i=1:field_dim
                C1M2[i:field_dim:end,i:field_dim:end] += M
            end
        end

        # Calculate normal-tangential constraints and weighted gap
        C2S2 = Q2'*C1S2
        C2M2 = Q2'*C1M2
        G += props.gap_sign*(C2S2*X1 - C2M2*X2)
        g += props.gap_sign*(C2S2*x1 - C2M2*x2)

        # Add contributions
        add!(local_assembly.C1, slave_dofs, slave_dofs, C1S2)
        add!(local_assembly.C1, slave_dofs, master_dofs, -C1M2)
        add!(local_assembly.C2, slave_dofs, slave_dofs, C2S2)
        add!(local_assembly.C2, slave_dofs, master_dofs, -C2M2)
        has_contribution = true

    end # all master elements are done

    if !has_contribution
        return
    end

    add!(local_assembly.g, slave_dofs, g)

    # if only equality constraints, i.e., mesh tying problem, we're done for this element.
    if !props.contact
        append!(assembly, local_assembly)
        return
    end

    lan = la[1:field_dim:end]
    lat = la[2:field_dim:end]
    gn = g[1:field_dim:end]
    gt = g[2:field_dim:end]

    # normal condition
    cn = 1.0 # complemementarity parameter
    Cn = lan - max(0, lan - cn*gn)
    inactive_nodes = find(lan - cn*gn .<= 0)
    active_nodes = find(lan - cn*gn .> 0)

    # if all nodes inactive, nothing to contribute.
    if length(active_nodes) == 0
        return
    end

    # manipulate local assembly (remove rows from it based on active set)
    # before adding it to global assembly
    C1 = sparse(local_assembly.C1)
    C2 = sparse(local_assembly.C2)
    D = spzeros(size(C2)...)
    g = sparse(local_assembly.g)

    node_ids = get_connectivity(slave_element)

    # normal constraint: remove inactive nodes
    for j in node_ids[inactive_nodes]
        if length(props.always_in_contact) != 0
            j in props.always_in_contact && continue
        end
        gdofs = [2*(j-1)+1, 2*(j-1)+2]
        # λⱼ = 0 ∀ j ∈ S
        C1[gdofs,:] = 0
        C2[gdofs,:] = 0
        D[gdofs,:] = 0
        g[gdofs,:] = 0
    end

    for (i, j) in enumerate(node_ids[active_nodes])
        gdofs = [2*(j-1)+1, 2*(j-1)+2]
        #D[gdofs[2],gdofs] = C2[gdofs[2],gdofs]
        D[gdofs[2],gdofs] = Q[i][:,2]
        C2[gdofs[2],:] = 0
        g[gdofs[2],:] = 0
    end

    local_assembly.C1 = C1
    local_assembly.C2 = C2
    local_assembly.D = D
    local_assembly.g = g
    append!(assembly, local_assembly)

    if props.store_debug_info
        slave_element["g"] = g
        slave_element["c"] = c
        slave_element["C1"] = C1
        slave_element["C2"] = C2
        slave_element["D"] = D
        slave_element["active nodes"] = active_nodes
    end

end

function calculate_gap_vector{E<:MortarElements2D}(
    problem::Problem{Mortar}, slave_element::Element{E},
    time::Real)

    # slave element must have a set of master elements
    haskey(slave_element, "master elements") || return
    props = problem.properties

    # get dimension and name of PARENT field
    field_dim = problem.dimension
    field_name = problem.parent_field_name

    nnodes = size(slave_element, 2)
    gap = zeros(2*nnodes)

    for master_element in slave_element["master elements"]

        xi1a = project_from_master_to_slave(slave_element, master_element, [-1.0], time, Val{:deformed})
        xi1b = project_from_master_to_slave(slave_element, master_element, [ 1.0], time, Val{:deformed})
        xi1 = clamp([xi1a xi1b], -1.0, 1.0)
        l = 1/2*abs(xi1[2]-xi1[1])
        isapprox(l, 0.0) && continue # no contribution

        # Calculate biorthogonal basis
        Ae = zeros(nnodes, nnodes)
        De = zeros(nnodes, nnodes)
        Me = zeros(nnodes, nnodes)
        for ip in get_integration_points(slave_element, Val{5})
            J = get_jacobian(slave_element, ip, time, Val{:deformed})
            w = ip.weight*norm(J)*l
            xi = 1/2*(1-ip.xi)*xi1[1] + 1/2*(1+ip.xi)*xi1[2]
            N = slave_element(xi, time)
            De += w*diagm(vec(N))
            Me += w*N'*N
        end
        Ae = De*inv(Me)

        # Calculate weighted gap
        for ip in get_integration_points(slave_element, Val{5})
            J = get_jacobian(slave_element, ip, time, Val{:deformed})
            w = ip.weight*norm(J)*l
            # integration point on slave side segment
            xi_slave = 1/2*(1-ip.xi)*xi1[1] + 1/2*(1+ip.xi)*xi1[2]
            # projected integration point to master side element
            xi_master = project_from_slave_to_master(slave_element, master_element, xi_slave, time, Val{:deformed})
            X1 = slave_element("geometry", xi_slave, time)
            u1 = zeros(2*nnodes)
            if haskey(slave_element, "displacement")
                u1 = slave_element("displacement", xi_slave, time)
            end
            x1 = X1 + u1
            X2 = master_element("geometry", xi_master, time)
            u2 = zeros(2*nnodes)
            if haskey(master_element, "displacement")
                u2 = master_element("displacement", xi_master, time)
            end
            x2 = X2 + u2
            Q = slave_element("normal-tangential coordinates", xi_slave, time)
            g = -Q'*(x1-x2)
            N1 = slave_element(xi_slave, time)
            Phi = vec(Ae*N1')
            gap[1:field_dim:end] += w*g[1]*Phi
            gap[2:field_dim:end] += w*g[2]*Phi
        end

    end # all master elements are done

    return gap

end
