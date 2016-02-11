# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

# Mortar projection calculation for 2d

""" Find projection from slave nodes to master element, i.e. find xi2 from
master element corresponding to the xi1.
"""
function project_from_slave_to_master{S,M}(slave::Element{S}, master::Element{M}, xi1::Vector, time::Float64=0.0; max_iterations=5, tol=1.0e-9)

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
    dR(xi2) = det([dX2(xi2) N1]')

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
function project_from_master_to_slave{S,M}(slave::Element{S}, master::Element{M}, xi2::Vector, time::Float64=0.0; max_iterations=5, tol=1.0e-9)

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
        #info("dxi1 = $dxi1, xi1 = $xi1, norm(dxi1) = $(norm(dxi1))")
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
    u = zeros(2*nnodes)
    c = zeros(2*nnodes)

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
        xi1a = project_from_master_to_slave(slave_element, master_element, [-1.0])
        xi1b = project_from_master_to_slave(slave_element, master_element, [ 1.0])
        xi1 = clamp([xi1a xi1b], -1.0, 1.0)
        l = 1/2*abs(xi1[2]-xi1[1])
        isapprox(l, 0.0) && continue # no contribution

        # Calculate slave side projection matrix D
        Ae = zeros(nnodes, nnodes)
        De = zeros(nnodes, nnodes)
        Me = zeros(nnodes, nnodes)
        if problem.properties.formulation == :Dual # Construct dual basis
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
            xi_master = project_from_slave_to_master(slave_element, master_element, xi_slave)
            N1 = slave_element(xi_slave, time)
            N2 = master_element(xi_master, time)
            M = w*kron(Ae*N1', N2)
            for i=1:field_dim
                C1M2[i:field_dim:end,i:field_dim:end] += M
            end
        end

        # Calculate normal-tangential constraints
        C2S2 = Q2'*C1S2
        C2M2 = Q2'*C1M2

        # initial weighted gap (capital G for "undeformed")
        G += -(C2S2*X1 - C2M2*X2)
        # change in weighted gap caused by deformation
        u += -(C2S2*u1 - C2M2*u2)

        # Add contributions
        add!(local_assembly.C1, slave_dofs, slave_dofs, C1S2)
        add!(local_assembly.C1, slave_dofs, master_dofs, -C1M2)
        add!(local_assembly.C2, slave_dofs, slave_dofs, C2S2)
        add!(local_assembly.C2, slave_dofs, master_dofs, -C2M2)

    end # all master elements are done

    add!(local_assembly.g, slave_dofs, G)

    # if only equality constraints, i.e., mesh tying problem, we're done for this element.
    if !props.contact
        append!(assembly, local_assembly)
        return
    end

    C1 = sparse(local_assembly.C1)
    C2 = sparse(local_assembly.C2)
    D = spzeros(size(C2)...)
    g = sparse(local_assembly.g)

    # complementarity condition
    lan = la[1:field_dim:end]
    lat = la[2:field_dim:end]
    Gn = G[1:field_dim:end]
    un = u[1:field_dim:end]
    cn = lan - (Gn + un)
    #Cn = lan - max(0, lan - (Gn+un))

    # normal condition
    inactive_nodes = find(cn .<= 0)
    active_nodes = find(cn .> 0)
    #inactive_nodes = find(Cn .>= 0)
    #active_nodes = find(Cn .== 0)

    # inactive element
    if length(active_nodes) == 0
        return
    end

    node_ids = get_connectivity(slave_element)

    # normal constraint: remove inactive nodes
    for j in node_ids[inactive_nodes]
        if length(props.always_in_contact) != 0
            j in props.always_in_contact && continue
        end
        gdofs = [2*(j-1)+1, 2*(j-1)+2]
        C1[gdofs,:] = 0
        C2[gdofs,:] = 0
        D[gdofs,:] = 0
        g[gdofs] = 0
    end

    # frictional contact, see Gitterle2010
    mu = 0.3
    ct = lat + c[2:field_dim:end]

    C = max(mu*cn, abs(ct)).*lat - mu*max(0, cn).*ct
    stick_nodes = find(abs(ct) - mu*cn .< 0)
    slip_nodes = find(abs(ct) - mu*cn .>= 0)
    stick_nodes = setdiff(stick_nodes, inactive_nodes)
    slip_nodes = setdiff(slip_nodes, inactive_nodes)

    for (i, j) in enumerate(node_ids[active_nodes])
        gdofs = [2*(j-1)+1, 2*(j-1)+2]
        #D[gdofs[2],gdofs] = C2[gdofs[2],gdofs]
        D[gdofs[2],gdofs] = Q[i][:,2]'
        C2[gdofs[2],:] = 0
        if props.friction
            g[gdofs[2]] = C[i]
        else
            g[gdofs[2]] = 0.0
        end
    end

    local_assembly.C1 = C1
    local_assembly.C2 = C2
    local_assembly.D = D
    local_assembly.g = g
    append!(assembly, local_assembly)

    if props.store_debug_info
        slave_element["G"] = G
        slave_element["g"] = g
        slave_element["c"] = c
        slave_element["C1"] = C1
        slave_element["C2"] = C2
        slave_element["D"] = D2
        slave_element["active nodes"] = active_nodes
    end

end

