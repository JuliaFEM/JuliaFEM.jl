# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

""" Find segment from slave element corresponding to master element nodes.
x1_, n1_
slave element geometry and normal direction

x2_ master element nodes to project onto slave
"""
function project_from_master_to_slave{E<:MortarElements2D}(
    slave_element::Element{E}, x1_::DVTI, n1_::DVTI, x2::Vector;
    tol=1.0e-10, max_iterations=20)

    function x1(xi1)
        N = get_basis(E, xi1)
        return vec(N)*x1_
    end

    function dx1(xi1)
        dN = get_dbasis(E, xi1)
        return vec(dN)*x1_
    end

    function n1(xi1)
        N = get_basis(E, xi1)
        return vec(N)*n1_
    end

    function dn1(xi1)
        dN = get_dbasis(E, xi1)
        return vec(dN)*n1_
    end

    cross2(a, b) = cross([a; 0], [b; 0])[3]
    R(xi1) = cross2(x1(xi1)-x2, n1(xi1))
    dR(xi1) = cross2(dx1(xi1), n1(xi1)) + cross2(x1(xi1)-x2, dn1(xi1))
    xi1 = 0.0
    dxi1 = 0.0
    for i=1:max_iterations
        dxi1 = -R(xi1)/dR(xi1)
        xi1 += dxi1
        if norm(dxi1) < tol
            return xi1
        end
    end

    info("x1 = $(ForwardDiff.get_value(x1_.data))")
    info("n1 = $(ForwardDiff.get_value(n1_.data))")
    info("x2 = $(ForwardDiff.get_value(x2))")
    info("xi1 = $(ForwardDiff.get_value(xi1)), dxi1 = $(ForwardDiff.get_value(dxi1))")
    info("-R(xi1) = $(ForwardDiff.get_value(-R(xi1)))")
    info("dR(xi1) = $(ForwardDiff.get_value(dR(xi1)))")
    error("find projection from master to slave: did not converge")

end

function project_from_slave_to_master{E<:MortarElements2D}(
    master_element::Element{E}, x1::Vector, n1::Vector, x2_::DVTI;
    tol=1.0e-10, max_iterations=20)

    function x2(xi2)
        N = get_basis(E, xi2)
        return vec(N)*x2_
    end

    function dx2(xi2)
        dN = get_dbasis(E, xi2)
        return vec(dN)*x2_
    end

    cross2(a, b) = cross([a; 0], [b; 0])[3]
    R(xi2) = cross2(x2(xi2)-x1, n1)
    dR(xi2) = cross2(dx2(xi2), n1)
    xi2 = 0.0
    dxi2 = 0.0
    for i=1:max_iterations
        dxi2 = -R(xi2) / dR(xi2)
        xi2 += dxi2
        if norm(dxi2) < tol
            return xi2
        end
    end

    error("find projection from slave to master: did not converge, last val: $xi2 and $dxi2")
    
end


function assemble!{E<:MortarElements2D}(assembly::Assembly, problem::Problem{Mortar}, slave_element::Element{E}, time::Real,
    ::Type{Val{:forwarddiff_old}})
    haskey(slave_element, "master elements") || return
    props = problem.properties
    field_dim = get_unknown_field_dimension(problem)
    field_name = get_parent_field_name(problem)
    slave_dofs = get_gdofs(slave_element, field_dim)
    nnodes = size(slave_element, 2)
    X1 = slave_element("geometry", time)
    #u1 = slave_element("displacement", time)
    #x1 = X1 + u1
    slave_element_nodes = get_connectivity(slave_element)
    adjacent_elements = find_elements(get_elements(problem), slave_element_nodes)
    adjacent_nodes = get_nodes(adjacent_elements) # including also nodes from adjacent elements
    Q = [0.0 -1.0; 1.0 0.0]

    X = spzeros(10000, 1)
    for element in get_elements(problem)
        conn = get_connectivity(element)
        geom = element("geometry", time)
        for (c, g) in zip(conn, geom)
            dofs = [field_dim*(c-1)+1, field_dim*(c-1)+2]
            X[dofs] = g
        end
    end

    # here x does not mean deformed configuration
    x = [problem.assembly.u; problem.assembly.la]
    if length(x) == 0
        info("mortar_2d_autodiff: length(x) == 0")
        # resize solution vectors according to initial configuration of this problem
        X = vec(full(sparse(findnz(X)...)))
        x = zeros(length(X)*2)
    else
        # resize initial configuration to match real dimension
        I, J, V = findnz(X)
        X = vec(full(sparse(I, J, V, length(problem.assembly.u), 1)))
    end
    ndofs = round(Int, length(x)/2)
    # at the end we should have 
#   info("mortar_2d_autodiff: size of x = $(size(x))")
#   info("mortar_2d_autodiff: size of X = $(size(X))")
#   info("mortar_2d_autodiff: ndofs = $ndofs")

    """ Calculate normal vector for slave element nodes in current configuration. """
    function calculate_normals(u::Matrix)
        normals = zeros(u)
        # 1. update nodal normals
        for element in adjacent_elements
            conn = get_connectivity(element)
            gdofs = get_gdofs(element, field_dim)
            X_el = element("geometry", time)
            u_el = Field(Vector[u[:, i] for i in conn])
            x_el = X_el + u_el
            for ip in get_integration_points(element, Val{3})
                dN = get_dbasis(element, ip)
                N = element(ip, time)
                t = sum([kron(dN[:,i], x_el[i]') for i=1:length(x_el)])
                normals[:, conn] += ip.weight*Q*t'*N
            end
        end
        slave_normals = Field(Vector[normals[:,i]/norm(normals[:,i]) for i in slave_element_nodes])
        return slave_normals
    end

    function calculate_mortar_projection(u::Matrix, n1::DVTI)
        B = SparseMatrixCOO{Real}([], [], [])

        u1 = Field([u[:,i] for i in slave_element_nodes])
        x1 = X1 + u1

        for master_element in slave_element["master elements"]
            X2 = master_element("geometry", time)
            master_element_nodes = get_connectivity(master_element)
            u2 = Field([u[:,i] for i in master_element_nodes])
            x2 = X2 + u2
            #info("master element coordinate 1 = $(ForwardDiff.get_value(x2[1]))")
            #info("master element coordinate 2 = $(ForwardDiff.get_value(x2[2]))")

            # calculate segmentation: we care only about endpoints
            # note: these are quadratic/cubic functions, analytical solution possible
            xi1a = project_from_master_to_slave(slave_element, x1, n1, x2[1])
            xi1b = project_from_master_to_slave(slave_element, x1, n1, x2[end])
            xi1 = clamp([xi1a; xi1b], -1.0, 1.0)
            l = 1/2*abs(xi1[2]-xi1[1])
            isapprox(l, 0.0) && continue # no contribution

            # integrate slave side
            D = zeros(nnodes, nnodes)
            Me = zeros(nnodes, nnodes)
            for ip in get_integration_points(slave_element, Val{5})
                dN = get_dbasis(slave_element, ip)
                # jacobian of slave element in deformed state
                j = sum([kron(dN[:,i], x1[i]') for i=1:length(x1)])
                w = ip.weight*norm(j)*l
                xi_s = dot([1/2*(1-ip.xi); 1/2*(1+ip.xi)], xi1)
                N = get_basis(slave_element, xi_s)
                D += w*diagm(vec(N))
                Me += w*N'*N
            end
            Ae = D*inv(Me)

            # integrate master side
            M = zeros(nnodes, nnodes)
            for ip in get_integration_points(slave_element, Val{5})
                dN = get_dbasis(slave_element, ip)
                # jacobian of slave element in deformed state
                j = sum([kron(dN[:,i], x1[i]') for i=1:length(x1)])
                w = ip.weight*norm(j)*l
                xi_g = dot([1/2*(1-ip.xi); 1/2*(1+ip.xi)], xi1)
                N1 = get_basis(slave_element, xi_g)
                x_g = vec(N1)*x1
                n_g = vec(N1)*n1
                #info("slave gauss point coordinates $(ForwardDiff.get_value(x_g))")
                #info("slave gauss point normal direction $(ForwardDiff.get_value(n_g))")
                xi_m = project_from_slave_to_master(master_element, x_g, n_g, x2)
                N2 = get_basis(master_element, xi_m)
                M += w*kron(Ae*N1', N2)
            end

            slave_dofs = get_gdofs(slave_element, field_dim)
            master_dofs = get_gdofs(master_element, field_dim)
            for i=1:field_dim
                add!(B, slave_dofs[i:field_dim:end], slave_dofs[i:field_dim:end], D)
                add!(B, slave_dofs[i:field_dim:end], master_dofs[i:field_dim:end], -M)
            end

        end

        return B
    end

    function calculate_contact_rhs(x::Vector)
        ndofs = round(Int, length(x)/2)
        u = x[1:ndofs]
        la = x[ndofs+1:end]
#       info("calculate_contact_rhs: size of u = $(size(u))")
#       info("calculate_contact_rhs: size of X = $(size(X))")
#       info("calculate_contact_rhs: size of la = $(size(la))")
#       info("calculate_contact_rhs: ndofs = $ndofs")
        u2 = reshape(u, field_dim, round(Int, length(u)/field_dim))

        normals = calculate_normals(u2)
        B = calculate_mortar_projection(u2, normals)
        B = sparse(B, ndofs, ndofs)
        fc = B' * la # contact force residual for r = fint + fc - fext = 0

        N = SparseMatrixCOO{Real}([], [], [])
        T = SparseMatrixCOO{Real}([], [], [])
        for (i, j) in enumerate(slave_element_nodes)
            dofs = [2*(j-1)+1, 2*(j-1)+2]
            add!(N, dofs, [dofs[1]], reshape(normals[i], 2, 1))
            add!(T, dofs, [dofs[2]], reshape(Q'*normals[i], 2, 1))
        end
        N = sparse(N, ndofs, ndofs)
        T = sparse(T, ndofs, ndofs)
        gn = -N*B*(X+u)
        gt = -T*B*(X+u)
#       gn = -N*B*u
        lan = N*la
        lat = T*la
        cn = 1.0e3
        C = lan - max(0, lan - cn*gn) + lat
        # C = lan - gn + lat - gt <-- ihan viturallensa
        # C = gn+gt <-- not working
        # C = B*(X+u)
        # C = B*u <- pitää kiinni, "tie".
        # C = N*B*u + T*la <- palikat menee väärään suuntaan
        # C = -N*B*(X+u) + T*la <- toimii suht hyvin mut kääntyy väärään suuntaan (t-suunnassa)
        # C = -N*B*(X+u) - T*la <- sama
        # C = -N*B*(X+u) - T*B*la <- sama
        # C = N*la + T*la - max(0, N*la + N*B*(X+u))
        cond = lan[1:field_dim:end] - cn*gn[1:field_dim:end]
        all_nodes = slave_element_nodes
        inactive_nodes = find(cond .<= 0)
        active_nodes = find(cond .> 0)
        inactive_nodes = setdiff(all_nodes, inactive_nodes)
        active_nodes = setdiff(all_nodes, active_nodes)
        info("S = $all_nodes, I = $inactive_nodes, A = $active_nodes")
        info("lambda = $(ForwardDiff.get_value(lan[slave_dofs]))")
        info("gn = $(ForwardDiff.get_value(gn[slave_dofs]))")
        #for j in active_nodes
        #    dofs = [field_dim*(j-1)+1, field_dim*(j-1)+2]
        #    C[dofs] = 0
        #end
        
#       C = -(N+T)*B*(X+u)
#       C = -B*(X+u)

        return [fc; C]

    end

    A, allresults = ForwardDiff.jacobian(calculate_contact_rhs, x, ForwardDiff.AllResults)
    b = -ForwardDiff.value(allresults)
    A = sparse(A)
    b = sparse(b)
    K = A[1:ndofs,1:ndofs]
    C1 = A[1:ndofs,ndofs+1:end]'
    C2 = A[ndofs+1:end,1:ndofs]
    D = A[ndofs+1:end,ndofs+1:end]
    f = b[1:ndofs]
    g = b[ndofs+1:end]
    C2[2:field_dim:end] = 0
    g[2:field_dim:end] = 0

#   C2[2:field_dim:end] = 0
#   g[2:field_dim:end] = 0
#   inactives = find(g[1:field_dim:end] .<= 0)
#   actives = find(g[1:field_dim:end] .> 0)
#   inactives = setdiff(slave_element_nodes, inactives)
#   actives = setdiff(slave_element_nodes, actives)
#   info("all nodes = $slave_element_nodes, inactives = $inactives, actives = $actives")
#   info("g = $g")

#   for j in inactives
    #    dofs = [field_dim*(j-1)+1, field_dim*(j-1)+2]
    #    K[dofs,:] = 0
    #    C1[dofs,:] = 0
    #    C2[dofs,:] = 0
    #    D[dofs,:] = 0
    #    g[dofs,:] = 0
    #end
    #for j in actives
    #    dofs = [field_dim*(j-1)+1, field_dim*(j-1)+2]
    #    C2[dofs[1],:] = 0
    #end
    add!(assembly.K, K)
    add!(assembly.C1, C1)
    add!(assembly.C2, C2)
    add!(assembly.D, D)
    add!(assembly.f, f)
    add!(assembly.g, g)
end


function assemble!{E<:MortarElements2D}(assembly::Assembly,
    problem::Problem{Mortar}, slave_element::Element{E},
    time::Real, ::Type{Val{:forwarddiff_old2}})
    haskey(slave_element, "master elements") || return
    props = problem.properties
    field_dim = get_unknown_field_dimension(problem)
    field_name = get_parent_field_name(problem)

    function calculate_interface(x::Vector)

        ndofs = round(Int, length(x)/2)
        nnodes = round(Int, ndofs/field_dim)
        u = reshape(x[1:ndofs], field_dim, nnodes)
        la = reshape(x[ndofs+1:end], field_dim, nnodes)
        fc = zeros(u)
        C = zeros(la)
        
        slave_element_nodes = get_connectivity(slave_element)
        X1 = slave_element("geometry", time)
        u1 = Field(Vector[u[:,i] for i in slave_element_nodes])
        la1 = Field(Vector[la[:,i] for i in slave_element_nodes])
        x1 = X1 + u1

        # 1. update nodal normals for this element. average nodes from adjacent elements
        adjacent_elements = find_elements(get_elements(problem), slave_element_nodes)
        adjacent_nodes = get_nodes(adjacent_elements) # including also nodes from adjacent elements
        Q = [0.0 -1.0; 1.0 0.0]
        normals = zeros(u)
        for element in adjacent_elements
            conn = get_connectivity(element)
            gdofs = get_gdofs(element, field_dim)
            X_el = element("geometry", time)
            u_el = Field(Vector[u[:, i] for i in conn])
            x_el = X_el + u_el
            for ip in get_integration_points(element, Val{3})
                dN = get_dbasis(element, ip)
                N = element(ip, time)
                t = sum([kron(dN[:,i], x_el[i]') for i=1:length(x_el)])
                normals[:, conn] += ip.weight*Q*t'*N
            end
        end
        # --> slave side normals in deformed state
        n1 = Field(Vector[normals[:,i]/norm(normals[:,i]) for i in slave_element_nodes])

        for master_element in slave_element["master elements"]

            master_element_nodes = get_connectivity(master_element)
            X2 = master_element("geometry", time)
            u2 = Field(Vector[u[:,i] for i in master_element_nodes])
            x2 = X2 + u2

            # calculate segmentation: we care only about endpoints
            # note: these are quadratic/cubic functions, analytical solution possible
            xi1a = project_from_master_to_slave(slave_element, x1, n1, x2[1])
            xi1b = project_from_master_to_slave(slave_element, x1, n1, x2[end])
            xi1 = clamp([xi1a; xi1b], -1.0, 1.0)
            l = 1/2*abs(xi1[2]-xi1[1])
            isapprox(l, 0.0) && continue # no contribution in this master element

            nnodes = size(slave_element, 2)
            De = zeros(nnodes, nnodes)
            Me = zeros(nnodes, nnodes)
            for ip in get_integration_points(slave_element, Val{5})
                # jacobian of slave element in deformed state
                dN = get_dbasis(slave_element, ip)
                j = sum([kron(dN[:,i], x1[i]') for i=1:length(x1)])
                w = ip.weight*norm(j)*l
                xi_s = dot([1/2*(1-ip.xi); 1/2*(1+ip.xi)], xi1)
                N1 = get_basis(slave_element, xi_s)
                De += w*diagm(vec(N1))
                Me += w*N1'*N1
            end
            Ae = De*inv(Me)

            slave_dofs = get_gdofs(slave_element, field_dim)
            master_dofs = get_gdofs(master_element, field_dim)

            for ip in get_integration_points(slave_element, Val{5})
                # jacobian of slave element in deformed state
                dN = get_dbasis(slave_element, ip)
                j = sum([kron(dN[:,i], x1[i]') for i=1:length(x1)])
                w = ip.weight*norm(j)*l

                # project gauss point from slave element to master element
                xi_s = dot([1/2*(1-ip.xi); 1/2*(1+ip.xi)], xi1)
                N1 = vec(get_basis(slave_element, xi_s))
                x_s = N1*x1 # coordinate in gauss point
                n_s = N1*n1 # normal direction in gauss point
                t_s = Q'*n_s # tangent direction in gauss point
                R_s = [n_s t_s]
                xi_m = project_from_slave_to_master(master_element, x_s, n_s, x2)
                N2 = vec(get_basis(master_element, xi_m))
                x_m = N2*x2
                Phi = Ae*N1

                u_s = N1*u1
                u_m = N2*u2

                la_s = Phi*la1 # traction force in gauss point
                #lan = dot(n_s, la_s) # normal component
                #lat = dot(t_s, la_s) # tangential component
                la_nt = R_s*la_s
                g = x_s-x_m
                gn = props.gap_sign*dot(n_s, g) # normal gap
                #gu = dot(n_s, u_s - u_m) # normal displacement gap
                #gt = dot(t_s, x_s - x_m) # tangential gap

                fc[:,slave_element_nodes] += w*la_s*N1'
                fc[:,master_element_nodes] -= w*la_s*N2'
                C[1,slave_element_nodes] += w*gn*Phi'
                #C[2,slave_element_nodes] += w*la_nt[2]*Phi'
                #C[2,slave_element_nodes] += w*la_nt[2,:]*Phi'
                #C[2,slave_element_nodes] += w*dot(t_s, u_s - u_m)*Phi' # <-- for tie

                #R += w*R_s

            end

        end # master elements done

        for (i, j) in enumerate(slave_element_nodes)
            n = n1[i]
            t = Q'*n
            R = [n t]
            la_nt = R*la[:,j]
            C[2,j] += la_nt[2]
        end

        return vec([fc C])

    end

    # x doesn't mean deformed configuration here
    x = [problem.assembly.u; problem.assembly.la]
    ndofs = round(Int, length(x)/2)
    #if ndofs == 0
    #    info("INITIALIZING THINGS")
    #    problem.assembly.u = zeros(16)
    #    problem.assembly.la = zeros(16)
    #    x = [problem.assembly.u; problem.assembly.la]
    #    ndofs = round(Int, length(x)/2)
    #end

    A, allresults = ForwardDiff.jacobian(calculate_interface, x, ForwardDiff.AllResults)
    b = -ForwardDiff.value(allresults)
    #b = -calculate_interface(x)
    #info("PE = $(ForwardDiff.value(allresults))")
    A = sparse(A)
    b = sparse(b)
    SparseMatrix.droptol!(A, 1.0e-12)
    SparseMatrix.droptol!(b, 1.0e-12)
    #println(A)
    K = A[1:ndofs,1:ndofs]
    C1 = transpose(A[1:ndofs,ndofs+1:end])
    C2 = A[ndofs+1:end,1:ndofs]
    D = A[ndofs+1:end,ndofs+1:end]
    f = b[1:ndofs]
    g = b[ndofs+1:end]
    add!(assembly.K, K)
    add!(assembly.C1, C1)
    add!(assembly.C2, C2)
    add!(assembly.D, D)
    add!(assembly.f, f)
    add!(assembly.g, g)

    return

end


function assemble!{E<:MortarElements2D}(assembly::Assembly,
    problem::Problem{Mortar}, slave_element::Element{E},
    time::Real, ::Type{Val{:forwarddiff}})
    haskey(slave_element, "master elements") || return
    props = problem.properties
    field_dim = get_unknown_field_dimension(problem)
    field_name = get_parent_field_name(problem)

    function calculate_interface(x::Vector)

        ndofs = round(Int, length(x)/2)
        nnodes = round(Int, ndofs/field_dim)
        u = reshape(x[1:ndofs], field_dim, nnodes)
        la = reshape(x[ndofs+1:end], field_dim, nnodes)
        fc = zeros(u)
        C = zeros(la)
        
        slave_element_nodes = get_connectivity(slave_element)
        X1 = slave_element("geometry", time)
        u1 = Field(Vector[u[:,i] for i in slave_element_nodes])
        la1 = Field(Vector[la[:,i] for i in slave_element_nodes])
        x1 = X1 + u1

        # 1. update nodal normals for this element. average nodes from adjacent elements
        adjacent_elements = find_elements(get_elements(problem), slave_element_nodes)
        adjacent_nodes = get_nodes(adjacent_elements) # including also nodes from adjacent elements
        Q = [0.0 -1.0; 1.0 0.0]
        normals = zeros(u)
        for element in adjacent_elements
            conn = get_connectivity(element)
            gdofs = get_gdofs(element, field_dim)
            X_el = element("geometry", time)
            u_el = Field(Vector[u[:, i] for i in conn])
            x_el = X_el + u_el
            for ip in get_integration_points(element, Val{3})
                dN = get_dbasis(element, ip)
                N = element(ip, time)
                t = sum([kron(dN[:,i], x_el[i]') for i=1:length(x_el)])
                normals[:, conn] += ip.weight*Q*t'*N
            end
        end
        # --> slave side normals in deformed state
        n1 = Field(Vector[ForwardDiff.get_value(normals[:,i]/norm(normals[:,i])) for i in slave_element_nodes])


        nnodes = size(slave_element, 2)
        lan_tot = zeros(nnodes) # normal pressure
        gap_tot = zeros(nnodes) # weighted normal gap

        for master_element in slave_element["master elements"]

            master_element_nodes = get_connectivity(master_element)
            X2 = master_element("geometry", time)
            u2 = Field(Vector[u[:,i] for i in master_element_nodes])
            x2 = X2 + u2

            # calculate segmentation: we care only about endpoints
            # note: these are quadratic/cubic functions, analytical solution possible
            xi1a = project_from_master_to_slave(slave_element, x1, n1, x2[1])
            xi1b = project_from_master_to_slave(slave_element, x1, n1, x2[end])
            xi1 = clamp([xi1a; xi1b], -1.0, 1.0)
            l = 1/2*abs(xi1[2]-xi1[1])
            isapprox(l, 0.0) && continue # no contribution in this master element

            De = zeros(nnodes, nnodes)
            Me = zeros(nnodes, nnodes)
            for ip in get_integration_points(slave_element, Val{5})
                # jacobian of slave element in deformed state
                dN = get_dbasis(slave_element, ip)
                j = sum([kron(dN[:,i], x1[i]') for i=1:length(x1)])
                w = ip.weight*norm(j)*l
                xi_s = dot([1/2*(1-ip.xi); 1/2*(1+ip.xi)], xi1)
                N1 = get_basis(slave_element, xi_s)
                De += w*diagm(vec(N1))
                Me += w*N1'*N1
            end
            Ae = De*inv(Me)

            slave_dofs = get_gdofs(slave_element, field_dim)
            master_dofs = get_gdofs(master_element, field_dim)

            for ip in get_integration_points(slave_element, Val{5})
                # jacobian of slave element in deformed state
                dN = get_dbasis(slave_element, ip)
                j = sum([kron(dN[:,i], x1[i]') for i=1:length(x1)])
                w = ip.weight*norm(j)*l

                # project gauss point from slave element to master element
                xi_s = dot([1/2*(1-ip.xi); 1/2*(1+ip.xi)], xi1)
                N1 = vec(get_basis(slave_element, xi_s))
                x_s = N1*x1 # coordinate in gauss point
                n_s = N1*n1 # normal direction in gauss point
                t_s = Q'*n_s # tangent direction in gauss point
                R_s = [n_s t_s]
                xi_m = project_from_slave_to_master(master_element, x_s, n_s, x2)
                N2 = vec(get_basis(master_element, xi_m))
                x_m = N2*x2
                Phi = Ae*N1

                u_s = N1*u1
                u_m = N2*u2

                la_s = Phi*la1 # traction force in gauss point
                la_nt = R_s*la_s
                gn = -dot(n_s, x_s - x_m) # normal gap

                fc[:,slave_element_nodes] += w*la_s*N1'
                fc[:,master_element_nodes] -= w*la_s*N2'
                #C[1,slave_element_nodes] += w*gn*Phi'
                
                lan_tot += w*la_nt[1]*Phi
                gap_tot += w*gn*Phi

            end

        end # master elements done

#       ncf = lan_tot - max(0, lan_tot - gap_tot)
#       info("pressure in nodes: $(ForwardDiff.get_value(lan_tot))")
#       info("weighted gap in nodes: $(ForwardDiff.get_value(gap_tot))")
#       info("ncf: $(ForwardDiff.get_value(ncf))")
#       cond = +lan_tot + gap_tot
#       cond = +lan_tot - gap_tot # singular
#       cond = -lan_tot + gap_tot
#       cond = -lan_tot - gap_tot

        for (i, j) in enumerate(slave_element_nodes)
            n = n1[i]
            t = Q'*n
            R = [n t]
            la_nt = R*la[:,j]
            info("node $j, n=$(ForwardDiff.get_value(n)) lan = $(ForwardDiff.get_value(la_nt[1])) gap = $(ForwardDiff.get_value(gap_tot[i]))")

#           if -lan_tot[i] + gap_tot[i] < 0
            if -la_nt[1] + gap_tot[i] < 0
            #if j in [31, 32, 33, 34, 35, 36]
                info("set node $j active")
                C[1,j] -= gap_tot[i]
#               C[1,j] += la_nt[1] - max(0, la_nt[1] - gap_tot[i])
                C[2,j] += la_nt[2]
            else
                info("set node $j inactive")
                C[1,j] += la1[i][1]
                C[2,j] += la1[i][2]
            end
        end

        return vec([fc C])

    end

    # x doesn't mean deformed configuration here
    x = [problem.assembly.u; problem.assembly.la]
    ndofs = round(Int, length(x)/2)
    A, allresults = ForwardDiff.jacobian(calculate_interface, x, ForwardDiff.AllResults)
    b = -ForwardDiff.value(allresults)
    #b = -calculate_interface(x)
    #info("PE = $(ForwardDiff.value(allresults))")
    A = sparse(A)
    b = sparse(b)
    SparseMatrix.droptol!(A, 1.0e-12)
    SparseMatrix.droptol!(b, 1.0e-12)
    #println(A)
    K = A[1:ndofs,1:ndofs]
    C1 = transpose(A[1:ndofs,ndofs+1:end])
    C2 = A[ndofs+1:end,1:ndofs]
    D = A[ndofs+1:end,ndofs+1:end]
    f = b[1:ndofs]
    g = b[ndofs+1:end]
    add!(assembly.K, K)
    add!(assembly.C1, C1)
    add!(assembly.C2, C2)
    add!(assembly.D, D)
    add!(assembly.f, f)
    add!(assembly.g, g)

    return

end

