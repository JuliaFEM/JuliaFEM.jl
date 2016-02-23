using JuliaFEM.Core: MortarElements2D, DVTI, Assembly

import JuliaFEM.Core: project_from_master_to_slave, project_from_slave_to_master, assemble!,
get_unknown_field_dimension, get_parent_field_name, get_gdofs, find_elements, get_nodes, Field,
get_integration_points, get_basis, get_dbasis, add!

""" Find segment from slave element corresponding to master element nodes.
x1_, n1_
slave element geometry and normal direction

x2_ master element nodes to project onto slave
"""
function project_from_master_to_slave{E<:MortarElements2D}(
    slave_element::Element{E}, x1_::DVTI, n1_::DVTI, x2::Vector)

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
    for i=1:5
        dxi1 = -R(xi1)/dR(xi1)
        xi1 += dxi1
        if norm(dxi1) < 1.0e-10
            return xi1
        end
    end

    error("find projection from master to slave: did not converge")

end

function project_from_slave_to_master{E<:MortarElements2D}(
    master_element::Element{E}, x1::Vector, n1::Vector, x2_::DVTI)

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
    for i=1:5
        dxi2 = -R(xi2) / dR(xi2)
        xi2 += dxi2
        if norm(dxi2) < 1.0e-10
            return xi2
        end
    end

    error("find projection from slave to master: did not converge, last val: $xi2 and $dxi2")
    
end


function assemble!{E<:MortarElements2D}(assembly::Assembly,
    problem::Problem{Mortar}, slave_element::Element{E},
    time::Real, ::Type{Val{:forwarddiff}})
    haskey(slave_element, "master elements") || return
    props = problem.properties
    field_dim = get_unknown_field_dimension(problem)
    field_name = get_parent_field_name(problem)

    function calculate_interface(u::Matrix, la::Matrix)

        X1 = slave_element("geometry", time)
        slave_element_nodes = get_connectivity(slave_element)
        u1 = Field(Vector[u[:,i] for i in slave_element_nodes])
        la1 = Field(Vector[la[:,i] for i in slave_element_nodes])
        x1 = X1 + u1

        adjacent_elements = find_elements(get_elements(problem), slave_element_nodes)
        adjacent_nodes = get_nodes(adjacent_elements) # including also nodes from adjacent elements
        Q = [0.0 -1.0; 1.0 0.0]
        # 1. update nodal normals for this element
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

        fc = SparseMatrixCOO{Real}([], [], []) # interface virtual work
        C = SparseMatrixCOO{Real}([], [], []) # constraints
        B = SparseMatrixCOO{Real}([], [], [])

        #info("u1.data = ", ForwardDiff.get_value(u1.data))
        info("normal calculations done. looping master elements.")
        for master_element in slave_element["master elements"]
            X2 = master_element("geometry", time)
            master_element_nodes = get_connectivity(master_element)
            u2 = Field(Vector[u[:,i] for i in master_element_nodes])
            x2 = X2 + u2
            info("master element ready.")

            # calculate segmentation: we care only about endpoints
            # note: these are quadratic/cubic functions, analytical solution possible
            info("calculating segmentation.")
            xi1a = project_from_master_to_slave(slave_element, x1, n1, x2[1])
            xi1b = project_from_master_to_slave(slave_element, x1, n1, x2[end])
            xi1 = clamp([xi1a; xi1b], -1.0, 1.0)
            l = 1/2*abs(xi1[2]-xi1[1])
            isapprox(l, 0.0) && continue # no contribution
            info("xi1 = $xi1")

            info("create bi-orthogonal basis")
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
            info("bi-orthogonal basis done. integrating fc.")

            slave_dofs = get_gdofs(slave_element, field_dim)
            master_dofs = get_gdofs(master_element, field_dim)

            info("integrate fc")
            D = zeros(nnodes, nnodes)
            M = zeros(nnodes, nnodes)
            gn = zeros(nnodes)
            lan = zeros(nnodes)
            lat = zeros(nnodes)

            for ip in get_integration_points(slave_element, Val{5})
                # jacobian of slave element in deformed state
                dN = get_dbasis(slave_element, ip)
                j = sum([kron(dN[:,i], x1[i]') for i=1:length(x1)])
                w = ip.weight*norm(j)*l
                xi_s = dot([1/2*(1-ip.xi); 1/2*(1+ip.xi)], xi1)
                N1 = get_basis(slave_element, xi_s)
                # project gauss point to master element to evaluate shape function there
                x_s = vec(N1)*x1 # coordinate in gauss point
                n_s = vec(N1)*n1 # normal direction in gauss point
                xi_m = project_from_slave_to_master(master_element, x_s, n_s, x2)
                N2 = get_basis(master_element, xi_m)
                x_m = vec(N2)*x2
                Phi = vec(Ae*N1')
                la_s = Phi*la1 # traction force in gauss point
                u_s = vec(N1)*u1
                u_m = vec(N2)*u2
                #info("la_s = $(ForwardDiff.get_value(la_s))")
                SM = [slave_dofs; master_dofs]
                #N1N2 = [N1 -N2]
                #info("all_dofs = $(SM)")
                #info("shape functions = $(ForwardDiff.get_value(N1N2))")
                #for i=1:field_dim
                #    #info("add to slave dofs $(slave_dofs[i:field_dim:end])")
                #    #info("add to master dofs $(master_dofs[i:field_dim:end])")
                #    add!(fc, slave_dofs[i:field_dim:end], [1, 1], -w*la_s[i]*N1)
                #    add!(fc, master_dofs[i:field_dim:end], [1, 1], +w*la_s[i]*N2)
                    #add!(fc, slave_dofs[i:field_dim:end], [1, 1], w*la_s'*u_s[i])
                    #add!(fc, master_dofs[i:field_dim:end], [1, 1], -w*la_s'*u_m[i])
                #    add!(fc, master_dofs[i:field_dim:end], [1, 1], -w*la_s[i]*u_m)
                #end
                #add!(fc, [slave_dofs; master_dofs], [1, 1, 1, 1, 1, 1, 1, 1], w*la_s*[u_s' -u_m'])
                D += w*kron(Ae*N1', N1)
                M += w*kron(Ae*N1', N2)
                gn += -w*dot(n_s, x_s-x_m)*Phi
                lan += w*dot(n_s, la_s)*Phi
                t_s = Q'*n_s
                lat += w*dot(t_s, la_s)*Phi
            end

            #D2 = zeros(2*nnodes, 2*nnodes)
            #M2 = zeros(2*nnodes, 2*nnodes)
            #for i=1:field_dim
            #    D2[i:field_dim:end, i:field_dim:end] += D
            #    M2[i:field_dim:end, i:field_dim:end] += M
            #end
            #info("size of D2 = $(size(D2))")
            #fco = [D2 -M2]*vec(la1)
            #fco = [D -M]*la[:,slave_element_nodes]
            #info("fco = $(ForwardDiff.get_value(fco))")
            #add!(fc, [slave_dofs; master_dofs], [1, 1, 1, 1], fco)

            for i=1:field_dim
                add!(B, slave_dofs[i:field_dim:end], slave_dofs[i:field_dim:end], D)
                add!(B, slave_dofs[i:field_dim:end], master_dofs[i:field_dim:end], -M)
            end
            info("gn = $gn")
            #Cj = lan - max(0, lan - gn) + lat
            add!(C, slave_dofs[1:field_dim:end], [1, 1], gn')

        end # master elements done

        ndofs = prod(size(la))
        N = SparseMatrixCOO{Real}([], [], [])
        T = SparseMatrixCOO{Real}([], [], [])
        for (i, j) in enumerate(slave_element_nodes)
            dofs = [2*(j-1)+1, 2*(j-1)+2]
            add!(N, [dofs[1]], dofs, reshape(n1[i], 1, 2))
            add!(T, [dofs[2]], dofs, reshape(Q'*n1[i], 1, 2))
        end
        N = sparse(N, ndofs, ndofs)
        T = sparse(T, ndofs, ndofs)
        B = sparse(B, ndofs, ndofs)
        fc = B'*vec(la)
        #println(sparse(fc))
        #fc = sparse(fc, ndofs, 1)
        #println(fc)
        #dump(full(fc))
        #C = sparse(C, ndofs, 1)
        C = N*B*vec(u) + T*vec(la)
        return fc, C

    end


    function calculate_interface_PE(x::Vector)

        ndofs = round(Int, length(x)/2)
        nnodes = round(Int, ndofs/field_dim)
        u = reshape(x[1:ndofs], field_dim, nnodes)
        la = reshape(x[ndofs+1:end], field_dim, nnodes)
        #fixed_la = ForwardDiff.get_value(la)
        #fixed_u = ForwardDiff.get_value(u)
        #u = ForwardDiff.get_value(u)

        X1 = slave_element("geometry", time)
        slave_element_nodes = get_connectivity(slave_element)
        u1 = Field(Vector[u[:,i] for i in slave_element_nodes])
        la1 = Field(Vector[la[:,i] for i in slave_element_nodes])
        #fixed_la1 = Field(Vector[fixed_la[:,i] for i in slave_element_nodes])
        x1 = X1 + u1

        # 1. update nodal normals for this element
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

        Wco = 0.0
        Wla = 0.0

        for master_element in slave_element["master elements"]
            X2 = master_element("geometry", time)
            master_element_nodes = get_connectivity(master_element)
            u2 = Field(Vector[u[:,i] for i in master_element_nodes])
            x2 = X2 + u2

            # calculate segmentation: we care only about endpoints
            # note: these are quadratic/cubic functions, analytical solution possible
            xi1a = project_from_master_to_slave(slave_element, x1, n1, x2[1])
            xi1b = project_from_master_to_slave(slave_element, x1, n1, x2[end])
            xi1 = clamp([xi1a; xi1b], -1.0, 1.0)
            l = 1/2*abs(xi1[2]-xi1[1])
            isapprox(l, 0.0) && continue # no contribution

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

            for ip in get_integration_points(slave_element, Val{5})
                # jacobian of slave element in deformed state
                dN = get_dbasis(slave_element, ip)
                j = sum([kron(dN[:,i], x1[i]') for i=1:length(x1)])
                w = ip.weight*norm(j)*l
                xi_s = dot([1/2*(1-ip.xi); 1/2*(1+ip.xi)], xi1)
                N1 = vec(get_basis(slave_element, xi_s))

                # project gauss point to master element to evaluate shape function there
                x_s = N1*x1 # coordinate in gauss point
                n_s = N1*n1 # normal direction in gauss point
                t_s = Q'*n_s
                xi_m = project_from_slave_to_master(master_element, x_s, n_s, x2)
                N2 = vec(get_basis(master_element, xi_m))
                x_m = N2*x2
                Phi = Ae*N1

                gn = -dot(n_s, x_s - x_m)
                gt = dot(t_s, x_s - x_m)
                lan = dot(n_s, Phi*la1)
                lat = dot(t_s, Phi*la1)
                u_s = N1*u1
                u_m = N2*u2
                gu = dot(n_s, u_s - u_m)

                Wco += w*dot(Phi*la1, N1*u1 - N2*u2)
                #Wla += w*(lan*gn + lat*gt)
                #gn = min(0, gn)
                Wla += 1/2*w*1e6*gn*gn
                #info("gn = $(ForwardDiff.get_value(gn))")
                #Wla += w*dot(dot(n_s, Phi*la1), dot(n_s, N1*u1 - N2*u2))
            end

        end
        return Wco, Wla
    end


    function calculate_contact_rhs(x::Vector)
        ndofs = round(Int, length(x)/2)
        nnodes = round(Int, ndofs/field_dim)
        u = reshape(x[1:ndofs], field_dim, nnodes)
        la = reshape(x[ndofs+1:end], field_dim, nnodes)
        fc, C = calculate_interface(u, la)
        info("interface vector calculated.")
        return vec(full([fc; C]))
    end

    # x doesn't mean deformed configuration here
    x = [problem.assembly.u; problem.assembly.la]
    ndofs = round(Int, length(x)/2)
    if ndofs == 0
        info("INITIALIZING THINGS")
        problem.assembly.u = zeros(16)
        problem.assembly.la = zeros(16)
        x = [problem.assembly.u; problem.assembly.la]
        ndofs = round(Int, length(x)/2)
    end

    function add_fco!()
        get_PI(x::Vector) = calculate_interface_PE(x)[1]
        A, allresults = ForwardDiff.hessian(get_PI, x, ForwardDiff.AllResults)
        b = -ForwardDiff.gradient(allresults)
        info("PE = $(ForwardDiff.value(allresults))")
        A = sparse(A)
        b = sparse(b)
        SparseMatrix.droptol!(A, 1.0e-12)
        SparseMatrix.droptol!(b, 1.0e-12)
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
    end
    #add_fco!()

    function add_wla!()
        get_PI(x::Vector) = calculate_interface_PE(x)[2]
        A, allresults = ForwardDiff.hessian(get_PI, x, ForwardDiff.AllResults)
        b = -ForwardDiff.gradient(allresults)
        info("PE = $(ForwardDiff.value(allresults))")
        A = sparse(A)
        b = sparse(b)
        SparseMatrix.droptol!(A, 1.0e-12)
        SparseMatrix.droptol!(b, 1.0e-12)
        #info("A")
        #println(full(A))
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
    end
    add_wla!()

    return

end

mesh, body1, body2, bc_top, bc_bottom, contact = divided_block_problem()
bc_top.properties.formulation = :incremental
bc_bottom.properties.formulation = :incremental
contact.properties.formulation = :forwarddiff
contact.assembly.u = zeros(16)
contact.assembly.la = zeros(16)
assemble!(contact.assembly, contact, contact.elements[1], 0.0, Val{:forwarddiff})

