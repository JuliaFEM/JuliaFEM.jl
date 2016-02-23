# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

""" Find segment from slave element corresponding to master element nodes.

Parameters
----------
x1_, n1_
    slave element geometry and normal direction
x2
    master element node to project onto slave

Returns
-------
xi
    dimensionless coordinate on slave corresponding to
    projected master
 
"""
function project_from_master_to_slave{E<:MortarElements2D}(
    slave_element::Element{E}, x1_::DVTI, n1_::DVTI, x2::Vector;
    tol=1.0e-10, max_iterations=20)

    x1(xi1) = vec(get_basis(E, xi1))*x1_
    dx1(xi1) = vec(get_dbasis(E, xi1))*x1_
    n1(xi1) = vec(get_basis(E, xi1))*n1_
    dn1(xi1) = vec(get_dbasis(E, xi1))*n1_
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

    x2(xi2) = vec(get_basis(E, xi2))*x2_
    dx2(xi2) = vec(get_dbasis(E, xi2))*x2_
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
        n1 = Field(Vector[normals[:,i]/norm(normals[:,i]) for i in slave_element_nodes])

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

        for (i, j) in enumerate(slave_element_nodes)
            n = n1[i]
            t = Q'*n
            R = [n t]
            la_nt = R'*la[:,j]
#           info("node $j, n=$(ForwardDiff.get_value(n)) lan = $(ForwardDiff.get_value(la_nt[1])) gap = $(ForwardDiff.get_value(gap_tot[i]))")

            if la_nt[1] - gap_tot[i] > 0
                info("set node $j active")
                C[1,j] += gap_tot[i]
                C[2,j] += la_nt[2]
            else
                info("set node $j inactive")
                C[:,j] = la[:,j]
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
