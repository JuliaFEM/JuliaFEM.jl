# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using ForwardDiff

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
    tol=1.0e-10, max_iterations=20, debug=false)

    x1(xi1) = vec(get_basis(slave_element, [xi1], time))*x1_
    dx1(xi1) = vec(get_dbasis(slave_element, [xi1], time))*x1_
    n1(xi1) = vec(get_basis(slave_element, [xi1], time))*n1_
    dn1(xi1) = vec(get_dbasis(slave_element, [xi1], time))*n1_
    cross2(a, b) = cross([a; 0], [b; 0])[3]
    R(xi1) = cross2(x1(xi1)-x2, n1(xi1))
    dR(xi1) = cross2(dx1(xi1), n1(xi1)) + cross2(x1(xi1)-x2, dn1(xi1))

    xi1 = 0.0
    xi1_next = 0.0
    dxi1 = 0.0
    for i=1:max_iterations
        dxi1 = -R(xi1)/dR(xi1)
        dxi1 = clamp(dxi1, -0.3, 0.3)
        xi1_next = clamp(xi1 + dxi1, -1.0, 1.0)
        if norm(xi1_next - xi1) < tol
            return xi1_next
        end
        if debug
            info("xi1 = $xi1")
            info("R(xi1) = $(R(xi1))")
            info("dR(xi1) = $(dR(xi1))")
            info("dxi1 = $dxi1")
            info("norm = $(norm(xi1_next - xi1))")
            info("xi1_next = $xi1_next")
        end
        xi1 = xi1_next
    end

    info("x1 = $x1_")
    info("n1 = $n1_")
    info("x2 = $x2")
    info("xi1 = $xi1, dxi1 = $dxi1")
    info("-R(xi1) = $(-R(xi1))")
    info("dR(xi1) = $(dR(xi1))")
    error("find projection from master to slave: did not converge")

end

function project_from_slave_to_master{E<:MortarElements2D}(
    master_element::Element{E}, x1::Vector, n1::Vector, x2_::DVTI;
    tol=1.0e-10, max_iterations=20)

    x2(xi2) = vec(get_basis(master_element, [xi2], time))*x2_
    dx2(xi2) = vec(get_dbasis(master_element, [xi2], time))*x2_
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

"""
Frictionless 2d finite sliding contact with forwarddiff.

true/false flags: finite_sliding, friction, use_forwarddiff
"""
function assemble!(problem::Problem{Contact}, time::Float64,
                   ::Type{Val{1}},     ::Type{Val{true}},
                   ::Type{Val{false}}, ::Type{Val{true}})

    props = problem.properties
    field_dim = get_unknown_field_dimension(problem)
    field_name = get_parent_field_name(problem)
    slave_elements = get_slave_elements(problem)

    function calculate_interface(x::Vector)

        ndofs = round(Int, length(x)/2)
        nnodes = round(Int, ndofs/field_dim)
        u = reshape(x[1:ndofs], field_dim, nnodes)
        la = reshape(x[ndofs+1:end], field_dim, nnodes)
        fc = zeros(u)
        gap = zeros(u)
        C = zeros(la)
        S = Set{Int64}()

        # 1. update nodal normals for slave elements
        Q = [0.0 -1.0; 1.0 0.0]
        normals = zeros(u)
        for element in slave_elements
            conn = get_connectivity(element)
            push!(S, conn...)
            gdofs = get_gdofs(element, field_dim)
            X_el = element("geometry", time)
            u_el = Field(Vector[u[:,i] for i in conn])
            x_el = X_el + u_el
            #=
            for ip in get_integration_points(element, 3)
                dN = get_dbasis(element, ip, time)
                N = element(ip, time)
                t = sum([kron(dN[:,i], x_el[i]') for i=1:length(x_el)])
                normals[:, conn] += ip.weight*Q*t'*N
            end
            =#
            dN = get_dbasis(element, [0.0], time)
            t = sum([kron(dN[:,i], x_el[i]') for i=1:length(x_el)])
            n = Q*t'
            n /= norm(n)
            for c in conn
                normals[:,c] += n
            end
        end
        for i in 1:size(normals,2)
            normals[:,i] /= norm(normals[:,i])
        end
        # swap element normals in 2d if they point to inside of body
        if props.rotate_normals
            for i=1:size(normals,2)
                normals[:,i] = -normals[:,i]
            end
        end
        normals2 = Dict()
        for j in S
            normals2[j] = normals[:,j]
        end
        update!(slave_elements, "normal", time => normals2)

        # 2. loop all slave elements
        for slave_element in slave_elements

            slave_element_nodes = get_connectivity(slave_element)
            X1 = slave_element("geometry", time)
            u1 = Field(Vector[u[:,i] for i in slave_element_nodes])
            x1 = X1 + u1
            la1 = Field(Vector[la[:,i] for i in slave_element_nodes])
            n1 = Field(Vector[normals[:,i] for i in slave_element_nodes])
            nnodes = size(slave_element, 2)

            # 3. loop all master elements
            for master_element in slave_element("master elements", time)

                master_element_nodes = get_connectivity(master_element)
                X2 = master_element("geometry", time)
                u2 = Field(Vector[u[:,i] for i in master_element_nodes])
                x2 = X2 + u2

                #x1_midpoint = 1/2*(x1[1]+x1[2])
                #x2_midpoint = 1/2*(x2[1]+x2[2])
                #distance = ForwardDiff.get_value(norm(x2_midpoint - x1_midpoint))
                #distance > props.maximum_distance && continue

                # calculate segmentation: we care only about endpoints
                xi1a = project_from_master_to_slave(slave_element, x1, n1, x2[1])
                xi1b = project_from_master_to_slave(slave_element, x1, n1, x2[2])
                xi1 = clamp([xi1a; xi1b], -1.0, 1.0)
                l = 1/2*abs(xi1[2]-xi1[1])
                isapprox(l, 0.0) && continue # no contribution in this master element

                De = zeros(nnodes, nnodes)
                Me = zeros(nnodes, nnodes)
                for ip in get_integration_points(slave_element, 3)
                    # jacobian of slave element in deformed state
                    dN = get_dbasis(slave_element, ip, time)
                    j = sum([kron(dN[:,i], x1[i]') for i=1:length(x1)])
                    w = ip.weight*norm(j)*l
                    xi = ip.coords[1]
                    xi_s = dot([1/2*(1-xi); 1/2*(1+xi)], xi1)
                    N1 = get_basis(slave_element, xi_s, time)
                    De += w*diagm(vec(N1))
                    Me += w*N1'*N1
                end
                Ae = De*inv(Me)

                slave_dofs = get_gdofs(slave_element, field_dim)
                master_dofs = get_gdofs(master_element, field_dim)

                # 4. loop integration points of segment
                for ip in get_integration_points(slave_element, 3)
                    # jacobian of slave element in deformed state
                    dN = get_dbasis(slave_element, ip, time)
                    j = sum([kron(dN[:,i], x1[i]') for i=1:length(x1)])
                    w = ip.weight*norm(j)*l

                    # project gauss point from slave element to master element
                    xi = ip.coords[1]
                    xi_s = dot([1/2*(1-xi); 1/2*(1+xi)], xi1)
                    N1 = vec(get_basis(slave_element, xi_s, time))
                    x_s = N1*x1 # coordinate in gauss point
                    n_s = N1*n1 # normal direction in gauss point
                    t_s = Q'*n_s # tangent direction in gauss point
                    xi_m = project_from_slave_to_master(master_element, x_s, n_s, x2)
                    N2 = vec(get_basis(master_element, xi_m, time))
                    x_m = N2*x2
                    Phi = Ae*N1

                    la_s = Phi*la1 # traction force in gauss point
                    gn = -dot(n_s, x_s - x_m) # normal gap

                    fc[:,slave_element_nodes] += w*la_s*N1'
                    fc[:,master_element_nodes] -= w*la_s*N2'
                    gap[1,slave_element_nodes] += w*gn*Phi'
                    #gap[1,slave_element_nodes] += w*gn*N1'

                end # done integrating segment

            end # master elements done

        end # slave elements done

        # at this point we have calculated contact force fc and gap for all slave elements.
        # next task is to find out are they in contact or not and remove inactive nodes

        #nzgap = sort(nonzeros(sparse(ForwardDiff.get_value(gap))))
        #info("gap: $nzgap")

        for (i, j) in enumerate(sort(collect(S)))
#           if j in props.always_inactive
#               info("special node $j always inactive")
#               C[:,j] = la[:,j]
#               continue
#           end
            n = normals[:,j]
            t = Q'*n
            lan = dot(n, la[:,j])
            lat = dot(t, la[:,j])

            if lan - gap[1, j] > 0
#               info("set node $j active, normal direction = $(ForwardDiff.get_value(n)), tangent plane = $(ForwardDiff.get_value(t))")
                C[1,j] += gap[1, j]
                C[2,j] += lat
            else
                C[:,j] = la[:,j]
            end
        end

        return vec([fc C])

    end

    # x doesn't mean deformed configuration here
    x = [problem.assembly.u; problem.assembly.la]
    if length(x) == 0
        error("2d autodiff contact problem: initialize problem.assembly.u & la before solution")
    end
    A = ForwardDiff.jacobian(calculate_interface, x)
    b = calculate_interface(x)
    A = sparse(A)
    b = sparse(b)
    SparseMatrix.droptol!(A, 1.0e-12)
    SparseMatrix.droptol!(b, 1.0e-12)

    ndofs = round(Int, length(x)/2)
    K = A[1:ndofs,1:ndofs]
    C1 = transpose(A[1:ndofs,ndofs+1:end])
    C2 = A[ndofs+1:end,1:ndofs]
    D = A[ndofs+1:end,ndofs+1:end]
    f = -b[1:ndofs]
    g = -b[ndofs+1:end]

    empty!(problem.assembly)
    add!(problem.assembly.K, K)
    add!(problem.assembly.C1, C1)
    add!(problem.assembly.C2, C2)
    add!(problem.assembly.D, D)
    add!(problem.assembly.f, f)
    add!(problem.assembly.g, g)

end

