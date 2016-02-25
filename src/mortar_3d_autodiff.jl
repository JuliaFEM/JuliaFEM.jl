# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

""" Assemble Mortar problem for three-dimensional problems, i.e. for Tri3, Tri6, Quad4, Quad8, Quad9 elements. """
function assemble!(problem::Problem{Mortar}, time::Real, ::Type{Val{3}})

    props = problem.properties
    field_dim = get_unknown_field_dimension(problem)
    field_name = get_parent_field_name(problem)

    function calculate_interface(x::Vector)

        ndofs = round(Int, length(x)/2) # x = [u; la]
        nnodes = round(Int, ndofs/field_dim)
        u = reshape(x[1:ndofs], field_dim, nnodes)
        la = reshape(x[ndofs+1:end], field_dim, nnodes)
        fc = zeros(u)
        gap = zeros(u)
        gap_added = zeros(size(u)...)
        C = zeros(la)
        all_slave_nodes = Set{Int64}()

        # 1. calculate and average node normals for slave element nodes
        normal = zeros(u)
        tangent1 = zeros(u)
        tangent2 = zeros(u)
        for element in get_elements(problem)
            haskey(element, "master elements") || continue
            conn = get_connectivity(element)
            push!(all_slave_nodes, conn...)
            gdofs = get_gdofs(element, field_dim)
            X_el = element("geometry", time)
            u_el = Field(Vector[u[:,i] for i in conn])
            x_el = X_el + u_el
            for ip in get_integration_points(element, Val{3})
                dN = get_dbasis(element, ip)
                N = element(ip, time)
                j = transpose(sum([kron(dN[:,i], x_el[i]') for i=1:length(x_el)]))
                #info("size of j = $(size(j))")
                n = reshape(cross(j[:,1], j[:,2]), 3, 1)
                normal[:, conn] += ip.weight*n*N
            end
        end
        # calculate tangents
        for i in 1:size(normal, 2)
            i in all_slave_nodes || continue
            normal[:,i] /= norm(normal[:,i])
            u1 = normal[:,i]
            j = indmax(abs(u1))
            v2 = zeros(3)
            v2[mod(j,3)+1] = 1.0
            u2 = v2 - dot(u1,v2)/dot(v2,v2)*v2
            u3 = cross(u1,u2)
            tangent1[:,i] = u2/norm(u2)
            tangent2[:,i] = u3/norm(u3)
        end
        if props.rotate_normals
            for i=1:size(normal, 2)
                normal[:,i] = -normal[:,i]
            end
        end

        # 2. loop slave elements and find contact segments
        for slave_element in get_elements(problem)
            haskey(slave_element, "master elements") || continue

            slave_element_nodes = get_connectivity(slave_element)
            X1 = slave_element("geometry", time)
            u1 = Field(Vector[u[:,i] for i in slave_element_nodes])
            x1 = X1 + u1
            la1 = Field(Vector[la[:,i] for i in slave_element_nodes])
            n1 = Field(Vector[normal[:,i] for i in slave_element_nodes])
            t1 = Field(Vector[tangent1[:,i] for i in slave_element_nodes])
            t2 = Field(Vector[tangent2[:,i] for i in slave_element_nodes])
            nnodes = size(slave_element, 2)
            update!(slave_element, "normals", time => ForwardDiff.get_value(n1.data))

            # create auxiliary plane (x0, Q)
            xi = get_reference_element_midpoint(slave_element)
            N = vec(get_basis(slave_element, xi))
            x0 = N*x1
            Q = [N*n1 N*t1 N*t2]

            # project slave nodes to auxiliary plane
            S = hcat([project_vertex_to_auxiliary_plane(p, x0, Q) for p in x1]...)

            # 3. loop all master elements
            for master_element in slave_element["master elements"]

                master_element_nodes = get_connectivity(master_element)
                X2 = master_element("geometry", time)
                u2 = Field(Vector[u[:,i] for i in master_element_nodes])
                x2 = X2 + u2

                x1_midpoint = mean(x1)
                x2_midpoint = mean(x2)
                distance = ForwardDiff.get_value(norm(x2_midpoint - x1_midpoint))
                distance > props.maximum_distance && continue

                # project master nodes to auxiliary plane
                M = hcat([project_vertex_to_auxiliary_plane(p, x0, Q) for p in x2]...)

                # create polygon clipping on auxiliary plane
#=
                P = nothing
                neighbours = nothing
                try
                catch
                    info("polygon clipping failed")
                    info("S = ")
                    dump(ForwardDiff.get_value(S))
                    info("M = ")
                    dump(ForwardDiff.get_value(M))
                    error("cannot continue")
                end
=#
                P, neighbours = clip_polygon(S, M)
                isa(P, Void) && continue # no clipping
                info("polygon clip found: S = $(ForwardDiff.get_value(S)), M = $(ForwardDiff.get_value(M)), P = $(ForwardDiff.get_value(P))")
                # special case, shared edge but no shared volume
                size(P, 2) < 3 && continue
                gap_added += 1

                # clip polygon centerpoint
                #C0 = calculate_polygon_centerpoint(P)
                C0 = vec(mean(P, 2))
                npts = size(P, 2) # number of vertices in polygon
                for pnt=1:npts # loop integration cells
                    x_cell = Field(Vector[C0, P[:,pnt], P[:,mod(pnt,npts)+1]])
#=
                    try
                    catch
                        info("centerpoint: $(ForwardDiff.get_value(C0))")
                        info("P1 = $(ForwardDiff.get_value(P[:,pnt]))")
                        info("P2 = $(ForwardDiff.get_value(P[:,mod(pnt,npts)+1]))")
                        data = Vector[C0, P[:,pnt], P[:,mod(pnt,npts)+1]]
                        info("data = $(ForwardDiff.get_value(data))")
                        rethrow()
                    end
=#
                    # create dual basis
                    De = zeros(nnodes, nnodes)
                    Me = zeros(nnodes, nnodes)
                    for ip in get_integration_points(Tri3, Val{5})
                        N = vec(get_basis(Tri3, ip.xi))
                        x_g = N*x_cell
                        theta = zeros(3)
                        try
                            theta = project_vertex_from_plane_to_surface(x_g, x0, Q, slave_element, x1, time)
                        catch
                            info("creating dual basis did fail for finding projection back to slave surface.")
                            info("cell coords in auxiliary plane are:")
                            info(ForwardDiff.get_value(x_cell.data))
                            info("interpolated value x_g is $(ForwardDiff.get_value(x_g))")
                            info("clip polygon centerpoint is $(ForwardDiff.get_value(C0))")
                            info("clip polygon is $(ForwardDiff.get_value(P))")
                            info("slave side nodes projected onto a plane are $(ForwardDiff.get_value(S))")
                            info("master side nodes projected onto a plane are $(ForwardDiff.get_value(M))")
                            rethrow()
                        end
                        xi_slave = theta[2:3]
                        N1 = slave_element(xi_slave, time)

                        # jacobian determinant on integration cell
                        dNC = get_dbasis(Tri3, ip.xi)
                        JC = sum([kron(dNC[:,j], x_cell[j]') for j=1:length(x_cell)])
                        wC = ip.weight*det(JC)

                        De += wC*diagm(vec(N1))
                        Me += wC*N1'*N1
                    end
                    Ae = De*inv(Me)

                    # loop integration points of cell
                    for ip in get_integration_points(Tri3, Val{5})
                        N = vec(get_basis(Tri3, ip.xi))
                        x_g = N*x_cell
                        # project gauss point back to element surfaces
                        theta1 = project_vertex_from_plane_to_surface(x_g, x0, Q, slave_element, x1, time)
                        theta2 = project_vertex_from_plane_to_surface(x_g, x0, Q, master_element, x2, time)
                        xi_slave = theta1[2:3]
                        xi_master = theta2[2:3]

                        # evaluate shape functions, calculate contact force and gap
                        N1 = vec(get_basis(slave_element, xi_slave))
                        N2 = vec(get_basis(master_element, xi_master))
                        Phi = Ae*N1

                        # jacobian determinant of integration cell
                        dNC = get_dbasis(Tri3, ip.xi)
                        JC = sum([kron(dNC[:,j], x_cell[j]') for j=1:length(x_cell)])
                        wC = ip.weight*det(JC)

                        x_s = N1*x1
                        n_s = N1*n1
                        x_m = N2*x2
                        la_s = Phi*la1
                        gn = props.gap_sign*dot(n_s, x_s - x_m)
                        fc[:,slave_element_nodes] += wC*la_s*N1'
                        fc[:,master_element_nodes] -= wC*la_s*N2'
                        wg = wC*gn*Phi'
                        if any(isnan(wg))
                            info("gap has NaNs!")
                            info("wC = $(ForwardDiff.get_value(wC))")
                            info("gn = $(ForwardDiff.get_value(gn))")
                            info("Phi = $(ForwardDiff.get_value(Phi))")
                            info("xi_slave = $(ForwardDiff.get_value(xi_slave))")
                            info("N1 = $(ForwardDiff.get_value(N1))")
                            info("Ae = $(ForwardDiff.get_value(Ae))")
                            info("De = $(ForwardDiff.get_value(De))")
                            info("Me = $(ForwardDiff.get_value(Me))")
                            info("x_cell(data) = $(ForwardDiff.get_value(x_cell.data))")
                            info("C0 = $(ForwardDiff.get_value(C0))")
                            info("P = $(ForwardDiff.get_value(P))")
                            error("fix this")
                        end
                        gap[1,slave_element_nodes] += wg
                        gap_added[1,slave_element_nodes] += 1
                    end # done integrating cell

                end # done for all cells in this segment

            end # done all master elements for this slave element

        end # done all slave elements

        # like in 2d, check contact in nodes based on a complementarity condition

        nzgap = sort(nonzeros(sparse(ForwardDiff.get_value(gap))))
        all_slave_nodes = sort(collect(all_slave_nodes))
        info("gap: $nzgap")
        info("size of normal = $(size(normal))")
        info("size of la = $(size(la))")
        info("size of C = $(size(C))")
        info("S = $all_slave_nodes")

        for (i, j) in enumerate(all_slave_nodes)
            if j in props.always_inactive
                info("special node $j always inactive")
                C[:,j] = la[:,j]
                continue
            end
            n = normal[:,j]
            t1 = tangent1[:,j]
            t2 = tangent2[:,j]
            lan = dot(n, la[:,j])

#           if lan - gap[1, j] > 0
                info("set node $j active, normal direction = $(ForwardDiff.get_value(n)), tangent plane = $(ForwardDiff.get_value(t1)) x $(ForwardDiff.get_value(t2))")
                C[1,j] += gap[1, j]
                C[2,j] += dot(t1, la[:,j])
                C[3,j] += dot(t2, la[:,j])
 #          else
 #              C[:,j] = la[:,j]
 #          end
        end

        for (i, j) in enumerate(all_slave_nodes)
            Ci = ForwardDiff.get_value(C[:,j])
            gapi = ForwardDiff.get_value(gap[:,j])
            fci = ForwardDiff.get_value(fc[:,j])
            lai = ForwardDiff.get_value(la[:,j])
            ui = ForwardDiff.get_value(u[:,j])
            ni = ForwardDiff.get_value(normal[:,j])
            gai = gap_added[:,j]
            info("$i/$j: C = $Ci, f = $fci, gap = $gapi, la = $lai, u = $ui, n = $ni, gai = $gai")
        end

        return vec([fc C])
    end

    # x doesn't mean deformed configuration here
    x = [problem.assembly.u; problem.assembly.la]
    ndofs = round(Int, length(x)/2)
    A, allresults = ForwardDiff.jacobian(calculate_interface, x,
                            ForwardDiff.AllResults, cache=autodiffcache)
    b = -ForwardDiff.value(allresults)

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

    slaves = [101,108,111,112,113,120,123,124,125,126,129,130,149,150,151,152]
    for j in slaves
        dofs = [3*(j-1)+1, 3*(j-1)+2, 3*(j-1)+3]
        info("slave node $j, dofs $dofs")
        info("Stiffness: $(K[dofs,:])")
        info("force fc: $(C1[dofs,:])")
        info("constraint: $(C2[dofs,:])")
        info("lambdas: $(D[dofs,:])")
        info("f  = $(f[dofs]), g = $(g[dofs])")
    end

    empty!(problem.assembly)
    add!(problem.assembly.K, K)
    add!(problem.assembly.C1, C1)
    add!(problem.assembly.C2, C2)
    add!(problem.assembly.D, D)
    add!(problem.assembly.f, f)
    add!(problem.assembly.g, g)

    return problem.assembly
end

