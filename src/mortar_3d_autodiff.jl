# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

""" Fast inverse of 3x3 matrix. """
function inv3(P::Matrix)
    n, m = size(P)
    @assert n == m == 3
    a, b, c, d, e, f, g, h, i = P
    A = e*i - f*h
    B = -d*i + f*g
    C = d*h - e*g
    D = -b*i + c*h
    E = a*i - c*g
    F = -a*h + b*g
    G = b*f - c*e
    H = -a*f + c*d
    I = a*e - b*d
    return 1/(a*A + b*B + c*C)*[A B C; D E F; G H I]
end

""" Project vertex `p` from element surface to auxiliary plane defined
with centerpoint `x0` and normal direction `n0`.
"""
function project_vertex_to_auxiliary_plane(p::Vector, x0::Vector, n0::Vector)
    return p - dot(p-x0, n0)*n0
end

""" Project vertex `p` from auxiliary plane (x0, n0) back to element surface.

This requires solving nonlinear system of equations

f(α,ξ₁,ξ₂) = Nₖξₖ - αn₀ - p = 0

"""
function project_vertex_to_surface{E}(p::Vector, x0::Vector, n0::Vector,
    element::Element{E}, x::DVTI, time::Real; max_iterations::Int=10, iter_tol::Float64=1.0e-9)
    basis(xi) = get_basis(E, xi)
    dbasis(xi) = get_dbasis(E, xi)
    f(theta) = basis(theta[1:2])*x - theta[3]*n0 - p
    L(theta) = inv3([dbasis(theta[1:2])*x -n0])
#   L2(theta) = inv(ForwardDiff.get_value([dbasis(theta[2:3])*x -n0]))
    # FIXME: for some reason forwarddiff gives NaN's here.
    theta = zeros(3)
    dtheta = zeros(3)
    for i=1:max_iterations
        dtheta = L(theta) * f(theta)
        theta -= dtheta
        norm(ForwardDiff.get_value(dtheta)) < iter_tol && return theta[1:2], theta[3]
    end

    info("failed to project vertex from auxiliary plane back to surface")
    info("element type: $E")
    info("element connectivity: $(get_connectivity(element))")
    info("auxiliary plane: x0 = $(ForwardDiff.get_value(x0)), n0 = $(ForwardDiff.get_value(n0))")
    info("element geometry: $(ForwardDiff.get_value(x.data))")
    info("vertex to project: $(ForwardDiff.get_value(p))")
    info("parameter vector before giving up: $(ForwardDiff.get_value(theta)')")
    info("increment in parameter vector before giving up: $(ForwardDiff.get_value(dtheta)')")
    info("norm(dtheta) before giving up: $(ForwardDiff.get_value(norm(dtheta)))")
    info("f([0.0, 0.0, 0.0]) = $(ForwardDiff.get_value(f([0.0, 0.0, 0.0]))')")
    info("L([0.0, 0.0, 0.0]) = $(ForwardDiff.get_value(L([0.0, 0.0, 0.0])))")

    info("iterations:")
    theta = zeros(3)
    dtheta = zeros(3)
    for i=1:max_iterations
        info("iter $i, theta = $(ForwardDiff.get_value(theta)')")
        info("f = $(ForwardDiff.get_value(f(theta))')")
        info("L = $(ForwardDiff.get_value(L(theta)))")
#       info("L2 = $(ForwardDiff.get_value(L2(theta)))")
        dtheta = L(theta) * f(theta)
        info("dtheta = $(ForwardDiff.get_value(dtheta)')")
        theta -= dtheta
    end

    error("project_point_to_surface: did not converge in $max_iterations iterations!")
end

""" Test is q inside sm.
http://bbs.dartmouth.edu/~fangq/MATH/download/source/Determining%20if%20a%20point%20lies%20on%20the%20interior%20of%20a%20polygon.htm
"""
function vertex_inside_polygon(q, P; atol=1.0e-6)
    N = length(P)
    angle = 0.0
    for i=1:N
        A = P[i] - q
        B = P[mod(i,N)+1] - q
        c = norm(A)*norm(B)
        isapprox(c, 0.0; atol=atol) && return true
        cosa = dot(A,B)/c
        isapprox(cosa, 1.0; atol=atol) && return false
        isapprox(cosa, -1.0; atol=atol) && return true
        try
            angle += acos(cosa)
        catch
            info("Unable to calculate acos($(ForwardDiff.get_value(cosa))) when determining is a vertex inside polygon.")
            info("Polygon is: $(ForwardDiff.get_value(P)) and vertex under consideration is $(ForwardDiff.get_value(q))")
            info("Polygon corner point in loop: A=$(ForwardDiff.get_value(A)), B=$(ForwardDiff.get_value(B))")
            info("c = ||A||*||B|| = $(ForwardDiff.get_value(c))")
            rethrow()
        end
    end
    return isapprox(angle, 2*pi; atol=atol)
end

function calculate_centroid(P)
    N = length(P)
    P0 = P[1]
    areas = [norm(1/2*cross(P[i]-P0, P[mod(i,N)+1]-P0)) for i=2:N]
    centroids = [1/3*(P0+P[i]+P[mod(i,N)+1]) for i=2:N]
    #info("areas: $areas")
    #info("centroids: $centroids")
    C = 1/sum(areas)*sum(areas.*centroids)
    return C
end

function get_polygon_clip(xs, xm, n)
    # objective: search does line xm1 - xm2 clip xs
    nm = length(xm)
    ns = length(xs)
    P = []

    # 1. test is master point inside slave, if yes, add to clip
    for i=1:nm
        vertex_inside_polygon(xm[i], xs) && push!(P, xm[i])
    end

    # 2. test is slave point inside master, if yes, add to clip
    for i=1:ns
        vertex_inside_polygon(xs[i], xm) && push!(P, xs[i])
    end
    
    for i=1:nm
        # 2. find possible intersection
        xm1 = xm[i]
        xm2 = xm[mod(i,nm)+1]
        #info("intersecting line $xm1 -> $xm2")
        for j=1:ns
            xs1 = xs[j]
            xs2 = xs[mod(j,ns)+1]
            #info("clipping polygon edge $xs1 -> $xs2")
            tnom = dot(cross(xm1-xs1, xm2-xm1), n)
            tdenom = dot(cross(xs2-xs1, xm2-xm1), n)
            isapprox(tdenom, 0) && continue
            t = tnom/tdenom
            (0 <= t <= 1) || continue
            q = xs1 + t*(xs2 - xs1)
            #info("t=$t, q=$q, q ∈ xm ? $(vertex_inside_polygon(q, xm))")
            vertex_inside_polygon(q, xm) && push!(P, q)
        end
    end

    return P
end

""" Divide polygon to triangle cells. """
function get_cells(P, C)
    N = length(P)
    cells = Vector[]
    # shared edge etc.
    N < 3 && return cells
    # trivial case, polygon already triangle / quadrangle
    #N == 3 && return Vector[P]
    #N == 4 && return Vector[P]
    #V = sum([cross(P[i], P[mod(i,N)+1]) for i=1:N])
    #A = 1/2*abs(dot(n, V))
    #info("A = $A")
    cells = Vector[Vector[C, P[i], P[mod(i,N)+1]] for i=1:N]
    return cells

    maxa = 0.0
    maxj = 0
    for i=1:N
        A = P[i] - C
        B = P[mod(i,N)+1] - C
        theta = acos(dot(A,B)/(norm(A)*norm(B)))
        if theta > maxa
            maxa = theta
            maxj = i
        end
    end
    info("max angle $(maxa/pi*180) at index $maxj, N=$N")
    indices = mod(collect(maxj:maxj+N), N)
    info("indices = $indices")
end


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
#       gap_added = zeros(size(u)...)
        C = zeros(la)
        all_slave_nodes = Set{Int64}()
        slave_surface_area = 0.0
        slave_surface_area_2 = 0.0
        slave_element_areas = []

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
        for i in all_slave_nodes
            normal[:,i] /= norm(normal[:,i])
            U1 = normal[:,i]
            j = indmax(abs(U1))
            V2 = zeros(3)
            V2[mod(j,3)+1] = 1.0
            U2 = V2 - dot(U1,V2)/dot(V2,V2)*V2
            U3 = cross(U1,U2)
            tangent1[:,i] = U2/norm(U2)
            tangent2[:,i] = U3/norm(U3)
        end
        if props.rotate_normals
            for i=1:size(normal, 2)
                normal[:,i] = -normal[:,i]
            end
        end

        # 2. loop slave elements and find contact segments
        for slave_element in get_elements(problem)
            slave_element_area = 0.0
            haskey(slave_element, "master elements") || continue
            info("new slave element")

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
            #Q = [N*n1 N*t1 N*t2]
            x0 = N*x1
            n0 = N*n1

            # project slave nodes to auxiliary plane
            #S = hcat([project_vertex_to_auxiliary_plane(p, x0, Q) for p in x1]...)
            S = Vector[project_vertex_to_auxiliary_plane(p, x0, n0) for p in x1]

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
                #M = hcat([project_vertex_to_auxiliary_plane(p, x0, Q) for p in x2]...)
                M = Vector[project_vertex_to_auxiliary_plane(p, x0, n0) for p in x2]

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
                #P, neighbours = clip_polygon(S, M)
                #isa(P, Void) && continue # no clipping
                #info("polygon clip found: S = $(ForwardDiff.get_value(S)), M = $(ForwardDiff.get_value(M)), P = $(ForwardDiff.get_value(P))")
                # special case, shared edge but no shared volume
                #size(P, 2) < 3 && continue
                #gap_added += 1
                P = get_polygon_clip(S, M, n0)
                length(P) < 3 && continue # no clipping or shared edge (no volume)

                # clip polygon centerpoint
                #C0 = calculate_polygon_centerpoint(P)
                #C0 = vec(mean(P, 2))
                act = setdiff(collect(1:3), indmax(n0))
                function comparator(A,B)
                    A -= mean(P)
                    B -= mean(P)
                    atan2(A[act[1]], A[act[2]]) < atan2(B[act[1]], B[act[2]])
                end
                sort!(P, lt=comparator)
                #sort!(P, lt=(A,B) -> dot(n0, cross(A-C0, B-C0)) > 0)
                #npts = size(P, 2) # number of vertices in polygon
                #for pnt=1:npts # loop integration cells
                #    x_cell = Field(Vector[C0, P[:,pnt], P[:,mod(pnt,npts)+1]])
                #info("integrating cell")
                #info("P = $(ForwardDiff.get_value(P))")

                for i=2:length(P)
                    V = cross(P[i]-P[1], P[mod(i,length(P))+1]-P[1])
                    slave_element_area += 1/2*dot(n0, V)
                end
                C0 = calculate_centroid(P)
                #=
                for x_cell in get_cells(P, C0)
                    V = cross(x_cell[2]-x_cell[1], x_cell[3]-x_cell[1])
                    slave_element_area += 1/2*dot(n0, V)
                end
                =#

                for x_cell_ in get_cells(P, C0)
                    #info("cell : $(ForwardDiff.get_value(x_cell_))")
                    x_cell = Field(x_cell_)
                    V = cross(x_cell[2]-x_cell[1], x_cell[3]-x_cell[1])
                    #info("V = $(ForwardDiff.get_value(V))")
                    #slave_element_area += 1/2*dot(n0, V)
#                   slave_surface_area_2 += 1/2*jC
#                   slave_element_area += 1/2*jC
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
#=
                        theta = zeros(3)
                        try
                            theta = project_vertex_to_surface(x_g, x0, n0, slave_element, x1, time)
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
=#
                        xi_slave, alpha = project_vertex_to_surface(x_g, x0, n0, slave_element, x1, time)
                        N1 = slave_element(xi_slave, time)

                        # jacobian determinant on integration cell combined with integration weight
                        dNC = get_dbasis(Tri3, ip.xi)
                        JC = transpose(sum([kron(dNC[:,j], x_cell[j]') for j=1:length(x_cell)]))
                        jC = norm(cross(JC[:,1], JC[:,2]))
                        #info("dNC = $dNC")
                        #info("x_cell = $(ForwardDiff.get_value(x_cell.data))")
                        #info("JC = $(ForwardDiff.get_value(JC))")
                        #det(JC)
                        wC = ip.weight*jC

                        De += wC*diagm(vec(N1))
                        Me += wC*N1'*N1
                    end
                    Ae = De*inv(Me)

                    # loop integration points of cell
                    for ip in get_integration_points(Tri3, Val{5})
                        N = vec(get_basis(Tri3, ip.xi))
                        x_g = N*x_cell
                        # project gauss point back to element surfaces
                        xi_slave, alpha = project_vertex_to_surface(x_g, x0, n0, slave_element, x1, time)
                        xi_master, alpha = project_vertex_to_surface(x_g, x0, n0, master_element, x2, time)

                        # evaluate shape functions, calculate contact force and gap
                        N1 = vec(get_basis(slave_element, xi_slave))
                        N2 = vec(get_basis(master_element, xi_master))
                        Phi = Ae*N1

                        dNC = get_dbasis(Tri3, ip.xi)
                        JC = transpose(sum([kron(dNC[:,j], x_cell[j]') for j=1:length(x_cell)]))
                        jC = norm(cross(JC[:,1], JC[:,2]))
                        #dNC = get_dbasis(Tri3, ip.xi)
                        #JC = sum([kron(dNC[:,j], x_cell[j]') for j=1:length(x_cell)])
                        #jC = det(JC)
                        wC = ip.weight*jC

                        x_s = N1*x1
                        n_s = N1*n1
                        x_m = N2*x2
                        la_s = Phi*la1
                        gn = props.gap_sign*dot(n_s, x_s - x_m)
                        fc[:,slave_element_nodes] += -1*wC*la_s*N1'
                        fc[:,master_element_nodes] -= -1*wC*la_s*N2'
                        gap[1,slave_element_nodes] += wC*gn*Phi'

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
#                       gap_added[1,slave_element_nodes] += 1
                        slave_surface_area += wC
                        #slave_element_area += wC
                    end # done integrating cell

                end # done for all cells in this segment

            end # done all master elements for this slave element
            push!(slave_element_areas, slave_element_area)

        end # done all slave elements

        # like in 2d, check contact in nodes based on a complementarity condition

        nzgap = sort(nonzeros(sparse(ForwardDiff.get_value(gap))))
        all_slave_nodes = sort(collect(all_slave_nodes))
        info("gap: $nzgap")
        info("size of normal = $(size(normal))")
        info("size of la = $(size(la))")
        info("size of C = $(size(C))")
        info("S = $all_slave_nodes")
        info("slave surface area: $(ForwardDiff.get_value(slave_surface_area))")
        info("slave surface area 2: $(ForwardDiff.get_value(slave_surface_area_2))")
        info("slave element areas:")
        for (i, a) in enumerate(slave_element_areas)
            info("element $i, area = $(ForwardDiff.get_value(a))")
        end
        for (i, element) in enumerate(get_elements(problem))
            haskey(element, "master elements") || continue
            info("element $i geometry: $(element("geometry", time).data)")
        end

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
#           else
#               C[:,j] = la[:,j]
#           end
        end

        for (i, j) in enumerate(all_slave_nodes)
            Ci = ForwardDiff.get_value(C[:,j])
            gapi = ForwardDiff.get_value(gap[:,j])
            fci = ForwardDiff.get_value(fc[:,j])
            lai = ForwardDiff.get_value(la[:,j])
            ui = ForwardDiff.get_value(u[:,j])
            ni = ForwardDiff.get_value(normal[:,j])
#           gai = gap_added[:,j]
            info("$i/$j: C = $Ci, f = $fci, gap = $gapi, la = $lai, u = $ui, n = $ni")
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

#=
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
=#

    empty!(problem.assembly)
    add!(problem.assembly.K, K)
    add!(problem.assembly.C1, C1)
    add!(problem.assembly.C2, C2)
    add!(problem.assembly.D, D)
    add!(problem.assembly.f, f)
    add!(problem.assembly.g, g)

    return problem.assembly
end

