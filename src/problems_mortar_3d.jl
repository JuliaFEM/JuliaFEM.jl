# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

typealias MortarElements3D Union{Tri3, Tri6, Quad4}

function project_vertex_to_auxiliary_plane(p::Vector, x0::Vector, n0::Vector)
    return p - dot(p-x0, n0)*n0
end

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

function vertex_inside_polygon(q, P; atol=1.0e-3)
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
    C = 1/sum(areas)*sum(areas.*centroids)
    return C
end

function get_cells(P, C; allow_quads=false)
    N = length(P)
    cells = Vector[]
    # shared edge etc.
    N < 3 && return cells
    # trivial cases, polygon already triangle / quadrangle
    if N == 3
        return Vector[P]
    end
    if N == 4 && allow_quads
        return Vector[P]
    end
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

""" Test does P contain q. """
function contains{T}(P::Vector{T}, q::T; check_is_close=true, rtol=1.0e-4)
    if q in P
        return true
    end
    if check_is_close
        for p in P
            if isapprox(p, q; rtol=rtol)
                return true
            end
        end
    end
    return false
end

function get_polygon_clip(xs, xm, n; debug=false)
    # objective: search does line xm1 - xm2 clip xs
    nm = length(xm)
    ns = length(xs)
    P = Vector[]

    # 1. test is master point inside slave, if yes, add to clip
    for i=1:nm
        if vertex_inside_polygon(xm[i], xs)
            debug && info("1. $(xm[i]) inside S -> push")
            push!(P, xm[i])
        end
    end

    # 2. test is slave point inside master, if yes, add to clip
    for i=1:ns
        if vertex_inside_polygon(xs[i], xm)
            contains(P, xs[i]) && continue
            debug && info("2. $(xs[i]) inside M -> push")
            push!(P, xs[i])
        end
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
            if vertex_inside_polygon(q, xm)
                contains(P, q) && continue
                debug && info("3. $q inside M -> push")
                push!(P, q)
            end
        end
    end

    return P
end

function project_vertex_to_surface{E}(p::Vector, x0::Vector, n0::Vector,
    element::Element{E}, x::DVTI, time::Real; max_iterations::Int=10, iter_tol::Float64=1.0e-6)
    basis(xi) = get_basis(element, xi, time)
    dbasis(xi) = get_dbasis(element, xi, time)
    f(theta) = basis(theta[1:2])*x - theta[3]*n0 - p
    L(theta) = inv3([dbasis(theta[1:2])*x -n0])
#   L2(theta) = inv(ForwardDiff.get_value([dbasis(theta[2:3])*x -n0]))
    # FIXME: for some reason forwarddiff gives NaN's here.
    theta = zeros(3)
    dtheta = zeros(3)
    for i=1:max_iterations
        dtheta = L(theta) * f(theta)
        theta -= dtheta
        if norm(dtheta) < iter_tol
            return theta[1:2], theta[3]
        end
    end

    info("failed to project vertex from auxiliary plane back to surface")
    info("element type: $E")
    info("element connectivity: $(get_connectivity(element))")
    info("auxiliary plane: x0 = $x0, n0 = $n0")
    info("element geometry: $(x.data)")
    info("vertex to project: $p")
    info("parameter vector before giving up: $theta")
    info("increment in parameter vector before giving up: $dtheta")
    info("norm(dtheta) before giving up: $(norm(dtheta))")
    info("f([0.0, 0.0, 0.0]) = $(f([0.0, 0.0, 0.0]))")
    info("L([0.0, 0.0, 0.0]) = $(L([0.0, 0.0, 0.0]))")

    info("iterations:")
    theta = zeros(3)
    dtheta = zeros(3)
    for i=1:max_iterations
        info("iter $i, theta = $theta")
        info("f = $(f(theta))")
        info("L = $(L(theta))")
        dtheta = L(theta) * f(theta)
        info("dtheta = $(dtheta)")
        theta -= dtheta
    end

    error("project_point_to_surface: did not converge in $max_iterations iterations!")
end

function calculate_normals(elements, time, ::Type{Val{2}}; rotate_normals=false)
    normals = Dict{Int64, Vector{Float64}}()
    for element in elements
        conn = get_connectivity(element)
        J = transpose(element([0.0, 0.0], time, Val{:Jacobian}))
        normal = cross(J[:,1], J[:,2])
        for nid in conn
            if haskey(normals, nid)
                normals[nid] += normal
            else
                normals[nid] = normal
            end
        end
    end
    # normalize to unit normal
    S = collect(keys(normals))
    for j in S
        normals[j] /= norm(normals[j])
    end
    if rotate_normals
        for j in S
            normals[j] = -normals[j]
        end
    end
    return normals
end

function check_orientation!(P, n; debug=false)
    C = mean(P)
    np = length(P)
    s = [dot(n, cross(P[i]-C, P[mod(i+1,np)+1]-C)) for i=1:np]
    all(s .< 0) && return
    debug && info("polygon not in ccw order, fixing")
    # project points to new orthogonal basis Q and sort there
    t1 = (P[1]-C)/norm(P[1]-C)
    t2 = cross(n, t1)
    Q = [n t1 t2]
    sort!(P, lt=(A, B) -> begin
        A_proj = Q'*(A-C)
        B_proj = Q'*(B-C)
        a = atan2(A_proj[3], A_proj[2])
        b = atan2(B_proj[3], B_proj[2])
        return a > b
    end)
end

function assemble!(problem::Problem{Mortar}, time::Real, ::Type{Val{2}}, ::Type{Val{false}}; debug=true)

    props = problem.properties
    field_dim = get_unknown_field_dimension(problem)
    field_name = get_parent_field_name(problem)
    slave_elements = get_slave_elements(problem)
    area = 0.0

    # 1. calculate nodal normals and tangents for slave element nodes j ∈ S
    normals = calculate_normals(slave_elements, time, Val{2};
                                rotate_normals=props.rotate_normals)
    update!(slave_elements, "normal", normals)

    # 2. loop all slave elements
    for slave_element in slave_elements

        slave_element_nodes = get_connectivity(slave_element)
        nsl = length(slave_element)
        X1 = slave_element("geometry", time)
        n1 = Field([normals[j] for j in slave_element_nodes])

        # project slave nodes to auxiliary plane (x0, Q)
        #xi = get_reference_element_midpoint(slave_element)
        xi = [1/3, 1/3]
        N = vec(get_basis(slave_element, xi, time))
        x0 = N*X1
        n0 = N*n1
        S = Vector[project_vertex_to_auxiliary_plane(p, x0, n0) for p in X1]

        # 3. loop all master elements
        for master_element in slave_element("master elements", time)

            master_element_nodes = get_connectivity(master_element)
            nm = length(master_element)
            X2 = master_element("geometry", time)

            # 3.1 project master nodes to auxiliary plane and create polygon clipping
            M = Vector[project_vertex_to_auxiliary_plane(p, x0, n0) for p in X2]
            P = get_polygon_clip(S, M, n0)
            length(P) < 3 && continue # no clipping or shared edge (no volume)
            check_orientation!(P, n0)
            C0 = calculate_centroid(P)

            De = zeros(nsl, nsl)
            Me = zeros(nsl, nm)
            ge = zeros(field_dim*nsl)

            # 4. loop integration cells
            for cell in get_cells(P, C0)
                virtual_element = Element(Tri3)
                update!(virtual_element, "geometry", cell)
                #x_cell = Field(cell)

                # construct bi-orthogonal basis
                nnodes = length(slave_element)
                if props.dual_basis
                    De = zeros(nnodes, nnodes)
                    Me = zeros(nnodes, nnodes)
                    for ip in get_integration_points(virtual_element, 3)
                        x_gauss = virtual_element("geometry", ip, time)
                        xi_s, alpha = project_vertex_to_surface(x_gauss, x0, n0, slave_element, X1, time)
                        detJ = virtual_element(ip, time, Val{:detJ})
                        w = ip.weight*detJ
                        N1 = vec(get_basis(slave_element, xi_s, time))
                        De += w*diagm(vec(N1))
                        Me += w*N1*N1'
                    end
                    Ae = De*inv(Me)
                else
                    Ae = eye(nnodes)
                end

                # 5. loop integration point of integration cell
                for ip in get_integration_points(virtual_element, 3)
                    N = vec(get_basis(virtual_element, ip, time))
                    #dN = vec(get_dbasis(virtual_element, ip, time))
                    #JC = transpose(sum([kron(dNC[:,j], x_cell[j]') for j=1:length(x_cell)]))
                    #wC = ip.weight*norm(cross(JC[:,1], JC[:,2]))
                    detJ = virtual_element(ip, time, Val{:detJ})
                    w = ip.weight*detJ

                    # project gauss point from auxiliary plane to master and slave element
                    #x_gauss = N*x_cell
                    x_gauss = virtual_element("geometry", ip, time)
                    if isnan(x_gauss[1])
                        info("is nan")
                        info("x_gauss = $x_gauss")
                        info("cell = $cell")
                        info("C0 = $C0")
                        info("P = $P")
                        info("S = $S")
                        info("M = $M")
                        info("n0 = $n0")
                        error("nan, unable to continue")
                    end
                    xi_s, alpha = project_vertex_to_surface(x_gauss, x0, n0, slave_element, X1, time)
                    xi_m, alpha = project_vertex_to_surface(x_gauss, x0, n0, master_element, X2, time)

                    # add contributions
                    N1 = vec(get_basis(slave_element, xi_s, time))
                    N2 = vec(get_basis(master_element, xi_m, time))
                    Phi = Ae*N1
                    De += w*Phi*N1'
                    Me += w*Phi*N2'
                    if props.adjust
                        u1 = slave_element("displacement", time)
                        u2 = master_element("displacement", time)
                        x_s = N1*(X1+u1)
                        x_m = N2*(X2+u2)
                        ge += w*vec((x_m-x_s)*Phi')
                    end
                    area += w
                end # integration points done

            end # integration cells done

            # 6. add contribution to contact virtual work
            sdofs = get_gdofs(problem, slave_element)
            mdofs = get_gdofs(problem, master_element)
            
            for i=1:field_dim
                lsdofs = sdofs[i:field_dim:end]
                lmdofs = mdofs[i:field_dim:end]
                add!(problem.assembly.C1, lsdofs, lsdofs, De)
                add!(problem.assembly.C1, lsdofs, lmdofs, -Me)
                add!(problem.assembly.C2, lsdofs, lsdofs, De)
                add!(problem.assembly.C2, lsdofs, lmdofs, -Me)
            end
            add!(problem.assembly.g, sdofs, ge)

        end # master elements done

    end # slave elements done, contact virtual work ready

    debug && info("area of interface: $area")

end

