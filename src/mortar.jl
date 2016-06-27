# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

type Mortar <: BoundaryProblem
    dimension :: Int
    rotate_normals :: Bool
    adjust :: Bool
    dual_basis :: Bool
    use_forwarddiff :: Bool
    distval :: Float64
    store_fields :: Vector{ASCIIString}
end

function Mortar()
    default_fields = []
    return Mortar(-1, false, false, false, false, Inf, default_fields)
end

function get_unknown_field_name(problem::Problem{Mortar})
    return "reaction force"
end

function get_formulation_type(problem::Problem{Mortar})
    return :incremental
    #=
    if problem.properties.use_forwarddiff
        return :forwarddiff
    else
        return :incremental
    end
    =#
end

typealias MortarElements2D Union{Seg2, Seg3}
typealias MortarElements3D Union{Tri3, Tri6, Quad4}

function newton(f, df, x; tol=1.0e-6, max_iterations=10)
    for i=1:max_iterations
        dx = -f(x)/df(x)
        x += dx
        if norm(dx) < tol
            return x
        end
    end
    error("Newton iteration did not converge in $max_iterations iterations")
end

function cross2(a, b)
    cross([a; 0], [b; 0])[3]
end

function get_slave_elements(problem::Problem)
    filter(el -> haskey(el, "master elements"), get_elements(problem))
end

function project_from_master_to_slave{E<:MortarElements2D}(slave_element::Element{E}, x2, time)
    x1_ = slave_element["geometry"](time)
    n1_ = slave_element["normal"](time)
    x1(xi1) = vec(get_basis(slave_element, [xi1], time))*x1_
    dx1(xi1) = vec(get_dbasis(slave_element, [xi1], time))*x1_
    n1(xi1) = vec(get_basis(slave_element, [xi1], time))*n1_
    dn1(xi1) = vec(get_dbasis(slave_element, [xi1], time))*n1_
    R(xi1) = cross2(x1(xi1)-x2, n1(xi1))
    dR(xi1) = cross2(dx1(xi1), n1(xi1)) + cross2(x1(xi1)-x2, dn1(xi1))
    xi1 = nothing
    try
        xi1 = newton(R, dR, 0.0)
    catch
        warn("projection from master to slave failed with following arguments:")
        warn("slave element x1: $x1_")
        warn("slave element n1: $n1_")
        warn("master element x2: $x2")
        warn("time: $time")
        len = norm(x1_[2] - x1_[1])
        midpnt = mean(x1_)
        dist = norm(midpnt - x2)
        distval = dist/len
        warn("midpoint of slave element: $midpnt")
        warn("length of slave element: $len")
        warn("distance between midpoint of slave element and x2: $dist")
        warn("charasteristic measure: $distval")
        rethrow()
    end
    return xi1
end

function project_from_slave_to_master{E<:MortarElements2D}(master_element::Element{E}, x1, n1, time)
    x2_ = master_element["geometry"](time)
    x2(xi2) = vec(get_basis(master_element, [xi2], time))*x2_
    dx2(xi2) = vec(get_dbasis(master_element, [xi2], time))*x2_
    cross2(a, b) = cross([a; 0], [b; 0])[3]
    R(xi2) = cross2(x2(xi2)-x1, n1)
    dR(xi2) = cross2(dx2(xi2), n1)
    xi2 = newton(R, dR, 0.0)
    return xi2
end

function calculate_normals(elements, time, ::Type{Val{1}}; rotate_normals=false)
    tangents = Dict{Int64, Vector{Float64}}()
    for element in elements
        conn = get_connectivity(element)
        X1 = element("geometry", time)
        dN = get_dbasis(element, [0.0], time)
        tangent = vec(sum([kron(dN[:,i], X1[i]') for i=1:length(X1)]))
        for nid in conn
            if haskey(tangents, nid)
                tangents[nid] += tangent
            else
                tangents[nid] = tangent
            end
        end
    end

    Q = [0.0 -1.0; 1.0 0.0]
    normals = Dict{Int64, Vector{Float64}}()
    S = collect(keys(tangents))
    for j in S
        tangents[j] /= norm(tangents[j])
        normals[j] = Q*tangents[j]
    end

    if rotate_normals
        for j in S
            normals[j] = -normals[j]
        end
    end

    return normals, tangents
end

function calculate_normals!(elements, time, ::Type{Val{1}}; rotate_normals=false)
    normals, tangents = calculate_normals(elements, time, Val{1}; rotate_normals=rotate_normals)
    for element in elements
        conn = get_connectivity(element)
        update!(element, "normal", time => [normals[j] for j in conn])
        update!(element, "tangent", time => [tangents[j] for j in conn])
    end
end

function assemble!(problem::Problem{Mortar}, time::Float64)
    if problem.properties.dimension == -1
        problem.properties.dimension = dim = size(first(problem.elements), 1)
        info("assuming dimension of mesh tie surface is $dim")
        info("if this is wrong set is manually using problem.properties.dimension")
    end
    dimension = Val{problem.properties.dimension}
    use_forwarddiff = Val{problem.properties.use_forwarddiff}
    assemble!(problem, time, dimension, use_forwarddiff)
end

function assemble!(problem::Problem{Mortar}, time::Float64, ::Type{Val{1}}, ::Type{Val{false}})

    props = problem.properties
    field_dim = get_unknown_field_dimension(problem)
    field_name = get_parent_field_name(problem)
    slave_elements = get_slave_elements(problem)

    # 1. calculate nodal normals and tangents for slave element nodes j ∈ S
    normals, tangents = calculate_normals(slave_elements, time, Val{1};
                                          rotate_normals=props.rotate_normals)
    update!(slave_elements, "normal", normals)
    update!(slave_elements, "tangent", tangents)

    # 2. loop all slave elements
    for slave_element in slave_elements

        nsl = length(slave_element)
        X1 = slave_element["geometry"](time)
        n1 = slave_element["normal"](time)

        # 3. loop all master elements
        for master_element in slave_element["master elements"](time)

            nm = length(master_element)
            X2 = master_element["geometry"](time)

            # 3.1 calculate segmentation
            xi1a = project_from_master_to_slave(slave_element, X2[1], time)
            xi1b = project_from_master_to_slave(slave_element, X2[2], time)
            xi1 = clamp([xi1a; xi1b], -1.0, 1.0)
            l = 1/2*abs(xi1[2]-xi1[1])
            isapprox(l, 0.0) && continue # no contribution in this master element

            # 3.2. bi-orthogonal basis
            De = zeros(nsl, nsl)
            Me = zeros(nsl, nsl)
            Ae = zeros(nsl, nsl)
            if props.dual_basis
                for ip in get_integration_points(slave_element, 3)
                    detJ = slave_element(ip, time, Val{:detJ})
                    w = ip.weight*detJ*l
                    xi = ip.coords[1]
                    xi_s = dot([1/2*(1-xi); 1/2*(1+xi)], xi1)
                    N1 = vec(get_basis(slave_element, xi_s, time))
                    De += w*diagm(N1)
                    Me += w*N1*N1'
                end
                Ae = De*inv(Me)
            else
                Ae = eye(nsl)
            end

            # 3.3. loop integration points of one integration segment and calculate
            # local mortar matrices
            fill!(De, 0.0)
            fill!(Me, 0.0)
            ge = zeros(field_dim*nsl)
            for ip in get_integration_points(slave_element, 2)
                detJ = slave_element(ip, time, Val{:detJ})
                w = ip.weight*detJ*l
                xi = ip.coords[1]
                xi_s = dot([1/2*(1-xi); 1/2*(1+xi)], xi1)
                N1 = vec(get_basis(slave_element, xi_s, time))
                Phi = Ae*N1
                # project gauss point from slave element to master element in direction n_s
                X_s = N1*X1 # coordinate in gauss point
                n_s = N1*n1 # normal direction in gauss point
                xi_m = project_from_slave_to_master(master_element, X_s, n_s, time)
                N2 = vec(get_basis(master_element, xi_m, time))
                X_m = N2*X2 
                De += w*Phi*N1'
                Me += w*Phi*N2'
                if props.adjust
                    haskey(slave_element, "displacement") || continue
                    haskey(master_element, "displacement") || continue
                    norm(mean(X1) - X2[1]) / norm(X1[2] - X1[1]) < props.distval || continue
                    norm(mean(X1) - X2[2]) / norm(X1[2] - X1[1]) < props.distval || continue
                    u1 = slave_element["displacement"](time)
                    u2 = master_element["displacement"](time)
                    x_s = X_s + N1*u1
                    x_m = X_m + N2*u2
                    ge += w*vec((x_m-x_s)*Phi')
                end
            end

            # add contribution to contact virtual work
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

end

# mesh tie 2d end

# mesh tie 2d forwarddiff start

function project_from_master_to_slave{E<:MortarElements2D}(
    slave_element::Element{E}, x1_::DVTI, n1_::DVTI, x2::Vector, time::Float64;
    tol=1.0e-10, max_iterations=20)

    x1(xi1) = vec(get_basis(slave_element, [xi1], time))*x1_
    dx1(xi1) = vec(get_dbasis(slave_element, [xi1], time))*x1_
    n1(xi1) = vec(get_basis(slave_element, [xi1], time))*n1_
    dn1(xi1) = vec(get_dbasis(slave_element, [xi1], time))*n1_
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
    master_element::Element{E}, x1::Vector, n1::Vector, x2_::DVTI, time::Float64;
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

""" 2d mesh tie using ForwardDiff.

Construct .. + fc*la and C(d,la)=0

"""
function assemble!(problem::Problem{Mortar}, time::Float64, ::Type{Val{1}}, ::Type{Val{true}})

    props = problem.properties
    field_dim = get_unknown_field_dimension(problem)
    field_name = get_parent_field_name(problem)
    slave_elements = get_slave_elements(problem)
    if field_name != "displacement"
        error("mortar forwarddiff assembly: only displacement field with adjust=yes supported")
    end

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
        tangents = zeros(u)
        for element in slave_elements
            conn = get_connectivity(element)
            push!(S, conn...)
            X1 = element("geometry", time)
            u1 = Field([u[:,i] for i in conn])
            x1 = X1 + u1
            dN = get_dbasis(element, [0.0], time)
            tangent = sum([kron(dN[:,i], x1[i]') for i=1:length(x1)])
            for nid in conn
                tangents[:,nid] += tangent[:]
            end
        end

        Q = [0.0 -1.0; 1.0 0.0]
        normals = zeros(u)
        for j in S
            tangents[:,j] /= norm(tangents[:,j])
            normals[:,j] = Q*tangents[:,j]
        end

        if props.rotate_normals
            for j in S
                normals[:,j] = -normals[:,j]
            end
        end

        normals2 = Dict()
        tangents2 = Dict()
        for j in S
            normals2[j] = normals[:,j]
            tangents2[j] = tangents[:,j]
        end
        update!(slave_elements, "normal", time => normals2)
        update!(slave_elements, "tangent", time => tangents2)

        # 2. loop all slave elements
        for slave_element in slave_elements

            nsl = length(slave_element)
            slave_element_nodes = get_connectivity(slave_element)
            X1 = slave_element["geometry"](time)
            u1 = Field(Vector[u[:,i] for i in slave_element_nodes])
            x1 = X1 + u1
            la1 = Field(Vector[la[:,i] for i in slave_element_nodes])
            n1 = Field(Vector[normals[:,i] for i in slave_element_nodes])


            # 3. loop all master elements
            for master_element in slave_element["master elements"](time)

                nm = length(master_element)
                master_element_nodes = get_connectivity(master_element)
                X2 = master_element["geometry"](time)
                u2 = Field(Vector[u[:,i] for i in master_element_nodes])
                x2 = X2 + u2

                # 3.1 calculate segmentation
                xi1a = project_from_master_to_slave(slave_element, x1, n1, x2[1], time)
                xi1b = project_from_master_to_slave(slave_element, x1, n1, x2[2], time)
#               xi1a = project_from_master_to_slave(slave_element, X2[1], time)
#               xi1b = project_from_master_to_slave(slave_element, X2[2], time)
                xi1 = clamp([xi1a; xi1b], -1.0, 1.0)
                l = 1/2*abs(xi1[2]-xi1[1])
                isapprox(l, 0.0) && continue # no contribution in this master element

                # 3.2. bi-orthogonal basis
                De = zeros(nsl, nsl)
                Me = zeros(nsl, nsl)
                Ae = zeros(nsl, nsl)
                if props.dual_basis
                    for ip in get_integration_points(slave_element, 3)
                        detJ = slave_element(ip, time, Val{:detJ})
                        w = ip.weight*detJ*l
                        xi = ip.coords[1]
                        xi_s = dot([1/2*(1-xi); 1/2*(1+xi)], xi1)
                        N1 = vec(get_basis(slave_element, xi_s, time))
                        De += w*diagm(N1)
                        Me += w*N1*N1'
                    end
                    Ae = De*inv(Me)
                else
                    Ae = eye(nsl)
                end

                # 3.3. loop integration points of one integration segment and calculate
                # local mortar matrices
                for ip in get_integration_points(slave_element, 3)
                    detJ = slave_element(ip, time, Val{:detJ})
                    w = ip.weight*detJ*l
                    #dN = get_dbasis(slave_element, ip, time)
                    #j = sum([kron(dN[:,i], x1[i]') for i=1:length(x1)])
                    #w = ip.weight*norm(j)*l
                    
                    xi = ip.coords[1]
                    xi_s = dot([1/2*(1-xi); 1/2*(1+xi)], xi1)
                    N1 = vec(get_basis(slave_element, xi_s, time))
                    Phi = Ae*N1
                    # project gauss point from slave element to master element in direction n_s
                    x_s = N1*x1 # coordinate in gauss point
                    n_s = N1*n1 # normal direction in gauss point
                    #xi_m = project_from_slave_to_master(master_element, X_s, n_s, time)
                    xi_m = project_from_slave_to_master(master_element, x_s, n_s, x2, time)
                    N2 = vec(get_basis(master_element, xi_m, time))
                    x_m = N2*x2 

                    la_s = Phi*la1
                    gn = dot(n_s, x_s-x_m)

                    u_s = N1*u1
                    u_m = N2*u2
                    X_s = N1*X1
                    X_m = N2*X2

                    fc[:,slave_element_nodes] += w*la_s*N1'
                    fc[:,master_element_nodes] -= w*la_s*N2'
                    #gap[1,slave_element_nodes] += w*gn*Phi'
                    gap[:,slave_element_nodes] += w*(u_s-u_m)*Phi'
                    if props.adjust
                        G = ForwardDiff.get_value(w*(X_s-X_m)*Phi')
                        gap[:,slave_element_nodes] += G
                    end
                end

            end # master elements done

        end # slave elements done, contact virtual work ready

        C = gap

        info("interface residual ready")
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

    empty!(problem.assembly)
    problem.assembly.K = K
    problem.assembly.C1 = C1
    problem.assembly.C2 = C2
    problem.assembly.D = D
    problem.assembly.f = f
    problem.assembly.g = g

end

## 3d Mortar mesh tie

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
    C = 1/sum(areas)*sum(areas.*centroids)
    return C
end

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

function get_polygon_clip(xs, xm, n; debug=false)
    # objective: search does line xm1 - xm2 clip xs
    nm = length(xm)
    ns = length(xs)
    P = Vector{Float64}[]

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
            xs[i] in P && continue
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
                q in P && continue
                debug && info("3. $q inside M -> push")
                push!(P, q)
            end
        end
    end

    return P
end

function project_vertex_to_surface{E}(p::Vector, x0::Vector, n0::Vector,
    element::Element{E}, x::DVTI, time::Real; max_iterations::Int=10, iter_tol::Float64=1.0e-9)
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

function assemble!(problem::Problem{Mortar}, time::Real, ::Type{Val{2}}; debug=true)

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
        X1 = slave_element["geometry"](time)
        n1 = Field([normals[j] for j in slave_element_nodes])

        # project slave nodes to auxiliary plane (x0, Q)
        #xi = get_reference_element_midpoint(slave_element)
        xi = [1/3, 1/3]
        N = vec(get_basis(slave_element, xi, time))
        x0 = N*X1
        n0 = N*n1
        S = Vector[project_vertex_to_auxiliary_plane(p, x0, n0) for p in X1]

        # 3. loop all master elements
        for master_element in slave_element["master elements"](time)

            master_element_nodes = get_connectivity(master_element)
            nm = length(master_element)
            X2 = master_element["geometry"](time)

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
                    De += w*N1*N1'
                    Me += w*N1*N2'
                    if props.adjust
                        u1 = slave_element["displacement"](time)
                        u2 = master_element["displacement"](time)
                        x_s = N1*(X1+u1)
                        x_m = N2*(X2+u2)
                        ge += w*vec((x_m-x_s)*N1')
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

