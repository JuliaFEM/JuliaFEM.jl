# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

const MortarElements3D = Union{Tri3,Tri6,Quad4}

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
        #try
        angle += acos(cosa)
        #catch
        #    @info("Unable to calculate acos($(ForwardDiff.get_value(cosa))) when determining is a vertex inside polygon.")
        #    @info("Polygon is: $(ForwardDiff.get_value(P)) and vertex under consideration is $(ForwardDiff.get_value(q))")
        #    @info("Polygon corner point in loop: A=$(ForwardDiff.get_value(A)), B=$(ForwardDiff.get_value(B))")
        #    @info("c = ||A||*||B|| = $(ForwardDiff.get_value(c))")
        #    rethrow()
        #end
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
    cells = Vector[Vector[C, P[i], P[mod(i,N)+1]] for i=1:N]
    return cells
end

""" Test does vector P contain approximately q. This function uses isapprox()
internally to make boolean test.

Examples
--------
julia> P = Vector[[1.0, 1.0], [2.0, 2.0]]
2-element Array{Array{T,1},1}:
 [1.0,1.0]
 [2.0,2.0]

julia> q = [1.0, 1.0] + eps(Float64)
2-element Array{Float64,1}:
 1.0
 1.0

julia> in(q, P)
false

julia> approx_in(q, P)
true

"""
function approx_in(q::T, P::Vector{T}; rtol=1.0e-4, atol=0.0) where T
    for p in P
        if isapprox(q, p; rtol=rtol, atol=atol)
            return true
        end
    end
    return false
end

function get_polygon_clip(xs::Vector{T}, xm::Vector{T}, n::T) where T
    # objective: search does line xm1 - xm2 clip xs
    nm = length(xm)
    ns = length(xs)
    P = T[]

    # 1. test is master point inside slave, if yes, add to clip
    for i=1:nm
        if vertex_inside_polygon(xm[i], xs)
            push!(P, xm[i])
        end
    end

    # 2. test is slave point inside master, if yes, add to clip
    for i=1:ns
        if vertex_inside_polygon(xs[i], xm)
            approx_in(xs[i], P) && continue
            push!(P, xs[i])
        end
    end

    for i=1:nm
        # 2. find possible intersection
        xm1 = xm[i]
        xm2 = xm[mod(i,nm)+1]
        # @info("intersecting line $xm1 -> $xm2")
        for j=1:ns
            xs1 = xs[j]
            xs2 = xs[mod(j,ns)+1]
            # @info("clipping polygon edge $xs1 -> $xs2")
            tnom = dot(cross(xm1-xs1, xm2-xm1), n)
            tdenom = dot(cross(xs2-xs1, xm2-xm1), n)
            isapprox(tdenom, 0) && continue
            t = tnom/tdenom
            (0 <= t <= 1) || continue
            q = xs1 + t*(xs2 - xs1)
            # @info("t=$t, q=$q, q ∈ xm ? $(vertex_inside_polygon(q, xm))")
            if vertex_inside_polygon(q, xm)
                approx_in(q, P) && continue
                push!(P, q)
            end
        end
    end

    return P
end

""" Project some vertex p to surface of element E using Newton's iterations. """
function project_vertex_to_surface(p, x0, n0,
                                   element::Element{E}, x, time;
                                   max_iterations=10, iter_tol=1.0e-6) where E
    basis(xi) = get_basis(element, xi, time)
    function dbasis(xi)
        return get_dbasis(element, xi, time)
    end
    nnodes = length(element)
    mul(a,b) = sum((a[:,i]*b[i]')' for i=1:length(b))

    function f(theta)
        b = [basis(theta[1:2])*collect(x)...;]
        b = b - theta[3]*n0 - p
        return b
    end

    L(theta) = inv3([mul(dbasis(theta[1:2]), x) -n0])
    theta = zeros(3)
    dtheta = zeros(3)
    for i=1:max_iterations
        invA = L(theta)
        b = f(theta)
        dtheta = invA * b
        theta -= dtheta
        if norm(dtheta) < iter_tol
            return theta[1:2], theta[3]
        end
    end
    #=
    @info("failed to project vertex from auxiliary plane back to surface")
    @info("element type: $E")
    @info("element connectivity: $(get_connectivity(element))")
    @info("auxiliary plane: x0 = $x0, n0 = $n0")
    @info("element geometry: $(x.data)")
    @info("vertex to project: $p")
    @info("parameter vector before giving up: $theta")
    @info("increment in parameter vector before giving up: $dtheta")
    @info("norm(dtheta) before giving up: $(norm(dtheta))")
    @info("f([0.0, 0.0, 0.0]) = $(f([0.0, 0.0, 0.0]))")
    @info("L([0.0, 0.0, 0.0]) = $(L([0.0, 0.0, 0.0]))")

    @info("iterations:")
    theta = zeros(3)
    dtheta = zeros(3)
    for i=1:max_iterations
        @info("iter $i, theta = $theta")
        @info("f = $(f(theta))")
        @info("L = $(L(theta))")
        dtheta = L(theta) * f(theta)
        @info("dtheta = $(dtheta)")
        theta -= dtheta
    end
    =#
    throw(error("project_point_to_surface: did not converge in $max_iterations iterations!"))
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

""" Given polygon P and normal direction n, check that polygon vertices are
ordered in counter clock wise direction with respect to surface normal and
sort if necessary. It is assumed that polygon is convex.

Examples
--------
Unit triangle, normal in z-direction:

julia> P = Vector[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]
3-element Array{Array{T,1},1}:
 [0.0,0.0,0.0]
 [0.0,1.0,0.0]
 [1.0,0.0,0.0]

julia> n = [0.0, 0.0, 1.0]
3-element Array{Float64,1}:
 0.0
 0.0
 1.0

julia> check_orientation!(P, n)
3-element Array{Array{T,1},1}:
 [1.0,0.0,0.0]
 [0.0,0.0,0.0]
 [0.0,1.0,0.0]

"""
function check_orientation!(P, n)
    C = mean(P)
    np = length(P)
    s = [dot(n, cross(P[i]-C, P[mod(i+1,np)+1]-C)) for i=1:np]
    all(s .< 0) && return
    # project points to new orthogonal basis Q and sort there
    t1 = (P[1]-C)/norm(P[1]-C)
    t2 = cross(n, t1)
    Q = [n t1 t2]
    sort!(P, lt=(A, B) -> begin
        A_proj = Q'*(A-C)
        B_proj = Q'*(B-C)
        a = atan(A_proj[3], A_proj[2])
        b = atan(B_proj[3], B_proj[2])
        return a > b
    end)
end

function convert_to_linear_element(element::Element{E}) where E
    return element
end

function convert_to_linear_element(element::Element{Tri6})
    new_element = Element(Tri3, element.connectivity[1:3])
    new_element.id = element.id
    new_element.fields = element.fields
    return new_element
end

function split_quadratic_element(element::Element{E}, time::Float64) where E
    return [element]
end

function split_quadratic_element(element::Element{Tri6}, time::Float64)
    element_maps = Vector{Int}[[1,4,6], [4,5,6], [4,2,5], [6,5,3]]
    new_elements = Element[]
    connectivity = get_connectivity(element)
    for elmap in element_maps
        new_element = Element(Tri3, connectivity[elmap])
        X = element("geometry", time)
        update!(new_element, "geometry", time => X[elmap])
        if haskey(element, "displacement")
            u = element("displacement", time)
            update!(new_element, "displacement", time => u[elmap])
        end
        if haskey(element, "normal")
            n = element("normal", time)
            update!(new_element, "normal", time => n[elmap])
        end
        push!(new_elements, new_element)
    end
    return new_elements
end

function split_quadratic_elements(elements::DVTI, time::Float64)
    return DVTI(split_quadratic_elements(elements.data, time))
end

""" Split quadratic surface elements to linear elements.  """
function split_quadratic_elements(elements::Vector, time::Float64)
    new_elements = Element[]
    for element in elements
        for splitted_element in split_quadratic_element(element, time)
            push!(new_elements, splitted_element)
        end
    end
    n1 = length(elements)
    n2 = length(new_elements)
    if n1 != n2
        @info("Splitted $n1 elements to $n2 (linear) sub-elements")
    end
    return new_elements
end

function get_mean_xi(element::Element)
    xi = zeros(2)
    coords = get_reference_coordinates(element)
    for (xi1,xi2) in coords
        xi[1] += xi1
        xi[2] += xi2
    end
    xi /= length(coords)
    return xi
end

""" Assemble linear surface element to problem.

Dual basis is constructed such that partially integrated slave segments are taken into account in a proper way.

Notes
-----
For full integrated slave element, coefficient matrix for Tri3 is
Ae = [3.0 -1.0 -1.0; -1.0 3.0 -1.0; -1.0 -1.0 3.0]

References
----------

[Popp2013] Popp, Alexander, et al. "Improved robustness and consistency of 3D contact algorithms based on a dual mortar approach." Computer Methods in Applied Mechanics and Engineering 264 (2013): 67-80.
"""
function assemble!(problem::Problem{Mortar}, slave_element::Element{E}, time::Real; first_slave_element=false) where E<:Union{Tri3, Quad4}

    props = problem.properties
    field_dim = get_unknown_field_dimension(problem)
    field_name = get_parent_field_name(problem)
    area = 0.0

    slave_element_nodes = get_connectivity(slave_element)
    nsl = length(slave_element)
    X1 = slave_element("geometry", time)
    n1 = slave_element("normal", time)

    # project slave nodes to auxiliary plane (x0, Q)
    xi = get_mean_xi(slave_element)
    N = vec(get_basis(slave_element, xi, time))
    x0 = interpolate(N, X1)
    n0 = interpolate(N, n1)
    S = Vector[project_vertex_to_auxiliary_plane(X1[i], x0, n0) for i=1:nsl]

    master_elements = slave_element("master elements", time)

    if props.dual_basis

        De = zeros(nsl, nsl)
        Me = zeros(nsl, nsl)

        for master_element in master_elements

            master_element_nodes = get_connectivity(master_element)
            nm = length(master_element)
            X2 = master_element("geometry", time)

            if norm(mean(X1) - mean(X2)) > problem.properties.distval
                continue
            end

            # 3.1 project master nodes to auxiliary plane and create polygon clipping
            M = Vector[project_vertex_to_auxiliary_plane(X2[i], x0, n0) for i=1:nm]
            P = get_polygon_clip(S, M, n0)
            length(P) < 3 && continue # no clipping or shared edge (no volume)
            check_orientation!(P, n0)
            N_P = length(P)
            P_area = sum([norm(1/2*cross(P[i]-P[1], P[mod(i,N_P)+1]-P[1])) for i=2:N_P])

            if isapprox(P_area, 0.0)
                @info("Polygon P has zero area: $P_area")
                continue
            end

            # 4. loop integration cells
            C0 = calculate_centroid(P)
            all_cells = get_cells(P, C0)
            for cell in all_cells
                virtual_element = Element(Tri3, Int[])
                update!(virtual_element, "geometry", tuple(cell...))
                for ip in get_integration_points(virtual_element, 3)
                    detJ = virtual_element(ip, time, Val{:detJ})
                    w = ip.weight*detJ
                    x_gauss = virtual_element("geometry", ip, time)
                    xi_s, alpha = project_vertex_to_surface(x_gauss, x0, n0, slave_element, X1, time)
                    N1 = slave_element(xi_s, time)
                    De += w*Matrix(Diagonal(vec(N1)))
                    Me += w*N1'*N1
                end
            end # integration cells done

        end # master elements done

        Ae = De*inv(Me)

        @info("Dual basis coefficient matrix: $Ae")

    else
        Ae = Matrix(1.0I, nsl, nsl)
    end

    for master_element in master_elements

        master_element_nodes = get_connectivity(master_element)
        nm = length(master_element)
        X2 = master_element("geometry", time)

        if norm(mean(X1) - mean(X2)) > problem.properties.distval
            continue
        end

        # 3.1 project master nodes to auxiliary plane and create polygon clipping
        M = Vector[project_vertex_to_auxiliary_plane(X2[i], x0, n0) for i=1:nm]
        P = get_polygon_clip(S, M, n0)
        length(P) < 3 && continue # no clipping or shared edge (no volume)
        check_orientation!(P, n0)
        N_P = length(P)
        P_area = sum([norm(1/2*cross(P[i]-P[1], P[mod(i,N_P)+1]-P[1])) for i=2:N_P])

        if isapprox(P_area, 0.0)
            @info("Polygon P has zero area: $P_area")
            continue
        end

        C0 = calculate_centroid(P)

        De = zeros(nsl, nsl)
        Me = zeros(nsl, nm)
        ge = zeros(field_dim*nsl)

        # 4. loop integration cells
        all_cells = get_cells(P, C0)
        for cell in all_cells
            virtual_element = Element(Tri3, Int[])
            update!(virtual_element, "geometry", tuple(cell...))

            # 5. loop integration point of integration cell
            for ip in get_integration_points(virtual_element, 3)
                detJ = virtual_element(ip, time, Val{:detJ})
                w = ip.weight*detJ

                # project gauss point from auxiliary plane to master and slave element
                x_gauss = virtual_element("geometry", ip, time)

                xi_s, alpha = project_vertex_to_surface(x_gauss, x0, n0, slave_element, X1, time)
                xi_m, alpha = project_vertex_to_surface(x_gauss, x0, n0, master_element, X2, time)

                # add contributions
                N1 = vec(get_basis(slave_element, xi_s, time))
                N2 = vec(get_basis(master_element, xi_m, time))
                Phi = Ae*N1
                # Phi = [3.0-4.0*xi_s[1]-4.0*xi_s[2], 4.0*xi_s[1]-1.0, 4.0*xi_s[2]-1.0]
                De += w*Phi*N1'
                Me += w*Phi*N2'
                if props.adjust && haskey(slave_element, "displacement") && haskey(master_element, "displacement")
                    u1 = slave_element("displacement", time)
                    u2 = master_element("displacement", time)
                    x_s = interpolate(N1, map(+,X1,u1))
                    x_m = interpolate(N2, map(+,X2,u2))
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

    return area
end


""" Assemble quadratic surface element to problem.

In polygon clipping element is divided to linear sub-elements proposed in [Puso2008].

References
----------

[Puso2008] Puso, Michael A., T. A. Laursen, and Jerome Solberg. "A segment-to-segment mortar contact method for quadratic elements and large deformations." Computer Methods in Applied Mechanics and Engineering 197.6 (2008): 555-566.

[Popp1012] Popp, Alexander, et al. "Dual quadratic mortar finite element methods for 3D finite deformation contact." SIAM Journal on Scientific Computing 34.4 (2012): B421-B446.

"""
function assemble!(problem::Problem{Mortar}, slave_element::Element{E}, time::Real; first_slave_element=false) where E<:Union{Tri6}

    props = problem.properties
    field_dim = get_unknown_field_dimension(problem)
    field_name = get_parent_field_name(problem)
    area = 0.0

    Xs = slave_element("geometry", time)

    alp = props.alpha

    if alp != 0.0
        T = [
                1.0 0.0 0.0 0.0 0.0 0.0
                0.0 1.0 0.0 0.0 0.0 0.0
                0.0 0.0 1.0 0.0 0.0 0.0
                alp alp 0.0 1.0-2*alp 0.0 0.0
                0.0 alp alp 0.0 1.0-2*alp 0.0
                alp 0.0 alp 0.0 0.0 1.0-2*alp
            ]
    else
        T = Matrix(1.0I, 6, 6)
    end

    #=
    invT = [
        1.0 0.0 0.0 0.0 0.0 0.0
        0.0 1.0 0.0 0.0 0.0 0.0
        0.0 0.0 1.0 0.0 0.0 0.0
        -alp/(1-2*alp) -alp/(1-2*alp) 0.0 1/(1-2*alp) 0.0 0.0
        0.0 -alp/(1-2*alp) -alp/(1-2*alp) 0.0 1/(1-2*alp) 0.0
        -alp/(1-2*alp) 0.0 -alp/(1-2*alp) 0.0 0.0 1/(1-2*alp)
    ]
    =#

    if props.dual_basis
        # @info("Creating dual basis for element $(slave_element.id)")
        nsl = length(slave_element)
        De = zeros(nsl, nsl)
        Me = zeros(nsl, nsl)

        # split slave element to linear sub-elements and loop
        for sub_slave_element in split_quadratic_element(slave_element, time)

            slave_element_nodes = get_connectivity(sub_slave_element)
            nsl = length(sub_slave_element)
            X1 = sub_slave_element("geometry", time)
            n1 = sub_slave_element("normal", time)

            # create auxiliary plane
            xi = get_mean_xi(sub_slave_element)
            N = vec(get_basis(sub_slave_element, xi, time))
            x0 = interpolate(N, X1)
            n0 = interpolate(N, n1)

            # project slave nodes to auxiliary plane
            S = Vector[project_vertex_to_auxiliary_plane(X1[i], x0, n0) for i=1:nsl]

            # 3. loop all master elements
            master_elements = slave_element("master elements", time)

            for master_element in master_elements

                Xm = master_element("geometry", time)

                if norm(mean(Xs) - mean(Xm)) > problem.properties.distval
                    continue
                end

                # split master element to linear sub-elements and loop
                for sub_master_element in split_quadratic_element(master_element, time)

                    master_element_nodes = get_connectivity(sub_master_element)
                    nm = length(sub_master_element)
                    X2 = sub_master_element("geometry", time)

                    # 3.1 project master nodes to auxiliary plane
                    M = Vector[project_vertex_to_auxiliary_plane(X2[i], x0, n0) for i=1:nm]

                    # create polygon clipping P
                    P = get_polygon_clip(S, M, n0)
                    length(P) < 3 && continue # no clipping or shared edge (no volume)
                    check_orientation!(P, n0)
                    N_P = length(P)
                    P_area = sum([norm(1/2*cross(P[i]-P[1], P[mod(i,N_P)+1]-P[1])) for i=2:N_P])

                    C0 = calculate_centroid(P)

                    # 4. loop integration cells
                    all_cells = get_cells(P, C0)
                    for cell in all_cells
                        virtual_element = Element(Tri3, Int[])
                        update!(virtual_element, "geometry", tuple(cell...))
                        for ip in get_integration_points(virtual_element, 3)
                            x_gauss = virtual_element("geometry", ip, time)
                            xi_s, alpha = project_vertex_to_surface(x_gauss, x0, n0, slave_element, Xs, time)
                            detJ = virtual_element(ip, time, Val{:detJ})
                            w = ip.weight*detJ
                            N1 = vec(slave_element(xi_s, time)*T)
                            De += w*Matrix(Diagonal(N1))
                            Me += w*N1*N1'
                        end

                    end # integration cells done

                end # sub master elements done

            end # master elements done

        end # sub slave elements done

        Ae = De*inv(Me)
        # @info("Dual basis construction finished.")
        # @info("Slave element geometry = $Xs")
        # @info("De = $De")
        # @info("Me = $Me")
        # @info("Dual basis coefficient matrix: $Ae")

    else
        nsl = length(slave_element)
        Ae = Matrix(1.0I, nsl, nsl)
    end

    # split slave element to linear sub-elements and loop
    for sub_slave_element in split_quadratic_element(slave_element, time)

        slave_element_nodes = get_connectivity(sub_slave_element)
        nsl = length(sub_slave_element)
        X1 = sub_slave_element("geometry", time)
        n1 = sub_slave_element("normal", time)

        # create auxiliary plane
        xi = get_mean_xi(sub_slave_element)
        N = vec(get_basis(sub_slave_element, xi, time))
        x0 = interpolate(N, X1)
        n0 = interpolate(N, n1)

        # project slave nodes to auxiliary plane
        S = Vector[project_vertex_to_auxiliary_plane(X1[i], x0, n0) for i=1:nsl]

        # 3. loop all master elements
        master_elements = slave_element("master elements", time)

        for master_element in master_elements

            Xm = master_element("geometry", time)

            if norm(mean(Xs) - mean(Xm)) > problem.properties.distval
                continue
            end

            # split master element to linear sub-elements and loop
            for sub_master_element in split_quadratic_element(master_element, time)

                master_element_nodes = get_connectivity(sub_master_element)
                nm = length(sub_master_element)
                X2 = sub_master_element("geometry", time)

                # 3.1 project master nodes to auxiliary plane
                M = Vector[project_vertex_to_auxiliary_plane(X2[i], x0, n0) for i=1:nm]

                # create polygon clipping P
                P = get_polygon_clip(S, M, n0)
                length(P) < 3 && continue # no clipping or shared edge (no volume)
                check_orientation!(P, n0)
                N_P = length(P)
                P_area = sum([norm(1/2*cross(P[i]-P[1], P[mod(i,N_P)+1]-P[1])) for i=2:N_P])

                if isapprox(P_area, 0.0)
                    @warn("Polygon P has zero area: $P_area")
                    continue
                end

                C0 = calculate_centroid(P)

                # while our polygon clipping algorithm is working in linear sub elements
                # contributions is calculated using quadratic shape functions
                De = zeros(length(slave_element), length(slave_element))
                Me = zeros(length(slave_element), length(master_element))
                ge = zeros(field_dim*length(slave_element))

                # 4. loop integration cells
                all_cells = get_cells(P, C0)
                for cell in all_cells
                    virtual_element = Element(Tri3, Int[])
                    update!(virtual_element, "geometry", tuple(cell...))

                    # 5. loop integration point of integration cell
                    for ip in get_integration_points(virtual_element, 3)

                        x_gauss = virtual_element("geometry", ip, time)
                        xi_s, alpha = project_vertex_to_surface(x_gauss, x0, n0, slave_element, Xs, time)
                        xi_m, alpha = project_vertex_to_surface(x_gauss, x0, n0, master_element, Xm, time)

                        # add contributions
                        N1 = vec(slave_element(xi_s, time)*T)
                        N2 = vec(master_element(xi_m, time))
                        Phi = Ae*N1

                        detJ = virtual_element(ip, time, Val{:detJ})
                        w = ip.weight*detJ

                        De += w*Phi*N1'
                        Me += w*Phi*N2'
                        if props.adjust && haskey(slave_element, "displacement") && haskey(master_element, "displacement")
                            u1 = slave_element("displacement", time)
                            u2 = master_element("displacement", time)
                            xs = interpolate(N1, map(+,Xs,u1))
                            xm = interpolate(N2, map(+,Xm,u2))
                            ge += w*vec((xm-xs)*Phi')
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

            end # sub aster elements done

        end # master elements done

    end # sub slave elements done

    return area
end


function assemble!(problem::Problem{Mortar}, time::Real, ::Type{Val{2}}, ::Type{Val{false}})

    props = problem.properties
    field_dim = get_unknown_field_dimension(problem)
    field_name = get_parent_field_name(problem)
    slave_elements = get_slave_elements(problem)
    area = 0.0

    #=
    if props.split_quadratic_slave_elements
        if !props.linear_surface_elements
            @warn("Mortar3D: split_quadratic_surfaces = true and linear_surface_elements = false maybe have unexpected behavior")
        end
        slave_elements = split_quadratic_elements(slave_elements, time)
    end
    =#

    # 1. calculate nodal normals and tangents for slave element nodes j ∈ S
    normals = calculate_normals(slave_elements, time, Val{2};
                                rotate_normals=props.rotate_normals)

    update!(slave_elements, "normal", time => normals)

    # 2. loop all slave elements
    first_slave_element = true

    for slave_element in slave_elements

        area += assemble!(problem, slave_element, time; first_slave_element=first_slave_element)
        first_slave_element = false

    end # slave elements done, contact virtual work ready

    C1 = sparse(problem.assembly.C1)
    C2 = sparse(problem.assembly.C2)

    maxdim = maximum(size(C1))
    if problem.properties.alpha != 0.0
        alp = problem.properties.alpha
        Te = [
                1.0 0.0 0.0 0.0 0.0 0.0
                0.0 1.0 0.0 0.0 0.0 0.0
                0.0 0.0 1.0 0.0 0.0 0.0
                alp alp 0.0 1.0-2*alp 0.0 0.0
                0.0 alp alp 0.0 1.0-2*alp 0.0
                alp 0.0 alp 0.0 0.0 1.0-2*alp
            ]
        invTe = [
            1.0 0.0 0.0 0.0 0.0 0.0
            0.0 1.0 0.0 0.0 0.0 0.0
            0.0 0.0 1.0 0.0 0.0 0.0
            -alp/(1-2*alp) -alp/(1-2*alp) 0.0 1/(1-2*alp) 0.0 0.0
            0.0 -alp/(1-2*alp) -alp/(1-2*alp) 0.0 1/(1-2*alp) 0.0
            -alp/(1-2*alp) 0.0 -alp/(1-2*alp) 0.0 0.0 1/(1-2*alp)
        ]
        # construct global transformation matrices T and invT
        T = SparseMatrixCOO()
        invT = SparseMatrixCOO()
        for element in slave_elements
            dofs = get_gdofs(problem, element)
            for i=1:field_dim
                ldofs = dofs[i:field_dim:end]
                add!(T, ldofs, ldofs, Te)
                add!(invT, ldofs, ldofs, invTe)
            end
        end
        T = sparse(T, maxdim, maxdim, (a, b) -> b)
        invT = sparse(invT, maxdim, maxdim, (a, b) -> b)
        # fill diagonal
        d = ones(size(T, 1))
        d[get_nonzero_rows(T)] .= 0.0
        T += sparse(Diagonal(d))
        invT += sparse(Diagonal(d))
        #invT2 = sparse(inv(full(T)))
        #@info("invT == invT2? ", invT == invT2)
        #maxabsdiff = maximum(abs(invT - invT2))
        #@info("max diff = $maxabsdiff")
        C1 = C1*invT
        C2 = C2*invT
    end

    tol = problem.properties.drop_tolerance
    SparseArrays.droptol!(C1, tol)
    SparseArrays.droptol!(C2, tol)

    problem.assembly.C1 = C1
    problem.assembly.C2 = C2

end
