# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

typealias MortarElements2D Union{Seg2, Seg3}

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
    cond(el) = haskey(el, "master elements") || haskey(el, "potential master elements")
    return filter(cond, get_elements(problem))
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
        #X1 = element("geometry", time)
        #dN = get_dbasis(element, [0.0], time)
        #tangent = vec(sum([kron(dN[:,i], X1[i]') for i=1:length(X1)]))
        tangent = vec(element([0.0], time, Val{:Jacobian}))
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

function assemble!(problem::Problem{Mortar}, time::Float64, ::Type{Val{1}}, ::Type{Val{false}})

    props = problem.properties
    field_dim = get_unknown_field_dimension(problem)
    field_name = get_parent_field_name(problem)
    slave_elements = get_slave_elements(problem)

    # 1. calculate nodal normals and tangents for slave element nodes j ∈ S
    normals, tangents = calculate_normals(slave_elements, time, Val{1};
                                          rotate_normals=props.rotate_normals)
    update!(slave_elements, "normal", time => normals)
    update!(slave_elements, "tangent", time => tangents)

    # 2. loop all slave elements
    for slave_element in slave_elements

        nsl = length(slave_element)
        X1 = slave_element("geometry", time)
        n1 = slave_element("normal", time)

        # 3. loop all master elements
        for master_element in slave_element("master elements", time)

            nm = length(master_element)
            X2 = master_element("geometry", time)

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
                    u1 = slave_element("displacement", time)
                    u2 = master_element("displacement", time)
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
