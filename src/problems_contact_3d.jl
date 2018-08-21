# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

const ContactElements3D = Union{Tri3,Tri6,Quad4,Quad8,Quad9}

function create_orthogonal_basis(n)
    I = [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
    k = argmax([norm(cross(n,I[:,k])) for k in 1:3])
    t1 = cross(n, I[:,k])/norm(cross(n, I[:,k]))
    t2 = cross(n, t1)
    return t1, t2
end

""" Create rotation matrix Q for element nodes rotating quantities to nt coordinaet system. """
function create_rotation_matrix(element::Element{Tri3}, time::Float64)
    n = element("normal", time)
    t11, t21 = create_orthogonal_basis(n[1])
    t12, t22 = create_orthogonal_basis(n[2])
    t13, t23 = create_orthogonal_basis(n[3])
    Q1_ = [n[1] t11 t21]
    Q2_ = [n[2] t12 t22]
    Q3_ = [n[3] t13 t23]
    Z = zeros(3, 3)
    Q = [
        Q1_ Z Z
        Z Q2_ Z
        Z Z Q3_]
    return Q
end

function create_rotation_matrix(element::Element{Quad4}, time::Float64)
    n = element("normal", time)
    t11, t21 = create_orthogonal_basis(n[1])
    t12, t22 = create_orthogonal_basis(n[2])
    t13, t23 = create_orthogonal_basis(n[3])
    t14, t24 = create_orthogonal_basis(n[4])
    Q1_ = [n[1] t11 t21]
    Q2_ = [n[2] t12 t22]
    Q3_ = [n[3] t13 t23]
    Q4_ = [n[4] t14 t24]
    Z = zeros(3, 3)
    Q = [
        Q1_ Z Z Z
        Z Q2_ Z Z
        Z Z Q3_ Z
        Z Z Z Q4_]
    return Q
end

function create_rotation_matrix(element::Element{Tri6}, time::Float64)
    n = element("normal", time)
    t11, t21 = create_orthogonal_basis(n[1])
    t12, t22 = create_orthogonal_basis(n[2])
    t13, t23 = create_orthogonal_basis(n[3])
    t14, t24 = create_orthogonal_basis(n[4])
    t15, t25 = create_orthogonal_basis(n[5])
    t16, t26 = create_orthogonal_basis(n[6])
    Q1_ = [n[1] t11 t21]
    Q2_ = [n[2] t12 t22]
    Q3_ = [n[3] t13 t23]
    Q4_ = [n[4] t14 t24]
    Q5_ = [n[5] t15 t25]
    Q6_ = [n[6] t16 t26]
    Z = zeros(3, 3)
    Q = [
        Q1_ Z Z Z Z Z
        Z Q2_ Z Z Z Z
        Z Z Q3_ Z Z Z
        Z Z Z Q4_ Z Z
        Z Z Z Z Q5_ Z
        Z Z Z Z Z Q6_]
    return Q
end

""" Create a contact segmentation between one slave element and list of master elements.

Returns
-------

Vector with tuples: (master_element, polygon_clip_vertices, polygon_clip_centroid, polygon_clip_area)
"""
function create_contact_segmentation(slave_element, master_elements, x0, n0, time::Float64; deformed=false)
    result = []
    x1 = slave_element("geometry", time)
    if deformed
        x1 = map(+, x1, slave_element("displacement", time))
    end
    S = Vector[project_vertex_to_auxiliary_plane(p, x0, n0) for p in x1]
    for master_element in master_elements
        x2 = master_element("geometry", time)
        if deformed
            x2 = map(+, x2, master_element("displacement", time))
        end
        M = Vector[project_vertex_to_auxiliary_plane(p, x0, n0) for p in x2]
        P = get_polygon_clip(S, M, n0)
        length(P) < 3 && continue # no clipping or shared edge (no volume)
        check_orientation!(P, n0)
        N_P = length(P)
        P_area = sum([norm(1/2*cross(P[i]-P[1], P[mod(i,N_P)+1]-P[1])) for i=2:N_P])
        if isapprox(P_area, 0.0)
            error("Polygon P has zero area")
        end
        C0 = calculate_centroid(P)
        push!(result, (master_element, P, C0, P_area))
    end
    return result
end

function assemble!(problem::Problem{Contact}, slave_element::Element{Tri3}, time::Float64)

    props = problem.properties
    field_dim = get_unknown_field_dimension(problem)

    nsl = length(slave_element)
    X1 = slave_element("geometry", time)
    u1 = slave_element("displacement", time)
    x1 = map(+, X1, u1)
    n1 = slave_element("normal", time)
    la = slave_element("lambda", time)

    Q3 = create_rotation_matrix(slave_element, time)

    # project slave nodes to auxiliary plane (x0, Q)
    xi = get_mean_xi(slave_element)
    N = vec(get_basis(slave_element, xi, time))
    x0 = interpolate(N, X1)
    n0 = interpolate(N, n1)

    # create contact segmentation
    segmentation = create_contact_segmentation(slave_element, slave_element("master elements", time), x0, n0, time)

    if length(segmentation) == 0 # no overlapping surface in slave and maters
        return
    end

    Ae = Matrix{Float64}(I, nsl, nsl)

    if problem.properties.dual_basis # construct dual basis

        De = zeros(nsl, nsl)
        Me = zeros(nsl, nsl)

        # loop all polygons
        for (master_element, P, C0, P_area) in segmentation

            # loop integration cells
            for cell in get_cells(P, C0)
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
                end # integration points done

            end # integration cells done

        end # master elements done

        Ae = De*inv(Me)

    end

    # loop all polygons
    for (master_element, P, C0, P_area) in segmentation

        nm = length(master_element)
        X2 = master_element("geometry", time)
        u2 = master_element("displacement", time)
        x2 = map(+, X2, u2)

        De = zeros(nsl, nsl)
        Me = zeros(nsl, nm)
        ce = zeros(field_dim*nsl)
        ge = zeros(field_dim*nsl)

        # loop integration cells
        for cell in get_cells(P, C0)
            virtual_element = Element(Tri3, Int[])
            update!(virtual_element, "geometry", tuple(cell...))
            # loop integration point of integration cell
            for ip in get_integration_points(virtual_element, 3)

                # project gauss point from auxiliary plane to master and slave element
                x_gauss = virtual_element("geometry", ip, time)
                xi_s, alpha = project_vertex_to_surface(x_gauss, x0, n0, slave_element, X1, time)
                xi_m, alpha = project_vertex_to_surface(x_gauss, x0, n0, master_element, X2, time)

                detJ = virtual_element(ip, time, Val{:detJ})
                w = ip.weight*detJ

                # add contributions
                N1 = vec(get_basis(slave_element, xi_s, time))
                N2 = vec(get_basis(master_element, xi_m, time))
                Phi = Ae*N1
                De += w*Phi*N1'
                Me += w*Phi*N2'

                x_s = interpolate(N1, map(+,X1,u1))
                x_m = interpolate(N2, map(+,X2,u2))
                ge += w*vec((x_m-x_s)*Phi')

            end # integration points done

        end # integration cells done

        # add contribution to contact virtual work
        sdofs = get_gdofs(problem, slave_element)
        mdofs = get_gdofs(problem, master_element)
        nsldofs = length(sdofs)
        nmdofs = length(mdofs)
        D3 = zeros(nsldofs, nsldofs)
        M3 = zeros(nsldofs, nmdofs)
        for i=1:field_dim
            D3[i:field_dim:end, i:field_dim:end] += De
            M3[i:field_dim:end, i:field_dim:end] += Me
        end

        add!(problem.assembly.C1, sdofs, sdofs, D3)
        add!(problem.assembly.C1, sdofs, mdofs, -M3)
        add!(problem.assembly.C2, sdofs, sdofs, Q3'*D3)
        add!(problem.assembly.C2, sdofs, mdofs, -Q3'*M3)
        add!(problem.assembly.g, sdofs, Q3'*ge)

    end # master elements done

end


""" Assemble quadratic surface element to contact problem. """
function assemble!(problem::Problem{Contact}, slave_element::Element{Tri6}, time::Float64)

    props = problem.properties
    field_dim = get_unknown_field_dimension(problem)

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
        T = Matrix(1.0*I, 6, 6)
    end

    nsl = length(slave_element)
    Xs = slave_element("geometry", time)
    n1 = slave_element("normal", time)

    Q3 = create_rotation_matrix(slave_element, time)

    Ae = Matrix(1.0*I, nsl, nsl)

    if problem.properties.dual_basis # construct dual basis

        nsl = length(slave_element)
        De = zeros(nsl, nsl)
        Me = zeros(nsl, nsl)

        for sub_slave_element in split_quadratic_element(slave_element, time)

            slave_element_nodes = get_connectivity(sub_slave_element)
            nsl = length(sub_slave_element)

            X1 = sub_slave_element("geometry", time)
            #u1 = sub_slave_element("displacement", time)
            #x1 = X1 + u1
            n1 = sub_slave_element("normal", time)
            #la = sub_slave_element("lambda", time)

            # create auxiliary plane
            xi = get_mean_xi(sub_slave_element)
            N = vec(get_basis(sub_slave_element, xi, time))
            x0 = interpolate(N, X1)
            n0 = interpolate(N, n1)

            # project slave nodes to auxiliary plane
            S = Vector[project_vertex_to_auxiliary_plane(p, x0, n0) for p in X1]

            # 3. loop all master elements
            for master_element in slave_element("master elements", time)

                Xm = master_element("geometry", time)

                if norm(mean(Xs) - mean(Xm)) > problem.properties.distval
                    continue
                end

                # split master element to linear sub-elements and loop
                for sub_master_element in split_quadratic_element(master_element, time)

                    master_element_nodes = get_connectivity(sub_master_element)
                    nm = length(sub_master_element)
                    X2 = sub_master_element("geometry", time)
                    #u2 = sub_master_element("displacement", time)
                    #x2 = X2 + u2

                    # 3.1 project master nodes to auxiliary plane and create polygon clipping
                    M = Vector[project_vertex_to_auxiliary_plane(p, x0, n0) for p in X2]
                    P = get_polygon_clip(S, M, n0)
                    length(P) < 3 && continue # no clipping or shared edge (no volume)
                    check_orientation!(P, n0)

                    N_P = length(P)
                    P_area = sum([norm(1/2*cross(P[i]-P[1], P[mod(i,N_P)+1]-P[1])) for i=2:N_P])
                    if isapprox(P_area, 0.0)
                        error("Polygon P has zero area")
                    end

                    C0 = calculate_centroid(P)

                    # 4. loop integration cells
                    for cell in get_cells(P, C0)
                        virtual_element = Element(Tri3, Int[])
                        update!(virtual_element, "geometry", tuple(cell...))
                        for ip in get_integration_points(virtual_element, 3)
                            detJ = virtual_element(ip, time, Val{:detJ})
                            w = ip.weight*detJ
                            x_gauss = virtual_element("geometry", ip, time)
                            xi_s, alpha = project_vertex_to_surface(x_gauss, x0, n0, slave_element, Xs, time)
                            N1 = vec(slave_element(xi_s, time)*T)
                            De += w*Matrix(Diagonal(N1))
                            Me += w*N1*N1'
                        end # integration points done

                    end # integration cells done

                end # sub master elements done

            end # master elements done

        end # sub slave elements done

        Ae = De*inv(Me)

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
        S = Vector[project_vertex_to_auxiliary_plane(p, x0, n0) for p in X1]

        # 3. loop all master elements
        for master_element in slave_element("master elements", time)

            Xm = master_element("geometry", time)

            if norm(mean(Xs) - mean(Xm)) > problem.properties.distval
                continue
            end

            # split master element to linear sub-elements and loop
            for sub_master_element in split_quadratic_element(master_element, time)

                master_element_nodes = get_connectivity(sub_master_element)
                nm = length(master_element)
                X2 = sub_master_element("geometry", time)
                #u2 = master_element("displacement", time)
                #x2 = X2 + u2

                # 3.1 project master nodes to auxiliary plane and create polygon clipping
                M = Vector[project_vertex_to_auxiliary_plane(p, x0, n0) for p in X2]
                P = get_polygon_clip(S, M, n0)
                length(P) < 3 && continue # no clipping or shared edge (no volume)
                check_orientation!(P, n0)

                N_P = length(P)
                P_area = sum([norm(1/2*cross(P[i]-P[1], P[mod(i,N_P)+1]-P[1])) for i=2:N_P])
                if isapprox(P_area, 0.0)
                    error("Polygon P has zero area")
                end

                C0 = calculate_centroid(P)

                # integration is done in quadratic elements
                nsl = length(slave_element)
                nm = length(master_element)
                De = zeros(nsl, nsl)
                Me = zeros(nsl, nm)
                ge = zeros(field_dim*nsl)

                # 4. loop integration cells
                for cell in get_cells(P, C0)
                    virtual_element = Element(Tri3, Int[])
                    update!(virtual_element, "geometry", tuple(cell...))

                    # 5. loop integration point of integration cell
                    for ip in get_integration_points(virtual_element, 3)

                        # project gauss point from auxiliary plane to master and slave element
                        x_gauss = virtual_element("geometry", ip, time)
                        xi_s, alpha = project_vertex_to_surface(x_gauss, x0, n0, slave_element, Xs, time)
                        xi_m, alpha = project_vertex_to_surface(x_gauss, x0, n0, master_element, Xm, time)

                        detJ = virtual_element(ip, time, Val{:detJ})
                        w = ip.weight*detJ

                        # add contributions
                        N1 = vec(get_basis(slave_element, xi_s, time)*T)
                        N2 = vec(get_basis(master_element, xi_m, time))
                        Phi = Ae*N1

                        De += w*Phi*N1'
                        Me += w*Phi*N2'

                        us = slave_element("displacement", time)
                        um = master_element("displacement", time)
                        xs = interpolate(N1, map(+,Xs,us))
                        xm = interpolate(N2, map(+,Xs,um))
                        ge += w*vec((xm-xs)*Phi')

                    end # integration points done

                end # integration cells done

                # 6. add contribution to contact virtual work
                sdofs = get_gdofs(problem, slave_element)
                mdofs = get_gdofs(problem, master_element)
                nsldofs = length(sdofs)
                nmdofs = length(mdofs)
                D3 = zeros(nsldofs, nsldofs)
                M3 = zeros(nsldofs, nmdofs)
                for i=1:field_dim
                    D3[i:field_dim:end, i:field_dim:end] += De
                    M3[i:field_dim:end, i:field_dim:end] += Me
                end

                add!(problem.assembly.C1, sdofs, sdofs, D3)
                add!(problem.assembly.C1, sdofs, mdofs, -M3)
                add!(problem.assembly.C2, sdofs, sdofs, Q3'*D3)
                add!(problem.assembly.C2, sdofs, mdofs, -Q3'*M3)
                add!(problem.assembly.g, sdofs, Q3'*ge)

            end # sub master elements done

        end # master elements done

    end # sub slave elements done

end


"""
Frictionless 3d small sliding contact.

problem
time
dimension
finite_sliding
friction
use_forwarddiff
"""
function assemble!(problem::Problem{Contact}, time::Float64, ::Type{Val{2}}, ::Type{Val{false}}, ::Type{Val{false}}, ::Type{Val{false}})

    props = problem.properties
    field_dim = get_unknown_field_dimension(problem)
    field_name = get_parent_field_name(problem)
    slave_elements = get_slave_elements(problem)

    # 1. calculate nodal normals and tangents for slave element nodes j ∈ S
    normals = calculate_normals(slave_elements, time, Val{2};
                                rotate_normals=props.rotate_normals)
    update!(slave_elements, "normal", time => normals)

    # 2. loop all slave elements
    for slave_element in slave_elements
        assemble!(problem, slave_element, time)
    end # slave elements done, contact virtual work ready

    S = sort(collect(keys(normals))) # slave element nodes
    weighted_gap = Dict{Int64, Vector{Float64}}()
    contact_pressure = Dict{Int64, Vector{Float64}}()
    complementarity_condition = Dict{Int64, Vector{Float64}}()
    is_active = Dict{Int64, Int}()
    is_inactive = Dict{Int64, Int}()
    is_slip = Dict{Int64, Int}()
    is_stick = Dict{Int64, Int}()

    la = problem.assembly.la

    # FIXME: for matrix operations, we need to know the dimensions of the
    # final matrices
    ndofs = 0
    ndofs = max(ndofs, size(problem.assembly.K, 2))
    ndofs = max(ndofs, size(problem.assembly.C1, 2))
    ndofs = max(ndofs, size(problem.assembly.C2, 2))
    ndofs = max(ndofs, size(problem.assembly.D, 2))
    ndofs = max(ndofs, size(problem.assembly.g, 2))
    ndofs = max(ndofs, size(problem.assembly.c, 2))

    C1 = sparse(problem.assembly.C1, ndofs, ndofs)
    C2 = sparse(problem.assembly.C2, ndofs, ndofs)
    D = sparse(problem.assembly.D, ndofs, ndofs)
    g = Vector(problem.assembly.g, ndofs)
    c = Vector(problem.assembly.c, ndofs)

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

    for j in S
        dofs = [3*(j-1)+1, 3*(j-1)+2, 3*(j-1)+3]
        weighted_gap[j] = g[dofs]
    end

    state = problem.properties.contact_state_in_first_iteration
    if problem.properties.iteration == 1
        @info("First contact iteration, initial contact state = $state")

        if state == :AUTO
            avg_gap = mean([weighted_gap[j][1] for j in S])
            std_gap = std([weighted_gap[j][1] for j in S])
            if (avg_gap < 1.0e-12) && (std_gap < 1.0e-12)
                state = :ACTIVE
            else
                state = :UNKNOWN
            end
            @info("Average weighted gap = $avg_gap, std gap = $std_gap, automatically determined contact state = $state")
        end

    end

    # active / inactive node detection
    for j in S
        dofs = [3*(j-1)+1, 3*(j-1)+2, 3*(j-1)+3]
        weighted_gap[j] = g[dofs]
        if length(la) != 0
            normal = normals[j]
            tangent1, tangent2 = create_orthogonal_basis(normal)
            p = dot(normal, la[dofs])
            t1 = dot(tangent1, la[dofs])
            t2 = dot(tangent2, la[dofs])
            contact_pressure[j] = [p, t1, t2]
        else
            contact_pressure[j] = [0.0, 0.0, 0.0]
        end
        complementarity_condition[j] = contact_pressure[j] - weighted_gap[j]

        if complementarity_condition[j][1] > 0.0
            is_inactive[j] = 0
            is_active[j] = 1
            is_slip[j] = 1
            is_stick[j] = 0
        else
            is_inactive[j] = 1
            is_active[j] = 0
            is_slip[j] = 0
            is_stick[j] = 0
        end
    end

    if (problem.properties.iteration == 1) && (state == :ACTIVE)
        for j in S
            is_inactive[j] = 0
            is_active[j] = 1
            is_slip[j] = 1
            is_stick[j] = 0
        end
    end

    if (problem.properties.iteration == 1) && (state == :INACTIVE)
        for j in S
            is_inactive[j] = 1
            is_active[j] = 0
            is_slip[j] = 0
            is_stick[j] = 0
        end
    end

    @info("# | active | stick | slip | gap | pres | comp")
    for j in S
        str1 = "$j | $(is_active[j]) | $(is_stick[j]) | $(is_slip[j]) | "
        str2 = "$(round(weighted_gap[j][1]; digits=3)) | $(round(contact_pressure[j][1]; digits=3)) | $(round(complementarity_condition[j][1]; digits=3))"
        @info(str1 * str2)
    end


    for j in S
        dofs = [3*(j-1)+1, 3*(j-1)+2, 3*(j-1)+3]
        tdofs = [3*(j-1)+2, 3*(j-1)+3]
        if is_inactive[j] == 1
            # remove inactive nodes from assembly
            C1[dofs,:] .= 0.0
            C2[dofs,:] .= 0.0
            D[dofs,:] .= 0.0
            g[dofs,:] .= 0.0
        elseif (is_active[j] == 1) && (is_slip[j] == 1)
            # constitutive modelling in tangent direction, frictionless contact
            C2[tdofs,:] .= 0.0
            g[tdofs] .= 0.0
            normal = normals[j]
            tangent1, tangent2 = create_orthogonal_basis(normal)
            D[tdofs[1], dofs] .= tangent1
            D[tdofs[2], dofs] .= tangent2
        end
    end

    problem.assembly.C1 = C1
    problem.assembly.C2 = C2
    problem.assembly.D = D
    problem.assembly.g = g

end

function postprocess!(problem::Problem{Contact}, time::Float64, ::Type{Val{Symbol("contact pressure")}})
    n = problem("normal", time)
    la = problem("lambda", time)
    node_ids = keys(n)
    cp = Dict(nid => dot(n[nid], la[nid]) for nid in node_ids)
    # FIXME: have to define zero contact pressure & lambda to master elements
    # elements because interface.elements = [slave_elements; master_elements]
    for nid in keys(la)
        if !haskey(cp, nid)
            cp[nid] = 0.0
        end
    end
    update!(problem, "contact pressure", time => cp)
end
