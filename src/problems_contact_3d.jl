# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

typealias ContactElements3D Union{Tri3, Tri6, Quad4, Quad8, Quad9}

function create_orthogonal_basis(n)
    I = eye(3)
    k = indmax([norm(cross(n,I[:,k])) for k in 1:3])
    t1 = cross(n, I[:,k])/norm(cross(n, I[:,k]))
    t2 = cross(n, t1)
    return t1, t2
end

"""
Frictionless 2d small sliding contact.

problem
time
dimension
finite_sliding
friction
use_forwarddiff
"""
function assemble!(problem::Problem{Contact}, time::Float64,
                   ::Type{Val{2}}, ::Type{Val{false}},
                   ::Type{Val{false}}, ::Type{Val{false}}; debug=true)

    props = problem.properties
    field_dim = get_unknown_field_dimension(problem)
    field_name = get_parent_field_name(problem)
    slave_elements = get_slave_elements(problem)

    # 1. calculate nodal normals and tangents for slave element nodes j ∈ S
    normals = calculate_normals(slave_elements, time, Val{2};
                                rotate_normals=props.rotate_normals)
    update!(slave_elements, "normal", normals)

    # 2. loop all slave elements
    for (slave_num, slave_element) in enumerate(slave_elements)

        nsl = length(slave_element)
        X1 = slave_element("geometry", time)
        u1 = slave_element("displacement", time)
        la = slave_element("reaction force", time)
        n1 = slave_element("normal", time)
        t11, t21 = create_orthogonal_basis(n1[1])
        t12, t22 = create_orthogonal_basis(n1[2])
        t13, t23 = create_orthogonal_basis(n1[3])
        Q1_ = [n1[1] t11 t21]
        Q2_ = [n1[2] t12 t22]
        Q3_ = [n1[3] t13 t23]
        Z = zeros(3, 3)
        Q3 = [Q1_ Z Z; Z Q2_ Z; Z Z Q3_]
        contact_area = 0.0
        contact_error = 0.0

        element_area = 0.0
        for ip in get_integration_points(slave_element)
            detJ = slave_element(ip, time, Val{:detJ})
            w = ip.weight*detJ
            element_area += w
        end
        if "element area" in props.store_fields
            update!(slave_element, "element area", time => element_area)
        end

        if slave_num == 1
            info("First slave element area = $element_area")
            info("NT basis of first slave element")
            dump(Q3)
        end

        # project slave nodes to auxiliary plane (x0, Q)
        #xi = get_reference_element_midpoint(slave_element)
        xi = [1/3, 1/3]
        N = vec(get_basis(slave_element, xi, time))
        x0 = N*X1
        n0 = N*n1
        S = Vector[project_vertex_to_auxiliary_plane(p, x0, n0) for p in X1]

        # 3. loop all master elements
        for master_element in slave_element("master elements", time)

            nm = length(master_element)
            X2 = master_element("geometry", time)
            u2 = master_element("displacement", time)
            x2 = X2 + u2

            #=
            norm(mean(X1) - X2[1]) / norm(X1[2] - X1[1]) < props.distval || continue
            norm(mean(X1) - X2[2]) / norm(X1[2] - X1[1]) < props.distval || continue
            norm(mean(X1) - X2[3]) / norm(X1[2] - X1[1]) < props.distval || continue
            =#

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

                # 5. loop integration point of integration cell
                for ip in get_integration_points(virtual_element, 3)

                    # project gauss point from auxiliary plane to master and slave element
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

                    detJ = virtual_element(ip, time, Val{:detJ})
                    w = ip.weight*detJ
                    # add contributions
                    N1 = vec(get_basis(slave_element, xi_s, time))
                    N2 = vec(get_basis(master_element, xi_m, time))
                    De += w*N1*N1'
                    Me += w*N1*N2'
                    
                    x_s = N1*(X1+u1)
                    x_m = N2*(X2+u2)
                    ge += w*vec((x_m-x_s)*N1')
                    contact_area += w
                    n_s = N1*n1
                    contact_error += 1/2*w*dot(n_s, x_s-x_m)^2
                end # integration points done

            end # integration cells done

            # 6. add contribution to contact virtual work
            sdofs = get_gdofs(problem, slave_element)
            mdofs = get_gdofs(problem, master_element)
            nsldofs = length(sdofs)
            nmdofs = length(mdofs)
            D3 = zeros(nsldofs, nsldofs)
            M3 = zeros(nmdofs, nmdofs)
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

        if "contact area" in props.store_fields
            update!(slave_element, "contact area", time => contact_area)
        end

    end # slave elements done, contact virtual work ready

    S = sort(collect(keys(normals))) # slave element nodes
    weighted_gap = Dict{Int64, Vector{Float64}}()
    contact_pressure = Dict{Int64, Vector{Float64}}()
    complementarity_condition = Dict{Int64, Vector{Float64}}()
    is_active = Dict{Int64, Int}()
    is_inactive = Dict{Int64, Int}()
    is_slip = Dict{Int64, Int}()
    is_stick = Dict{Int64, Int}()

    g = full(problem.assembly.g)
    la = problem.assembly.la

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
        if complementarity_condition[j][1] < 0
            is_inactive[j] = 1
            is_active[j] = 0
            is_slip[j] = 0
            is_stick[j] = 0
        else
            is_inactive[j] = 0
            is_active[j] = 1
            is_slip[j] = 1
            is_stick[j] = 0
        end
    end

    if "weighted gap" in props.store_fields
        update!(slave_elements, "weighted gap", time => weighted_gap)
    end
    if "contact pressure" in props.store_fields
        update!(slave_elements, "contact pressure", time => contact_pressure)
    end
    if "complementarity condition" in props.store_fields
        update!(slave_elements, "complementarity condition", time => complementarity_condition)
    end
    if "active nodes" in props.store_fields
        update!(slave_elements, "active nodes", time => is_active)
    end
    if "inactive nodes" in props.store_fields
        update!(slave_elements, "inactive nodes", time => is_inactive)
    end
    if "stick nodes" in props.store_fields
        update!(slave_elements, "stick nodes", time => is_stick)
    end
    if "slip nodes" in props.store_fields
        update!(slave_elements, "slip nodes", time => is_slip)
    end

    info("# | active | inactive | stick | slip | gap | pres | comp")
    for j in S
        str1 = "$j | $(is_active[j]) | $(is_inactive[j]) | $(is_stick[j]) | $(is_slip[j]) | "
        str2 = "$(round(weighted_gap[j], 3)) | $(round(contact_pressure[j], 3)) | $(round(complementarity_condition[j], 3))"
        info(str1 * str2)
    end

    # solve variational inequality
    
    C1 = sparse(problem.assembly.C1)
    ndofs = size(C1, 1)
    C2 = sparse(problem.assembly.C2)
    D = spzeros(ndofs, ndofs)

    # constitutive modelling in tangent direction, frictionless contact
    for j in S
        dofs = [3*(j-1)+1, 3*(j-1)+2, 3*(j-1)+3]
        tdofs = dofs[[2,3]]
        if (is_active[j] == 1) && (is_slip[j] == 1)
            info("$j is in active/slip, removing tangential constraints $tdofs")
            C2[tdofs,:] = 0.0
            g[tdofs] = 0.0
            normal = normals[j]
            tangent1, tangent2 = create_orthogonal_basis(normal)
            D[tdofs[1], dofs] = tangent1
            D[tdofs[2], dofs] = tangent2
        end
    end

    # remove inactive nodes from assembly
    for j in S
        dofs = [3*(j-1)+1, 3*(j-1)+2, 3*(j-1)+3]
        if is_inactive[j] == 1
            info("$j is inactive, removing dofs $dofs")
            C1[dofs,:] = 0.0
            C2[dofs,:] = 0.0
            D[dofs,:] = 0.0
            g[dofs,:] = 0.0
        end
    end

    problem.assembly.C1 = C1
    problem.assembly.C2 = C2
    problem.assembly.D = D
    problem.assembly.g = g

end

