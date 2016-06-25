# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

type Contact <: BoundaryProblem
    dimension :: Int
    rotate_normals :: Bool
    finite_sliding :: Bool
    friction :: Bool
    dual_basis :: Bool
    use_forwarddiff :: Bool
    minimum_active_set_size :: Int
end

function Contact()
    return Contact(-1, false, false, false, true, false, 0)
end

function get_unknown_field_name(problem::Problem{Contact})
    return "reaction force"
end

function get_formulation_type(problem::Problem{Contact})
    return :incremental
end

typealias ContactElements2D Union{Seg2}

function assemble!(problem::Problem{Contact}, time::Real)
    if problem.properties.dimension == -1
        problem.properties.dimension = dim = size(first(problem.elements), 1)
        info("assuming dimension of mesh tie surface is $dim")
        info("if this is wrong set is manually using problem.properties.dimension")
    end
    dimension = Val{problem.properties.dimension}
    finite_sliding = Val{problem.properties.finite_sliding}
    friction = Val{problem.properties.friction}
    dual_basis = Val{problem.properties.dual_basis}
    use_forwarddiff = Val{problem.properties.use_forwarddiff}
    assemble!(problem, time, dimension, finite_sliding, friction, dual_basis, use_forwarddiff)
end

""" Frictionless 2d small sliding contact with dual basis without forwarddiff. """
function assemble!(problem::Problem{Contact}, time::Real,
                   ::Type{Val{1}}, ::Type{Val{false}}, ::Type{Val{false}},
                   ::Type{Val{true}}, ::Type{Val{false}}; debug=false)

    props = problem.properties
    field_dim = get_unknown_field_dimension(problem)
    field_name = get_parent_field_name(problem)
    slave_elements = get_slave_elements(problem)

    # 1. calculate nodal normals and tangents for slave element nodes j ∈ S
    normals, tangents = calculate_normals(slave_elements, time, Val{1}; rotate_normals=props.rotate_normals)
    update!(slave_elements, "normal", normals)
    update!(slave_elements, "tangent", tangents)

    # 2. loop all slave elements
    for slave_element in slave_elements

        X1 = slave_element["geometry"](time)
        u1 = slave_element["displacement"](time)
        la1 = slave_element["reaction force"](time)
        x1 = X1 + u1
        n1 = slave_element["normal"](time)
        t1 = slave_element["tangent"](time)
        Q1_ = [n1[1] t1[1]]
        Q2_ = [n1[2] t1[2]]
        Z = zeros(2, 2)
        Q2 = [Q1_ Z; Z Q2_]

        # 3. loop all master elements
        for master_element in slave_element["master elements"](time)

            X2 = master_element["geometry"](time)
            u2 = master_element["displacement"](time)
            x2 = X2 + u2

            # 3.1 calculate segmentation
            xi1a = project_from_master_to_slave(slave_element, X2[1], time)
            xi1b = project_from_master_to_slave(slave_element, X2[end], time)
            xi1 = clamp([xi1a; xi1b], -1.0, 1.0)
            l = 1/2*abs(xi1[2]-xi1[1])
            isapprox(l, 0.0) && continue # no contribution in this master element

            # 3.2. bi-orthogonal basis
            nsl = length(slave_element)
            nm = length(master_element)
            De = zeros(nsl, nsl)
            Me = zeros(nsl, nsl)
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
            
            # 3.3. loop integration points of one integration segment and calculate
            # local mortar matrices
            fill!(De, 0.0)
            fill!(Me, 0.0)
            ge = zeros(field_dim*nsl)
            lae = zeros(field_dim*nsl)
            for ip in get_integration_points(slave_element, 3)
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
                x_s = X_s + N1*u1
                x_m = X_m + N2*u2
                la_s = Phi*la1
                ge += w*vec((x_m-x_s)*Phi')
                lae += w*vec(la_s*Phi')
            end

            # add contribution to contact virtual work
            sdofs = get_gdofs(problem, slave_element)
            mdofs = get_gdofs(problem, master_element)
            nsldofs = length(sdofs)
            nmdofs = length(mdofs)
            D2 = zeros(nsldofs, nsldofs)
            M2 = zeros(nmdofs, nmdofs)
            for i=1:field_dim
                D2[i:field_dim:end, i:field_dim:end] += De
                M2[i:field_dim:end, i:field_dim:end] += Me
            end
            add!(problem.assembly.C1, sdofs, sdofs, D2)
            add!(problem.assembly.C1, sdofs, mdofs, -M2)

            add!(problem.assembly.C2, sdofs, sdofs, Q2'*D2)
            add!(problem.assembly.C2, sdofs, mdofs, -Q2'*M2)
            add!(problem.assembly.g, sdofs, Q2'*ge)
            add!(problem.assembly.c, sdofs, Q2'*lae)

        end # master elements done

    end # slave elements done, contact virtual work ready

    S = sort(collect(keys(normals))) # slave element nodes
    C1 = sparse(problem.assembly.C1)
    ndofs = size(C1, 1)
    debug && info("ndofs = $ndofs")
    C2 = sparse(problem.assembly.C2)
    D = spzeros(ndofs, ndofs)
    g = sparse(problem.assembly.g)
    g = full(g)
    c = sparse(problem.assembly.c)
    c = full(c)
    debug && info("Contact slave nodes: $S")

    # constitutive modelling in tangent direction, frictionless contact
    for j in S
        dofs = [2*(j-1)+1, 2*(j-1)+2]
        C2[dofs[2],:] = 0.0
        g[dofs[2]] = 0.0
        D[dofs[2], dofs] = tangents[j]
    end
    debug && info("Constitutive modelling ready")

    # active / inactive node detection
    A = Set()
    I = Set()
    la = problem.assembly.la
    for j in S
        dofs = [2*(j-1)+1, 2*(j-1)+2]
        
        Cn = -g[dofs[1]]
        if length(la) != 0
            Cn += dot(normals[j], la[dofs])
            debug && info("slave $j: $(normals[j]) | $(la[dofs]) | $(c[dofs]) | $(g[dofs]) | $Cn")
        else
            debug && info("slave $j: $(normals[j]) |             | $(c[dofs]) | $(g[dofs]) | $Cn")
        end
        if Cn < 0
            push!(I, j)
            debug && info("slave $j INACTIVE")
            C1[dofs,:] = 0.0
            C2[dofs,:] = 0.0
            D[dofs,:] = 0.0
            g[dofs,:] = 0.0
        else
            push!(A, j)
        end
    end
    debug && info("active nodes: $A, inactive nodes: $I")

    problem.assembly.C1 = C1
    problem.assembly.C2 = C2
    problem.assembly.D = D
    problem.assembly.g = g

    return

end

