# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

type Mortar <: BoundaryProblem
    rotate_normals :: Bool
    adjust :: Bool
    tolerance :: Float64
end

function Mortar()
    return Mortar(false, false, 0.0)
end

function get_unknown_field_name(::Type{Mortar})
    return "reaction force"
end

function get_formulation_type(problem::Problem{Mortar})
    return :incremental
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

function project_from_master_to_slave{E<:MortarElements2D}(slave_element::Element{E}, x2, time)
    x1_ = slave_element["geometry"](time)
    n1_ = slave_element["normal"](time)
    x1(xi1) = vec(get_basis(slave_element, [xi1], time))*x1_
    dx1(xi1) = vec(get_dbasis(slave_element, [xi1], time))*x1_
    n1(xi1) = vec(get_basis(slave_element, [xi1], time))*n1_
    dn1(xi1) = vec(get_dbasis(slave_element, [xi1], time))*n1_
    R(xi1) = cross2(x1(xi1)-x2, n1(xi1))
    dR(xi1) = cross2(dx1(xi1), n1(xi1)) + cross2(x1(xi1)-x2, dn1(xi1))
    xi1 = newton(R, dR, 0.0)
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

function calculate_normals(elements, time, rotate_normals=false)
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
    S = sort(collect(keys(tangents)))
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

function calculate_normals!(elements, time, rotate_normals=false)
    normals, tangents = calculate_normals(elements, time, rotate_normals)
    for element in elements
        conn = get_connectivity(element)
        update!(element, "normal", time => [normals[j] for j in conn])
        update!(element, "tangent", time => [tangents[j] for j in conn])
    end
end

function assemble!(problem::Problem{Mortar}, time::Real)

    props = problem.properties
    field_dim = get_unknown_field_dimension(problem)
    field_name = get_parent_field_name(problem)
    slave_elements = filter(el -> haskey(el, "master elements"), get_elements(problem))

    # 1. calculate nodal normals and tangents for slave element nodes j ∈ S
    normals, tangents = calculate_normals(slave_elements, time, props.rotate_normals)
    update!(slave_elements, "normal", normals)
    update!(slave_elements, "tangent", tangents)
    S = sort(collect(keys(normals)))

    # 2. loop all slave elements
    for slave_element in slave_elements
        haskey(slave_element, "master elements") || continue

        slave_element_nodes = get_connectivity(slave_element)
        nsl = length(slave_element)
        X1 = slave_element["geometry"](time)
        n1 = Field([normals[j] for j in slave_element_nodes])

        # 3. loop all master elements
        for master_element in slave_element["master elements"](time)

            master_element_nodes = get_connectivity(master_element)
            nm = length(master_element)
            X2 = master_element["geometry"](time)

            # 3.1 calculate segmentation
            xi1a = project_from_master_to_slave(slave_element, X2[1], time)
            xi1b = project_from_master_to_slave(slave_element, X2[end], time)
            xi1 = clamp([xi1a; xi1b], -1.0, 1.0)
            l = 1/2*abs(xi1[2]-xi1[1])
            isapprox(l, 0.0) && continue # no contribution in this master element

            # 3.3. loop integration points of one integration segment and calculate
            # local mortar matrices
            De = zeros(nsl, nsl)
            Me = zeros(nsl, nm)
            ge = zeros(nsl)
            for ip in get_integration_points(slave_element, 2)
                detJ = slave_element(ip, time, Val{:detJ})
                w = ip.weight*detJ*l
                xi = ip.coords[1]
                xi_s = dot([1/2*(1-xi); 1/2*(1+xi)], xi1)
                N1 = vec(get_basis(slave_element, xi_s, time))
                # project gauss point from slave element to master element in direction n_s
                X_s = N1*X1 # coordinate in gauss point
                n_s = N1*n1 # normal direction in gauss point
                xi_m = project_from_slave_to_master(master_element, X_s, n_s, time)
                N2 = vec(get_basis(master_element, xi_m, time))
                X_m = N2*X2 
                De += w*N1*N1'
                Me += w*N1*N2'
                if props.adjust
                    g = X_s-X_m
                    if g < props.tol
                        ge += w*g
                    end
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
                add!(problem.assembly.g, lsdofs, ge)
            end


        end # master elements done

    end # slave elements done, contact virtual work ready

end
