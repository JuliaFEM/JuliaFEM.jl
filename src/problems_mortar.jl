# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
Parameters
----------
dimension
    dimension of surface, 1 for 2d problems (plane strain, plane stress,
    axisymmetric) and 2 for 3d problems. It not given, try to determine problem
    dimension from first element
rotate_normals
    if all surface elements are in cw order instead of ccw, this can be used to
    swap normal directions so that normals point to outward of body
adjust
    for elasticity problems only; closes any gaps between surfaces if found
dual_basis
    use bi-orthogonal basis when interpolating Lagrange multiplier space
use_forwarddiff
    use forwarddiff to linearize contact constraints directly from weighted
    gap function
distval
    charasteristic measure, contact pairs with distance over this value are
    skipped from contact segmentation algorithm
linear_surface_elements
    convert quadratic surface elements to linear elements on the fly, notice
    that middle nodes are missing Lagrange multipliers
split_quadratic_slave_elements
    split quadratic surface elements to several linear sub-elements to get
    Lagrange multiplier to middle nodes also
split_quadratic_master_elements
    split quadratic master elements to several linear sub-elements
store_fields
    not used
"""
mutable struct Mortar <: BoundaryProblem
    dimension :: Int
    rotate_normals :: Bool
    adjust :: Bool
    dual_basis :: Bool
    use_forwarddiff :: Bool
    distval :: Float64
    linear_surface_elements :: Bool
    split_quadratic_slave_elements :: Bool
    split_quadratic_master_elements :: Bool
    alpha :: Float64
    drop_tolerance :: Float64
    store_fields :: Vector{Symbol}
end

function Mortar()
    default_fields = []
    return Mortar(-1, false, false, false, false, Inf, true, true, true, 0.0, 1.0e-9, default_fields)
end

function assemble!(problem::Problem{Mortar}, time::Float64)
    if length(problem.elements) == 0
        @warn("No elements defined in interface $(problem.name), this will result empty assembly!")
        return
    end
    if problem.properties.dimension == -1
        problem.properties.dimension = dim = size(first(problem.elements), 1)
        @info("Assuming dimension of mesh tie surface is $dim. If this is wrong set is manually using problem.properties.dimension")
    end
    dimension = Val{problem.properties.dimension}
    use_forwarddiff = Val{problem.properties.use_forwarddiff}
    assemble!(problem, time, dimension, use_forwarddiff)
end

function get_slave_elements(problem::Problem)
    cond(el) = haskey(el, "master elements") || haskey(el, "potential master elements")
    return filter(cond, get_elements(problem))
end

""" Given a CCW ordered set of vertices, calculate area of polygon.

Examples
--------

julia> P = Vector[[1/3, 5/12, 1/2], [1/3, 1/2, 1/2], [1/2, 1/2, 1/2], [1/2, 1/3, 1/2], [5/12, 1/3, 1/2]]
5-element Array{Array{T,1},1}:
 [0.333333,0.416667,0.5]
 [0.333333,0.5,0.5]
 [0.5,0.5,0.5]
 [0.5,0.333333,0.5]
 [0.416667,0.333333,0.5]

julia> A = calculate_polygon_area(P)
0.02430555555555556

julia> isapprox(A, 7/288)
true

"""
function calculate_polygon_area(P)
    N_P = length(P)
    A = sum([norm(1/2*cross(P[i]-P[1], P[mod(i,N_P)+1]-P[1])) for i=2:N_P])
    return A
end

""" Function to print useful debug information from interface to find bugs. """
function diagnose_interface(problem::Problem{Mortar}, time::Float64)
    @info("Diagnosing Mortar interface...")
    props = problem.properties
    field_dim = get_unknown_field_dimension(problem)
    field_name = get_parent_field_name(problem)
    slave_elements = get_slave_elements(problem)

    I_area = 0.0

    if props.split_quadratic_slave_elements
        @info("props.split_quadratic_slave_elements = true")
        if !props.linear_surface_elements
            @warn("Mortar3D: split_quadratic_surfaces = true and linear_surface_elements = false maybe have unexpected behavior")
        end
        slave_elements = split_quadratic_elements(slave_elements, time)
    end
    @info("Number of slave elements in interface: $(length(slave_elements))")

    # 1. calculate nodal normals and tangents for slave element nodes j ∈ S
    normals = calculate_normals(slave_elements, time, Val{2};
                                rotate_normals=props.rotate_normals)
    update!(slave_elements, "normal", time => normals)

    S_areas = []
    C_areas = []
    P_areas = []

    for slave_element in slave_elements

        @info(repeat("-", 80))
        @info("Processing slave element $(slave_element.id), type = $(get_element_type(slave_element))")
        @info(repeat("-", 80))

        S_area = 0.0
        S_area_in_contact = 0.0
        for ip in get_integration_points(slave_element)
            S_area += ip.weight*slave_element(ip, time, Val{:detJ})
        end
        @info("Total area of slave element = $S_area")


        if props.linear_surface_elements
            @info("Converting slave element to linear surface element")
            slave_element = convert_to_linear_element(slave_element)
        end

        slave_element_nodes = get_connectivity(slave_element)
        @info("Slave element connectivity = $slave_element_nodes")
        nsl = length(slave_element)
        X1 = slave_element("geometry", time)
        n1 = tuple(collect(normals[j] for j in slave_element_nodes)...)

        # project slave nodes to auxiliary plane (x0, Q)
        xi = get_mean_xi(slave_element)
        N = vec(get_basis(slave_element, xi, time))
        x0 = interpolate(N,X1)
        n0 = interpolate(N,n1)
        @info("Auxiliary plane x0 = $x0, n0 = $n0")
        S = Vector[project_vertex_to_auxiliary_plane(X1[i], x0, n0) for i=1:nsl]
        check_orientation!(S, n0)
        @info("Slave element $(slave_element.id) vertices in auxiliary plane: $S")

        # 3. loop all master elements
        master_elements = slave_element("master elements", time)
        if props.split_quadratic_master_elements
            master_elements = split_quadratic_elements(master_elements, time)
        end

        for master_element in master_elements

            if props.linear_surface_elements
                master_element = convert_to_linear_element(master_element)
            end

            master_element_nodes = get_connectivity(master_element)
            nm = length(master_element)
            X2 = master_element("geometry", time)

            if norm(mean(X1) - mean(X2)) > problem.properties.distval
                # elements are "far enough"
                continue
            end

            # 3.1 project master nodes to auxiliary plane and create polygon clipping
            M = Vector[project_vertex_to_auxiliary_plane(X2[i], x0, n0) for i=1:nm]
            check_orientation!(M, n0)
            P = get_polygon_clip(S, M, n0)
            if length(P) < 3
                if length(P) == 0
                    continue
                end
                if length(P) == 1
                    @info("length(P) == 1, shared vertex")
                end
                if length(P) == 2
                    @info("length(P) == 2, shared edge")
                end
                continue
            end
            @info("Master element $(master_element.id) vertices in auxiliary plane = $M")
            check_orientation!(P, n0)
            P_area_ = calculate_polygon_area(P)
            @info("Polygon clip found, P=$P, N_P = $(length(P)), area of polygon = $P_area_")
            if isapprox(P_area_, 0.0)
                error("Polygon P has zero area: $P_area_")
            end

            P_area = 0.0

            C0 = calculate_centroid(P)
            @info("Centroid of polygon = $C0")

            # 4. loop integration cells
            all_cells = get_cells(P, C0)
            @info("Polygon is splitted to $(length(all_cells)) integration cells.")
            for (cell_id, cell) in enumerate(all_cells)
                C_area = 0.0
                virtual_element = Element(Tri3, Int[])
                update!(virtual_element, "geometry", tuple(cell...))

                # 5. loop integration point of integration cell
                for ip in get_integration_points(virtual_element, 3)
                    N = vec(get_basis(virtual_element, ip, time))
                    detJ = virtual_element(ip, time, Val{:detJ})
                    w = ip.weight*detJ
                    # project gauss point from auxiliary plane to master and slave element
                    x_gauss = virtual_element("geometry", ip, time)
                    xi_s, alpha = project_vertex_to_surface(x_gauss, x0, n0, slave_element, X1, time)
                    xi_m, alpha = project_vertex_to_surface(x_gauss, x0, n0, master_element, X2, time)
                    C_area += w
                end # integration points done
                @info("Cell $cell_id has area of $C_area")
                P_area += C_area
                push!(C_areas, C_area)
            end # integration cells done

            if !isapprox(P_area, P_area_)
                error("P_area = $P_area, should be $P_area_")
            end

            S_area_in_contact += P_area
            push!(P_areas, P_area)

        end # master elements done

        S_perc = S_area_in_contact / S_area * 100.0
        push!(S_areas, S_area_in_contact)
        @info("Area of slave element in contact: $S_area_in_contact, it's $S_perc % of total element area")

        I_area += S_area_in_contact

    end # slave elements done, contact virtual work ready

    @info("Area of interface: $I_area")
    @info("Smallest cell area: $(minimum(C_areas))")
    @info("Smallest polygon area: $(minimum(P_areas))")
    @info("Smallest slave element area in contact: $(minimum(S_areas))")

end
