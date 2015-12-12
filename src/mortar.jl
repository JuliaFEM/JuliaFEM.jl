# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

# Mortar projection calculation for 2d

""" Find projection from slave nodes to master element, i.e. find xi2 from
master element corresponding to the xi1.
"""
function project_from_slave_to_master{S,M}(slave::Element{S}, master::Element{M}, xi1::Vector, time::Float64=0.0; max_iterations=5, tol=1.0e-9)
#   slave_basis = get_basis(slave)

    # slave side geometry and normal direction at xi1
    X1 = slave("geometry", xi1, time)
    N1 = slave("normal-tangential coordinates", xi1, time)[:,1]

    # master side geometry at xi2
    #master_basis = master.basis.data.basis
    #master_dbasis = master.basis.data.dbasis
    master_basis(xi) = get_basis(M, [xi])
    master_dbasis(xi) = get_dbasis(M, [xi])
    master_geometry = master("geometry")(time)

    function X2(xi2)
        N = master_basis(xi2)
        return sum([N[i]*master_geometry[i] for i=1:length(N)])
    end

    function dX2(xi2)
        dN = master_dbasis(xi2)
        return sum([dN[i]*master_geometry[i] for i=1:length(dN)])
    end

#    master_basis = get_basis(master)
#    X2(xi2) = master_basis("geometry", [xi2], time)
#    dX2(xi2) = dmaster_basis("geometry", xi2, time)

    # equation to solve
    R(xi2) = det([X2(xi2)-X1  N1]')
    dR(xi2) = det([dX2(xi2) N1]')
#   dR = ForwardDiff.derivative(R)

    # go!
    xi2 = 0.0
    for i=1:max_iterations
        dxi2 = -R(xi2) / dR(xi2)
        xi2 += dxi2
        if norm(dxi2) < tol
            return Float64[xi2]
        end
    end
    error("find projection from slave to master: did not converge")
end

""" Find projection from master surface to slave point, i.e. find xi1 from slave
element corresponding to the xi2. """
function project_from_master_to_slave{S,M}(slave::Element{S}, master::Element{M}, xi2::Vector, time::Float64=0.0; max_iterations=5, tol=1.0e-9)
#   slave_basis = get_basis(slave)

    # slave side geometry and normal direction at xi1

    slave_geometry = slave("geometry")(time)
    slave_normals = slave("normal-tangential coordinates")(time)
    #slave_basis = slave.basis.data.basis
    #slave_dbasis = slave.basis.data.dbasis
    slave_basis(xi) = get_basis(S, [xi])
    slave_dbasis(xi) = get_dbasis(S, [xi])

    function X1(xi1)
        N = slave_basis(xi1)
        return sum([N[i]*slave_geometry[i] for i=1:length(N)])
    end

    function dX1(xi1)
        dN = slave_dbasis(xi1)
        return sum([dN[i]*slave_geometry[i] for i=1:length(dN)])
    end

    function N1(xi1)
        N = slave_basis(xi1)
        return sum([N[i]*slave_normals[i] for i=1:length(N)])[:,1]
    end

    function dN1(xi1)
        dN = slave_dbasis(xi1)
        return sum([dN[i]*slave_normals[i] for i=1:length(dN)])[:,1]
    end

    #X1(xi1) = slave_basis("geometry", [xi1], time)
    #N1(xi1) = slave_basis("normal-tangential coordinates", [xi1], time)[:,1]

    #master_basis = get_basis(master)

    # master side geometry at xi2
    #X2 = master_basis("geometry", xi2, time)
    X2 = master("geometry", xi2, time)

    # equation to solve
    R(xi1) = det([X1(xi1)-X2  N1(xi1)]')
    dR(xi1) = det([dX1(xi1) N1(xi1)]') + det([X1(xi1)-X2 dN1(xi1)]')

    #=
    info("R(-1.0) = $(R(-1.0))")
    info("R( 0.0) = $(R(0.0))")
    info("R( 1.0) = $(R(1.0))")
    info("R( 1.5) = $(R(1.5))")
    info("dR(-1.0) = $(dR(-1.0))")
    info("dR( 0.0) = $(dR(0.0))")
    info("dR( 1.0) = $(dR(1.0))")
    info("dR( 1.5) = $(dR(1.5))")
    =#

    #dR = ForwardDiff.derivative(R)

    # go!
    xi1 = 0.0
    for i=1:max_iterations
        dxi1 = -R(xi1) / dR(xi1)
        xi1 += dxi1
        #info("dxi1 = $dxi1, xi1 = $xi1, norm(dxi1) = $(norm(dxi1))")
        if norm(dxi1) < tol
            return Float64[xi1]
        end
    end
    error("find projection from master to slave: did not converge")
end

### Mortar projection calculation for 3d cases

"""
Construct auxiliary plane for surface.

Parameters
----------
x::Array{Float64, 2}
    Node coordinates
ximp::Array{Float64, 1}
    Element mid-point in dimensionless mother element coordinates ξ
normals::Array{Float64, 2}
    Normal directions in nodes

Returns
-------
x0, Q
    x0::Array{Float64, 1} - origo of auxiliary plane
    Q::Array{Float64, 2} - orthogonal basis, first vector is normal direction
    and two rest vectors create orthonormal right-handed basis.

Examples
--------
Calculate auxiliary plane given nodal coordinates, midpoint of mother element,
node normals and suitable function space:

julia> xquad = [
...     -2.5   2.5  2.0  -2.0
...     -2.0  -2.0  2.3   2.0
...      1.0   0.7  0.0   1.0]
julia> m_midpoint = [0.0, 0.0]
julia> normals = [
...     0.05989060  0.0590504  0.225612   0.2445800
...    -0.00748633  0.1670810  0.182034  -0.0305725
...     0.99817700  0.9841730  0.957059   0.9691470]
julia> basis(xi) = [
...     (1-xi[1])(1-xi[2])/4
...     (1+xi[1])(1-xi[2])/4
...     (1+xi[1])(1+xi[2])/4
...     (1-xi[1])(1+xi[2])/4]'
julia> x0, Q = create_auxiliary_plane(xquad, mmidpoint, normals, basis)
julia> x0
3-element Array{Float64,1}:
 0.0
 0.075
 0.675
julia> Q
3x3 Array{Float64,2}:
 0.148586    0.988899    0.0
 0.0784519  -0.0117877   0.996848
 0.985783   -0.148118   -0.0793325

Notes
-----
- Midpoint in mother element typically (0, 0) for quadrangles and (1/3, 1/3)
  for triangles.
- Uses Gram-Schmidt process to find orthogonal basis
- [1](http://www.math.umn.edu/~olver/aims_/qr.pdf)
- [2](http://www.ecs.umass.edu/ece/ece313/Online_help/gram.pdf)
- [3](http://www.terathon.com/code/tangent.html)

"""
# function create_auxiliary_plane(x, ximp, normals, basis)
function create_auxiliary_plane{E}(element::Element{E}, time::Real)
#   proj(u, v) = dot(v, u) / dot(u, u) * u
#    xi = [1.0/3.0, 1.0/3.0]

    xi = get_reference_element_midpoint(E)
    x0 = element("geometry", xi, time)
    ntbasis = element("normal-tangential coordinates", xi, time)
    return x0, ntbasis
#=
    n = element("normal-tangential coordinates", xi, time)[:, 1]
    n /= norm(n)
    # gram-schmidt
    u1 = n
    j = indmax(abs(u1))
    v2 = zeros(3)
    v2[mod(j,3)+1] = 1.0
    u2 = v2 - proj(u1, v2)
    u3 = cross(u1, u2)
    t1 = u2/norm(u2)
    t2 = u3/norm(u3)
    new_basis = [n t1 t2]
    return x0, new_basis
=#
end

"""
Project point q onto a plane given by a point p and normal n.

Parameters
----------
q::Array{Float64, 2}
    point to project (row vector)
x0::Array{Float64, 2}
    origo of plane
n::Array{Float64, 2}
    normal vector of plane

Returns
-------
y::Array{Float64, 2}
    projected point

Examples
--------
julia> p = [-0.5 -1.0 4.0]'
julia> x0 = [0.0 0.075 0.675]'
julia> n = [0.1485860 0.0784519 0.9857830]'
julia> project_node_to_auxiliary_plane(p, x0, n)
3-element Array{Float64,1}:
  0.963455
 -1.2447
  0.925247

Notes
-----
[1](http://stackoverflow.com/questions/8942950/how-do-i-find-the-orthogonal-projection-of-a-point-onto-a-plane)

"""
function project_point_to_auxiliary_plane(p::Vector, x0::Vector, Q::Matrix)
    n = Q[:,1]
    ph = p - dot(p-x0, n)*n
    qproj = Q'*(ph-x0)
    if !isapprox(qproj[1], 0.0; atol=1.0e-12)
        info("project_point_to_auxiliary_plane(): point not projected correctly.")
        info("p: $p")
        info("x0: $x0")
        info("Q: \n$Q")
        info("qproj: $qproj")
        error("Failed to project point to auxiliary plane.")
    end
    return qproj[2:3]
end

"""
Find edge intersections of two planar arbitrary shape polygons.

Parameters
----------
S::Array{Float64,2}
M::Array{Float64,2}

Matrices with size (2, n) where n is number of vertices of each polygon.

Returns
-------
P::Array{Float64,2}
    Intersection points of polygons
n::Array{Float64,2}
    Neighbour info matrix with size (ns, mn). This keeps information which
    edges of polygons are intersecting. See further explanation in example
    below.

Examples
--------
Find intersection points of two triangles:

julia> S = [0 0; 3 0; 0 3]'
julia> M = [-1 1; 2 -1/2; 1 3/2]'
julia> P, n = get_edge_intersections(S, M)
julia> P
2x4 Array{Float64,2}:
 1.0  1.75  0.0  0.0
 0.0  0.0   0.5  1.25
julia> n
3x3 Array{Int64,2}:
 1  1  0
 0  0  0
 1  0  1)

So intersection points are: (1.00, 0.00), (1.75, 0.00), (0.00, 0.50), (0.00, 1.25).
"Neighbour matrix" can be interpreted as following:

    1 1 0 <--> First edge of S intersects edges 1 and 2 of M
    0 0 0 <--> Second edge of S doesn't intersect at all
    1 0 1 <--> Third edge of S intersects with edges 1 and 3 of M

"""
function get_edge_intersections(S::Matrix, M::Matrix)
    ns = size(S, 2)
    nm = size(M, 2)
    P = zeros(2, 0)
    n = zeros(Int64, ns, nm)
    k = 0
    for i=1:ns
        for j=1:nm
            b = M[:,j]-S[:,i]
            A = [S[:,mod(i,ns)+1]-S[:,i]  -M[:,mod(j,nm)+1]+M[:,j]]
            if rank(A) == 2
                r = A\b
                if (r[1]>=0) & (r[1]<=1) & (r[2]>=0) & (r[2]<=1)  # intersection found
                    k += 1
                    f = S[:,i]+r[1]*(S[:,mod(i,ns)+1] - S[:,i])
                    f = f''
                    P = hcat(P, f)
                    n[i, j] = 1
                end
            end
        end
    end
    return P, n
end


"""
Find any points laying inside or border of triangle.

Parameters
----------
Y::Array{Float64, 2}
    Triangle coordinates in 2×3 matrix
X::Array{Float64, 2}
    List of points to test in 2×n matrix

Returns
-------
P::Array{Float64, 2}
    List of points in triangle in 2×m matrix, where m is number of points inside triangle

Examples
--------
julia> S = [0.0 0.0; 3.0 0.0; 0.0 3.0]'  # triangle corner points
julia> pts = [-1.0 1.0; 2.0 -0.5; 1.0 1.5; 0.5 1.5]'  # points to tests
julia> points_in_triangle(S, pts)
2x2 Array{Float64,2}:
 1.0  0.5
 1.5  1.5
"""
function get_points_inside_triangle(Y::Matrix, X::Matrix)
    @assert size(Y, 2) == 3  # "Point in TRIANGLE..."
    P = zeros(2, 0)
    v0 = Y[:,2] - Y[:,1]
    v1 = Y[:,3] - Y[:,1] # find interior points of X in Y
    d00 = (v0'*v0)[1]
    d01 = (v0'*v1)[1]
    d11 = (v1'*v1)[1]  # using baricentric coordinates
    id = 1/(d00*d11 - d01*d01)
    for i=1:size(X, 2)
        v2 = X[:,i] - Y[:,1]
        d02 = (v0'*v2)[1]
        d12 = (v1'*v2)[1]
        u = (d11*d02-d01*d12)*id
        v = (d00*d12-d01*d02)*id
        if (u>=0) & (v>=0) & (u+v<=1)  # also include nodes on the boundary
            P = hcat(P, X[:,i]'')
        end
    end
    return P
end


""" Return unique objects with some given tolerance. This is used in next function
    because traditional unique() command returns row vectors as non-unique if they
    differs only a "little".
"""
function uniquetol(P, dim::Int; args...)
    @assert dim == 2
    items = Vector{Float64}[P[:,i] for i=1:size(P,dim)]
    new_items = Vector{Float64}[]
    for item in items
        has_found = false
        for new_item in new_items
            if isapprox(item, new_item; args...)
                has_found = true
                break
            end
        end
        if !has_found
            push!(new_items, item)
        end
    end
    return reshape([new_items...;], length(new_items[]), length(new_items))
end


"""
Make polygon clipping of shapes S and M.

Parameters
----------
S::Array{Float64, 2}
M::Array{Float64, 2}
    Shapes to clip. Needs to be triangles at the moment.

Returns
-------
Array{Float64, 2}, Array{Float64, 2}
- Polygon vertices in 2×n matrix, sorted in counter-clockwise order.
- 3×3 "neighbouring" matrix, see example.

Examples
--------
julia> S = [0 0; 3 0; 0 3]'
julia> M = [-1 1; 2 -1/2; 2 2]'
julia> P, n = clip_polygon(S, M)
julia> P
2x6 Array{Float64,2}:
 0.0  1.0  2.0  2.0  1.25  0.0
 0.5  0.0  0.0  1.0  1.75  1.33333,
julia> n
3x3 Array{Int64,2}:
 1  0  1 <- first edge of M ([-1 1; 2 -1/2]') intersects with edges 1 and 3 of S ([0 0; 3 0]' and [0 3; 0 0]')
 1  1  0 <- second edge of M ([2 -1/2; 2 2]') intersects with edges 1 and 2 of S
 0  1  1 <- third edge of M ([2 2; -1 1]') intersects with edgse 2 and 3 of S

"""
function clip_polygon(S::Matrix, M::Matrix)
    P1, neighbours = get_edge_intersections(M, S)
    P2 = get_points_inside_triangle(M, S)
    P3 = get_points_inside_triangle(S, M)
    P = hcat(P1, P2, P3)
    P = uniquetol(P, 2)
    meanval = mean(P, 2)
    tmp = P .- meanval
    angles = atan2(tmp[2,:], tmp[1,:])
    angles = reshape(angles, length(angles))
    order = sortperm(angles)
    return P[:, order], neighbours
end


"""
Calculate polygon geometric center point

Parameters
----------
P::Array{Float64, 2}
    Polygon vertices in 2×n matrix

Returns
-------
Array{Float63, 2}
    Center point

Examples
--------
julia> P
2x6 Array{Float64,2}:
 0.0  1.0  2.0  2.0  1.25  0.0
 0.5  0.0  0.0  1.0  1.75  1.33333,
julia> C = get_polygon_cp(P)
2x1 Array{Float64,2}:
 1.039740
 0.804701

"""
function calculate_polygon_centerpoint(P::Matrix)
    n = size(P, 2)
    A = 0.0
    for i=1:n
        A += 1/2*(P[1,i]*P[2,mod(i,n)+1] - P[1,mod(i,n)+1]*P[2,i])
    end
    Cx = 0.0
    Cy = 0.0
    for i=1:n
        inext = mod(i, n)+1
        Cx += 1/(6*A)*(P[1,i] + P[1,inext])*(P[1,i]*P[2,inext] - P[1,inext]*P[2,i])
        Cy += 1/(6*A)*(P[2,i] + P[2,inext])*(P[1,i]*P[2,inext] - P[1,inext]*P[2,i])
    end
    return Float64[Cx, Cy]
end

"""
Project point from auxiliary plane to parametric surface given by (ξ₁, ξ₂)

Parameters
----------
p::Array{Float64,1}
    point in auxiliary plane, in (n,t1,t2) coordinate system
x0::Array{Float64,1}
    origo of auxiliary plane cs
Q::Array{Float64,2}
    basis of auxiliary plane cs
x::Array{Float64,2}
    surface node coords
basis::Array{Float64,2}
    surface basis functions
dbasis::Array{Float64,2}
    partial derivatives of surface basis functions

Returns
-------
Array{Float64,2}
    solution vector (d, ξ₁, ξ₂) where d is distance to surface

Examples
--------
Define surface with node points, basis + dbasis

julia> xquad = [
...     -2.5 -2.0 1.0
...      2.5 -2.0 0.7
...      2.0  2.3 0.0
...     -2.0  2.0 1.0]'
julia> basis(xi) = [
...     (1-xi[1])(1-xi[2])/4
...     (1+xi[1])(1-xi[2])/4
...     (1+xi[1])(1+xi[2])/4
...     (1-xi[1])(1+xi[2])/4]
julia> dbasis(xi) = [
...     -(1-xi[2])/4    -(1-xi[1])/4
...      (1-xi[2])/4    -(1+xi[1])/4
...      (1+xi[2])/4     (1+xi[1])/4
...     -(1+xi[2])/4     (1-xi[1])/4]

We aim to find point p, which we first project to auxiliary plane defined as following
julia> p = [-2.5 -2.0 1.0]'
julia> x0 = [0.0 0.075 0.675]'
julia> Q = [
...     0.1485860   0.9888990   0.0000000
...     0.0784519  -0.0117877   0.9968480
...     0.9857830  -0.1481180  -0.0793325]

Our projected point is therefore
julia> n = Q[:,1] # first component is normal direction
julia> ph = project_node_to_auxiliary_plane(p, x0, n)
julia> ph = Q'(ph-x0)
julia> ph
3x1 Array{Float64,2}:
  1.33264e-7
 -2.49593
 -2.09424

Our point ph is now in auxiliary plane in n,t1,t2 coordinate system. Next we
project it back to surface defined by xquad*basis

julia> theta = project_point_from_plane_to_surface(ph, x0, Q, xquad, basis, dbasis)
julia> theta
3x1 Array{Float64,2}:
 -0.213874
 -0.999999
 -1.0

We see that our ξ₁ = ξ₂ = -1 so we found first point of xquad
[-2.5 -2.0 1.0]' correctly.

julia> xquad*basis(theta[2:3])
3-element Array{Float64,1}:
 -2.5
 -2.0
  1.0

"""
function project_point_from_plane_to_surface{E}(p::Vector, x0::Vector, Q::Matrix, element::Element{E}, time::Real; max_iterations::Int=10, iter_tol::Float64=1.0e-9)
    basis(xi) = get_basis(E, xi)
    dbasis(xi) = get_dbasis(E, xi)
    x = element("geometry", time)
    ph = Q*[0; p] + x0
    theta = Float64[0.0, 0.0, 0.0]
    n = Q[:,1]
    for i=1:max_iterations
        b = ph + theta[1]*n - basis(theta[2:3])*x
        J = [n -dbasis(theta[2:3])*x]
        dtheta = J \ -b
        theta += dtheta
        if norm(dtheta) < iter_tol
            return theta
        end
    end
    error("project_point_to_auxiliary_plane: did not converge in $max_iterations iterations!")
end


### Mortar problem

"""
Parameters
----------
node_csys
    coordinate system in node, normal + tangent + "binormal"
    in 3d 3x3 matrix, in 2d 2x2 matrix, respectively
"""
abstract MortarProblem <: AbstractProblem

function MortarProblem(parent_field_name, parent_field_dim, dim=1, elements=[])
    return BoundaryProblem{MortarProblem}(parent_field_name, parent_field_dim, dim, elements)
end

# Mortar assembly

typealias MortarElements2D Union{Seg2, Seg3}

function assemble!{E<:MortarElements2D}(assembly::Assembly, problem::BoundaryProblem{MortarProblem}, slave_element::Element{E}, time::Real)

    # get dimension and name of PARENT field
    field_dim = problem.parent_field_dim
    field_name = problem.parent_field_name

    slave_dofs = get_gdofs(slave_element, field_dim)

    for master_element in slave_element["master elements"]
        xi1a = project_from_master_to_slave(slave_element, master_element, [-1.0])
        xi1b = project_from_master_to_slave(slave_element, master_element, [ 1.0])
        xi1 = clamp([xi1a xi1b], -1.0, 1.0)
        l = 1/2*(xi1[2]-xi1[1])
        if abs(l) < 1.0e-6
            warn("No contribution")
            continue # no contribution
        end
        master_dofs = get_gdofs(master_element, field_dim)
        for ip in get_integration_points(slave_element, Val{5})
            w = ip.weight*det(slave_element, ip, time)*l

            # integration point on slave side segment
            xi_gauss = 1/2*(1-ip.xi)*xi1[1] + 1/2*(1+ip.xi)*xi1[2]
            # projected integration point
            xi_projected = project_from_slave_to_master(slave_element, master_element, xi_gauss)

            # add contribution to left hand side
            N1 = slave_element(xi_gauss, time)
            N2 = master_element(xi_projected, time)
            S = w*N1'*N1
            M = w*(N1'*N2)'
#           M = w*N1'*N2
            # FIXME: why this needs now to be transpose?
            # assembly / repeat
            for i=1:field_dim
                sd = slave_dofs[i:field_dim:end]
                md = master_dofs[i:field_dim:end]
                add!(assembly.stiffness_matrix, sd, sd, S)
                add!(assembly.stiffness_matrix, sd, md, -M)
            end

        end
    end
end


typealias MortarElements3D Union{Tri3}

function assemble!{E<:MortarElements3D}(assembly::Assembly, problem::BoundaryProblem{MortarProblem}, slave_element::Element{E}, time::Real)
    field_dim = problem.parent_field_dim
    field_name = problem.parent_field_name
    slave_dofs = get_gdofs(slave_element, field_dim)
#   info("Slave dofs: $slave_dofs")
#   info("Field dim: $field_dim")

    # create auxiliary plane and project slave nodes to it
    # x0 = origo, Q = local basis
    x0, Q = create_auxiliary_plane(slave_element, time)
    S = Vector{Float64}[]
    for p in slave_element("geometry", time)
        push!(S, project_point_to_auxiliary_plane(p, x0, Q))
    end
    S = reshape([S...;], 2, 3)

    integration_points = get_integration_points(E, Val{5})

    for master_element in slave_element["master elements"]
        master_dofs = get_gdofs(master_element, field_dim)
        # project master nodes to auxiliary plane and create polygon clipping
        M = Vector{Float64}[]
        for p in master_element("geometry", time)
            push!(M, project_point_to_auxiliary_plane(p, x0, Q))
        end
        M = reshape([M...;], 2, 3)
        P, neighbours = clip_polygon(S, M)
        C = calculate_polygon_centerpoint(P)

        npts = size(P, 2) # number of vertices in polygon
#       S = zeros(3, 3)
#       M = zeros(3, 3)
        for i=1:npts # loop vertices and create temporary integrate cells
            xvec = [C[1], P[1, i], P[1, mod(i, npts)+1]]
            yvec = [C[2], P[2, i], P[2, mod(i, npts)+1]]
            X = hcat(xvec, yvec)'
            geom = Field(Vector{Float64}[X[:,j] for j=1:size(X,2)])
            for ip in integration_points
                # calculate determiant of jacobian
                dN = get_dbasis(E, ip.xi)
                J = sum([kron(dN[:,j], geom[j]') for j=1:length(geom)])
                w = ip.weight*det(J)
                # gauss point in auxiliary plane
                N = get_basis(E, ip.xi)
                x = vec(N*geom)
                # find projection of gauss point to master and slave elements
                theta1 = project_point_from_plane_to_surface(x, x0, Q, slave_element, time)
                theta2 = project_point_from_plane_to_surface(x, x0, Q, master_element, time)
                # evaluate shape functions values in gauss point and add contribution to matrices
                N1 = slave_element(theta1[2:3], time)
                N2 = master_element(theta2[2:3], time)
                S = w*N1'*N1
                M = w*N1'*N2
                for k=1:field_dim
                    sd = slave_dofs[k:field_dim:end]
                    md = master_dofs[k:field_dim:end]
                    add!(assembly.stiffness_matrix, sd, sd, S)
                    add!(assembly.stiffness_matrix, sd, md, -M)
#                   info("sd = $sd")
#                   info("md = $md")
                end
            end
        end
#       info("S = \n$S")
#       info("M = \n$M")
    end
end

