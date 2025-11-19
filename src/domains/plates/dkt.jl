# DKT (Discrete Kirchhoff Triangle) Plate Element
#
# Reference implementation following modern JuliaFEM architecture.
# Fixes Issue #265: Parenthesis error in dHxdeta function (line 180 in old code).
#
# References:
# - Batoz, J.-L., Bathe, K.-J., & Ho, L.-W. (1980).
#   "A study of three-node triangular plate bending elements."
#   International Journal for Numerical Methods in Engineering, 15(12), 1771-1812.
# - Lucena Neto, E., et al. (2017).
#   "An Explicit Consistent Geometric Stiffness Matrix for the DKT Element."
#   Latin American Journal of Solids and Structures, 14(4), 613-628.

# Dependencies
# When integrated: using ..JuliaFEM
# When standalone: Use mock Physics from test file
using LinearAlgebra
using SparseArrays
using Tensors

const DKT_NNODE = 3
const DKT_DOF_PER_NODE = 3
const DKT_ELEMENT_DOF = DKT_NNODE * DKT_DOF_PER_NODE
const DKT_GAUSS_POINTS = ((1.0 / 6.0, 1.0 / 6.0), (2.0 / 3.0, 1.0 / 6.0), (1.0 / 6.0, 2.0 / 3.0))
const DKT_GAUSS_WEIGHT = 1.0 / 3.0

triangle_area(X::NTuple{3,Vec{2,T}}) where T =
    0.5 * abs((X[2][1] - X[1][1]) * (X[3][2] - X[1][2]) - (X[3][1] - X[1][1]) * (X[2][2] - X[1][2]))

function fill_coordinate_buffer!(buffer::Vector{Vec{2,Float64}}, nodes::Vector{Vec{3,Float64}}, conn::NTuple{DKT_NNODE,UInt32})
    @inbounds for (i, node_id) in enumerate(conn)
        coords = nodes[Int(node_id)]
        buffer[i] = Vec{2}((coords[1], coords[2]))
    end
    return buffer
end

function bending_matrix(material::AbstractMaterial, thickness::Float64)
    D_tensor = constitutive_matrix_plate(material, thickness)
    D_mat = Matrix{Float64}(undef, 3, 3)
    for i in 1:3, j in 1:3
        D_mat[i, j] = D_tensor[i, j]
    end
    return D_mat
end

function plate_global_dofs!(gdofs::Vector{Int}, conn::NTuple{DKT_NNODE,UInt32})
    @inbounds for (local_idx, node_id) in enumerate(conn)
        base = DKT_DOF_PER_NODE * (Int(node_id) - 1)
        offset = DKT_DOF_PER_NODE * (local_idx - 1)
        gdofs[offset + 1] = base + 1
        gdofs[offset + 2] = base + 2
        gdofs[offset + 3] = base + 3
    end
    return gdofs
end

function build_plate_sparsity_pattern(mesh::Mesh)
    ndofs_total = DKT_DOF_PER_NODE * length(mesh.nodes)
    ndofs_per_elem = DKT_DOF_PER_NODE * DKT_NNODE
    capacity = length(mesh.connectivity) * ndofs_per_elem * ndofs_per_elem

    I = Vector{Int}()
    J = Vector{Int}()
    sizehint!(I, capacity)
    sizehint!(J, capacity)

    elem_dofs = Vector{Int}(undef, ndofs_per_elem)

    for conn in mesh.connectivity
        plate_global_dofs!(elem_dofs, conn)
        @inbounds for a in 1:ndofs_per_elem
            for b in 1:ndofs_per_elem
                push!(I, elem_dofs[a])
                push!(J, elem_dofs[b])
            end
        end
    end

    return I, J, ndofs_total
end

function plate_element_indices(mesh::Mesh, element_set::Symbol)
    if element_set == :all
        return collect(UInt32(1):UInt32(length(mesh.connectivity)))
    end
    elem_set = get_element_set(mesh, element_set)
    return sort!(collect(elem_set))
end

# ============================================================================
# DKT Shape Functions
# ============================================================================

"""
    DKTShapeFunctions

Precomputed shape functions and derivatives for DKT element.

The DKT element uses modified shape functions that enforce Kirchhoff
constraints (zero transverse shear) at specific points on each edge.
"""
struct DKTShapeFunctions{T}
    # Geometric parameters (element-specific)
    x23::T
    x31::T
    x12::T
    y23::T
    y31::T
    y12::T
    L23::T
    L31::T
    L12::T

    # Shape function coefficients (for edges 23, 31, 12 → nodes 1, 2, 3)
    a4::T
    a5::T
    a6::T
    b4::T
    b5::T
    b6::T
    c4::T
    c5::T
    c6::T
    d4::T
    d5::T
    d6::T
    e4::T
    e5::T
    e6::T

    # Derivative coefficients
    p4::T
    p5::T
    p6::T
    q4::T
    q5::T
    q6::T
    r4::T
    r5::T
    r6::T
    t4::T
    t5::T
    t6::T
end

"""
    DKTShapeFunctions(X::NTuple{3, Vec{2, T}}) where T

Construct DKT shape functions from nodal coordinates.

# Arguments
- `X`: Tuple of 3 nodal coordinates (Vec{2} each)

# Returns
- `DKTShapeFunctions` with precomputed coefficients
"""
function DKTShapeFunctions(X::NTuple{3,Vec{2,T}}) where T
    # Extract nodal coordinates
    x1, y1 = X[1][1], X[1][2]
    x2, y2 = X[2][1], X[2][2]
    x3, y3 = X[3][1], X[3][2]

    # Edge vectors: xij = xi - xj, yij = yi - yj
    x23 = x2 - x3
    x31 = x3 - x1
    x12 = x1 - x2
    y23 = y2 - y3
    y31 = y3 - y1
    y12 = y1 - y2

    # Edge lengths squared
    L23 = x23^2 + y23^2
    L31 = x31^2 + y31^2
    L12 = x12^2 + y12^2

    # Shape function coefficients (Batoz eq. 17-19)
    # For edge k opposite to node i (k=4,5,6 for edges 23,31,12)
    a4 = -x23 / L23
    a5 = -x31 / L31
    a6 = -x12 / L12

    b4 = 3 * x23 * y23 / (4 * L23)
    b5 = 3 * x31 * y31 / (4 * L31)
    b6 = 3 * x12 * y12 / (4 * L12)

    c4 = (x23^2 - 2 * y23^2) / (4 * L23)
    c5 = (x31^2 - 2 * y31^2) / (4 * L31)
    c6 = (x12^2 - 2 * y12^2) / (4 * L12)

    d4 = -y23 / L23
    d5 = -y31 / L31
    d6 = -y12 / L12

    e4 = (y23^2 - 2 * x23^2) / (4 * L23)
    e5 = (y31^2 - 2 * x31^2) / (4 * L31)
    e6 = (y12^2 - 2 * x12^2) / (4 * L12)

    # Derivative coefficients (Batoz eq. 20-21)
    p4 = -6 * x23 / L23
    p5 = -6 * x31 / L31
    p6 = -6 * x12 / L12

    q4 = 3 * x23 * y23 / L23
    q5 = 3 * x31 * y31 / L31
    q6 = 3 * x12 * y12 / L12

    r4 = 3 * y23^2 / L23
    r5 = 3 * y31^2 / L31
    r6 = 3 * y12^2 / L12

    t4 = -6 * y23 / L23
    t5 = -6 * y31 / L31
    t6 = -6 * y12 / L12

    return DKTShapeFunctions(
        x23, x31, x12, y23, y31, y12, L23, L31, L12,
        a4, a5, a6, b4, b5, b6, c4, c5, c6, d4, d5, d6, e4, e5, e6,
        p4, p5, p6, q4, q5, q6, r4, r5, r6, t4, t5, t6
    )
end

"""
    tri3_shape_functions(ξ::T, η::T) where T

Standard Tri3 quadratic shape functions in area coordinates.

Returns tuple of 6 shape function values (N1, N2, N3, N4, N5, N6).
"""
function tri3_shape_functions(ξ::T, η::T) where T
    ζ = 1 - ξ - η
    N1 = ζ * (2 * ζ - 1)
    N2 = ξ * (2 * ξ - 1)
    N3 = η * (2 * η - 1)
    N4 = 4 * ξ * η
    N5 = 4 * η * ζ
    N6 = 4 * ξ * ζ
    return (N1, N2, N3, N4, N5, N6)
end

"""
    rotation_shape_functions_x(sf::DKTShapeFunctions, ξ::T, η::T) where T

Shape functions for rotation θx (rotation about x-axis).

Returns 1×9 row vector: [Hx1 Hx2 Hx3 Hx4 Hx5 Hx6 Hx7 Hx8 Hx9]
corresponding to DOFs: [w1 θx1 θy1 w2 θx2 θy2 w3 θx3 θy3]
"""
function rotation_shape_functions_x(sf::DKTShapeFunctions, ξ::T, η::T) where T
    N1, N2, N3, N4, N5, N6 = tri3_shape_functions(ξ, η)

    # Batoz eq. 14 (rotation about x-axis)
    Hx1 = 1.5 * (sf.a6 * N6 - sf.a5 * N5)
    Hx2 = sf.b5 * N5 + sf.b6 * N6
    Hx3 = N1 - sf.c5 * N5 - sf.c6 * N6

    Hx4 = 1.5 * (sf.a4 * N4 - sf.a6 * N6)
    Hx5 = sf.b4 * N4 + sf.b6 * N6
    Hx6 = N2 - sf.c4 * N4 - sf.c6 * N6

    Hx7 = 1.5 * (sf.a5 * N5 - sf.a4 * N4)
    Hx8 = sf.b4 * N4 + sf.b5 * N5
    Hx9 = N3 - sf.c4 * N4 - sf.c5 * N5

    return (Hx1, Hx2, Hx3, Hx4, Hx5, Hx6, Hx7, Hx8, Hx9)
end

"""
    rotation_shape_functions_y(sf::DKTShapeFunctions, ξ::T, η::T) where T

Shape functions for rotation θy (rotation about y-axis).

Returns 1×9 row vector: [Hy1 Hy2 Hy3 Hy4 Hy5 Hy6 Hy7 Hy8 Hy9]
"""
function rotation_shape_functions_y(sf::DKTShapeFunctions, ξ::T, η::T) where T
    N1, N2, N3, N4, N5, N6 = tri3_shape_functions(ξ, η)

    # Batoz eq. 15 (rotation about y-axis)
    Hy1 = 1.5 * (sf.d6 * N6 - sf.d5 * N5)
    Hy2 = -N1 + sf.e5 * N5 + sf.e6 * N6
    Hy3 = -sf.b5 * N5 - sf.b6 * N6

    Hy4 = 1.5 * (sf.d4 * N4 - sf.d6 * N6)
    Hy5 = -N2 + sf.e4 * N4 + sf.e6 * N6
    Hy6 = -sf.b4 * N4 - sf.b6 * N6

    Hy7 = 1.5 * (sf.d5 * N5 - sf.d4 * N4)
    Hy8 = -N3 + sf.e4 * N4 + sf.e5 * N5
    Hy9 = -sf.b4 * N4 - sf.b5 * N5

    return (Hy1, Hy2, Hy3, Hy4, Hy5, Hy6, Hy7, Hy8, Hy9)
end

"""
    dHxdξ(sf::DKTShapeFunctions, ξ::T, η::T) where T

Derivative of Hx with respect to ξ (area coordinate).

**BUG FIX (Issue #265):** Corrected parentheses in all derivative functions.
"""
function dHxdξ(sf::DKTShapeFunctions, ξ::T, η::T) where T
    d1 = sf.p6 * (1 - 2 * ξ) + (sf.p5 - sf.p6) * η
    d2 = sf.q6 * (1 - 2 * ξ) - (sf.q5 + sf.q6) * η
    d3 = -4 + 6 * (ξ + η) + sf.r6 * (1 - 2 * ξ) - η * (sf.r5 + sf.r6)

    d4 = -sf.p6 * (1 - 2 * ξ) + η * (sf.p4 + sf.p6)
    d5 = sf.q6 * (1 - 2 * ξ) - η * (sf.q6 - sf.q4)
    d6 = -2 + 6 * ξ + sf.r6 * (1 - 2 * ξ) + η * (sf.r4 - sf.r6)

    d7 = -η * (sf.p5 + sf.p4)
    d8 = η * (sf.q4 - sf.q5)
    d9 = -η * (sf.r5 - sf.r4)

    return (d1, d2, d3, d4, d5, d6, d7, d8, d9)
end

"""
    dHxdη(sf::DKTShapeFunctions, ξ::T, η::T) where T

Derivative of Hx with respect to η (area coordinate).

**BUG FIX (Issue #265):** Line 180 from old code had incorrect parentheses.
CORRECT: d9 = -2 + 6*η + r5*(1-2*η) + ξ*(r4-r5)
WRONG:   d9 = -2 + 6*η + r5*(1-2*η + ξ*(r4-r5))  # Old bug
"""
function dHxdη(sf::DKTShapeFunctions, ξ::T, η::T) where T
    d1 = -sf.p5 * (1 - 2 * η) - ξ * (sf.p6 - sf.p5)
    d2 = sf.q5 * (1 - 2 * η) - ξ * (sf.q5 + sf.q6)
    d3 = -4 + 6 * (ξ + η) + sf.r5 * (1 - 2 * η) - ξ * (sf.r5 + sf.r6)

    d4 = ξ * (sf.p4 + sf.p6)
    d5 = ξ * (sf.q4 - sf.q6)
    d6 = -ξ * (sf.r6 - sf.r4)

    d7 = sf.p5 * (1 - 2 * η) - ξ * (sf.p4 + sf.p5)
    d8 = sf.q5 * (1 - 2 * η) + ξ * (sf.q4 - sf.q5)
    # BUG FIX: Issue #265 - Corrected parentheses
    d9 = -2 + 6 * η + sf.r5 * (1 - 2 * η) + ξ * (sf.r4 - sf.r5)  # FIXED!

    return (d1, d2, d3, d4, d5, d6, d7, d8, d9)
end

"""
    dHydξ(sf::DKTShapeFunctions, ξ::T, η::T) where T

Derivative of Hy with respect to ξ (area coordinate).
"""
function dHydξ(sf::DKTShapeFunctions, ξ::T, η::T) where T
    d1 = sf.t6 * (1 - 2 * ξ) + η * (sf.t5 - sf.t6)
    d2 = 1 + sf.r6 * (1 - 2 * ξ) - η * (sf.r5 + sf.r6)
    d3 = -sf.q6 * (1 - 2 * ξ) + η * (sf.q5 + sf.q6)

    d4 = -sf.t6 * (1 - 2 * ξ) + η * (sf.t4 + sf.t6)
    d5 = -1 + sf.r6 * (1 - 2 * ξ) + η * (sf.r4 - sf.r6)
    d6 = -sf.q6 * (1 - 2 * ξ) - η * (sf.q4 - sf.q6)

    d7 = -η * (sf.t4 + sf.t5)
    d8 = η * (sf.r4 - sf.r5)
    d9 = -η * (sf.q4 - sf.q5)

    return (d1, d2, d3, d4, d5, d6, d7, d8, d9)
end

"""
    dHydη(sf::DKTShapeFunctions, ξ::T, η::T) where T

Derivative of Hy with respect to η (area coordinate).
"""
function dHydη(sf::DKTShapeFunctions, ξ::T, η::T) where T
    d1 = -sf.t5 * (1 - 2 * η) - ξ * (sf.t6 - sf.t5)
    d2 = 1 + sf.r5 * (1 - 2 * η) - ξ * (sf.r5 + sf.r6)
    d3 = -sf.q5 * (1 - 2 * η) + ξ * (sf.q5 + sf.q6)

    d4 = ξ * (sf.t4 + sf.t6)
    d5 = ξ * (sf.r4 - sf.r6)
    d6 = -ξ * (sf.q4 - sf.q6)

    d7 = sf.t5 * (1 - 2 * η) - ξ * (sf.t4 + sf.t5)
    d8 = -1 + sf.r5 * (1 - 2 * η) + ξ * (sf.r4 - sf.r5)
    d9 = -sf.q5 * (1 - 2 * η) - ξ * (sf.q4 - sf.q5)

    return (d1, d2, d3, d4, d5, d6, d7, d8, d9)
end

# ============================================================================
# Strain-Displacement Matrix (B-matrix)
# ============================================================================

"""
    curvature_matrix(sf::DKTShapeFunctions, X::NTuple{3, Vec{2, T}}, ξ::T, η::T) where T

Compute 3×9 strain-displacement (B) matrix relating DOFs to curvatures.

Curvatures: κ = [κx, κy, κxy]ᵀ = -[∂²w/∂x², ∂²w/∂y², 2∂²w/∂x∂y]ᵀ

In DKT, curvatures are expressed via rotations:
- κx  = ∂θy/∂x
- κy  = -∂θx/∂y
- κxy = ∂θy/∂y - ∂θx/∂x

Returns B such that κ = B * u, where u = [w1 θx1 θy1 ... w3 θx3 θy3]ᵀ
"""
function curvature_matrix!(B::AbstractMatrix{T}, sf::DKTShapeFunctions, ξ::T, η::T) where T
    # Jacobian: relates (ξ,η) derivatives to (x,y) derivatives
    # J = [∂x/∂ξ  ∂y/∂ξ]
    #     [∂x/∂η  ∂y/∂η]
    x31 = sf.x31
    x12 = sf.x12
    y31 = sf.y31
    y12 = sf.y12

    # Inverse Jacobian factor (assumes constant Jacobian for linear triangle)
    detJ_inv = 1 / (x31 * y12 - x12 * y31)

    # Get shape function derivatives in (ξ,η) coordinates
    dHx_dξ = dHxdξ(sf, ξ, η)
    dHx_dη = dHxdη(sf, ξ, η)
    dHy_dξ = dHydξ(sf, ξ, η)
    dHy_dη = dHydη(sf, ξ, η)

    # Transform to (x,y) derivatives using chain rule
    # ∂/∂x = (∂ξ/∂x)∂/∂ξ + (∂η/∂x)∂/∂η = detJ_inv * (y12*∂/∂ξ - y31*∂/∂η)
    # ∂/∂y = (∂ξ/∂y)∂/∂ξ + (∂η/∂y)∂/∂η = detJ_inv * (-x12*∂/∂ξ + x31*∂/∂η)

    # Curvature-displacement matrix (3×9)
    for i in 1:9
        # ∂θy/∂x (for κx)
        B[1, i] = detJ_inv * (y12 * dHy_dξ[i] - y31 * dHy_dη[i])

        # -∂θx/∂y (for κy)
        B[2, i] = detJ_inv * (x12 * dHx_dξ[i] - x31 * dHx_dη[i])

        # ∂θy/∂y - ∂θx/∂x (for κxy)
        dθy_dy = detJ_inv * (-x12 * dHy_dξ[i] + x31 * dHy_dη[i])
        dθx_dx = detJ_inv * (y12 * dHx_dξ[i] - y31 * dHx_dη[i])
        B[3, i] = dθy_dy - dθx_dx
    end

    return B
end

function curvature_matrix(sf::DKTShapeFunctions, X::NTuple{3,Vec{2,T}}, ξ::T, η::T) where T
    B = zeros(T, 3, 9)
    curvature_matrix!(B, sf, ξ, η)
    return B
end

# ============================================================================
# Element Stiffness Matrix
# ============================================================================

"""
    element_stiffness_matrix(X::NTuple{3, Vec{2, T}}, formulation::DKTFormulation, material::AbstractMaterial) where T

Compute 9×9 element stiffness matrix for DKT element.

Uses 3-point Gauss quadrature (exact for DKT).

# Arguments
- `X`: Tuple of 3 nodal coordinates
- `formulation`: DKT formulation with thickness
- `material`: Material with E and ν properties (e.g., LinearElastic)

# Returns
- `Matrix{T}`: 9×9 stiffness matrix
"""
function element_stiffness_matrix!(
    Ke::AbstractMatrix{T},
    B::AbstractMatrix{T},
    DB::AbstractMatrix{T},
    X::NTuple{3,Vec{2,T}},
    sf::DKTShapeFunctions,
    D_mat::AbstractMatrix{T},
    gauss_points::NTuple{3,Tuple{Float64,Float64}},
    gauss_weight::Float64) where T
    area = triangle_area(X)
    fill!(Ke, zero(T))
    for (ξ, η) in gauss_points
        curvature_matrix!(B, sf, ξ, η)
        mul!(DB, D_mat, B)
        mul!(Ke, adjoint(B), DB, gauss_weight * area, one(T))
    end
    return Ke
end

function element_stiffness_matrix(X::NTuple{3,Vec{2,T}}, formulation::DKTFormulation, material::AbstractMaterial) where T
    sf = DKTShapeFunctions(X)
    D_mat = bending_matrix(material, formulation.thickness)
    Ke = zeros(Float64, DKT_ELEMENT_DOF, DKT_ELEMENT_DOF)
    B = zeros(Float64, 3, DKT_ELEMENT_DOF)
    DB = similar(B)
    element_stiffness_matrix!(Ke, B, DB, X, sf, D_mat, DKT_GAUSS_POINTS, DKT_GAUSS_WEIGHT)
    return Ke
end

# ============================================================================
# Assembly into Global System
# ============================================================================

struct DKTAssemblyCache
    K_csc::SparseMatrixCSC{Float64,Int}
    sorteddofs::Vector{Int}
    permutation::Vector{Int}
    gdofs::Vector{Int}
    Ke::Matrix{Float64}
    B::Matrix{Float64}
    DB::Matrix{Float64}
    X_buffer::Vector{Vec{2,Float64}}
    elements::Vector{UInt32}
    D_mat::Matrix{Float64}
    f::Vector{Float64}
end

function DKTAssemblyCache(physics::Physics{DKTFormulation,PlateDisplacement,M,Mat}) where {M<:AbstractMesh,Mat<:AbstractMaterial}
    mesh = physics.mesh
    @assert typeof(mesh).parameters[1] == DKT_NNODE "DKT requires 3-node triangles"

    I, J, ndofs_total = build_plate_sparsity_pattern(mesh)
    K_csc = sparse(I, J, ones(length(I)), ndofs_total, ndofs_total)
    fill!(K_csc.nzval, 0.0)

    ndofs_elem = DKT_ELEMENT_DOF

    sorteddofs = Vector{Int}(undef, ndofs_elem)
    permutation = Vector{Int}(undef, ndofs_elem)
    gdofs = Vector{Int}(undef, ndofs_elem)

    Ke = zeros(Float64, ndofs_elem, ndofs_elem)
    B = zeros(Float64, 3, ndofs_elem)
    DB = similar(B)
    X_buffer = [Vec{2}((0.0, 0.0)) for _ in 1:DKT_NNODE]

    elements = plate_element_indices(mesh, physics.element_set)
    D_mat = bending_matrix(physics.material, physics.formulation.thickness)
    f = zeros(ndofs_total)

    return DKTAssemblyCache(K_csc, sorteddofs, permutation, gdofs, Ke, B, DB, X_buffer, elements, D_mat, f)
end

function apply_neumann_forces!(f::Vector{Float64}, bc_neumann, nnodes::Int)
    if hasfield(typeof(bc_neumann), :surface_ids) && hasfield(typeof(bc_neumann), :values)
        surface_ids = getfield(bc_neumann, :surface_ids)
        values = getfield(bc_neumann, :values)
        for (node, force) in zip(surface_ids, values)
            if node <= nnodes
                @inbounds for α in 1:min(DKT_DOF_PER_NODE, length(force))
                    f[DKT_DOF_PER_NODE * (node - 1) + α] += force[α]
                end
            end
        end
    end
end

function apply_dirichlet_constraints!(K::SparseMatrixCSC{Float64,Int}, f::Vector{Float64}, bc_dirichlet, ndofs::Int)
    node_ids = getfield(bc_dirichlet, :node_ids)
    components = getfield(bc_dirichlet, :components)
    values = getfield(bc_dirichlet, :values)

    for i in 1:length(node_ids)
        node = node_ids[i]
        comp_list = components[i]
        val_entry = values[i]
        for comp in comp_list
            dof = DKT_DOF_PER_NODE * (node - 1) + comp
            if 1 <= dof <= ndofs
                K[dof, :] .= 0.0
                K[:, dof] .= 0.0
                K[dof, dof] = 1.0
                value = val_entry isa AbstractVector ? val_entry[comp] : val_entry
                f[dof] = value
            end
        end
    end
end


function assemble_elements_dkt!(
    cache::DKTAssemblyCache,
    physics::Physics{DKTFormulation,PlateDisplacement,M,Mat}) where {M<:AbstractMesh,Mat<:AbstractMaterial}
    mesh = physics.mesh
    ndofs_elem = DKT_ELEMENT_DOF
    fill!(cache.K_csc.nzval, 0.0)

    for elem_id in cache.elements
        conn = mesh.connectivity[Int(elem_id)]
        fill_coordinate_buffer!(cache.X_buffer, mesh.nodes, conn)
        X = (cache.X_buffer[1], cache.X_buffer[2], cache.X_buffer[3])
        sf = DKTShapeFunctions(X)
        element_stiffness_matrix!(cache.Ke, cache.B, cache.DB, X, sf, cache.D_mat, DKT_GAUSS_POINTS, DKT_GAUSS_WEIGHT)
        plate_global_dofs!(cache.gdofs, conn)

        sortperm!(cache.permutation, cache.gdofs)
        @inbounds for k in 1:ndofs_elem
            cache.sorteddofs[k] = cache.gdofs[cache.permutation[k]]
        end

        for i_local in 1:ndofs_elem
            i_global = cache.sorteddofs[i_local]
            col_start = cache.K_csc.colptr[i_global]
            col_end = cache.K_csc.colptr[i_global + 1] - 1
            Ri = col_start
            for ri in 1:ndofs_elem
                row_sorted = cache.sorteddofs[ri]
                while Ri <= col_end && cache.K_csc.rowval[Ri] < row_sorted
                    Ri += 1
                end
                if Ri <= col_end && cache.K_csc.rowval[Ri] == row_sorted
                    orig_row = cache.permutation[ri]
                    orig_col = cache.permutation[i_local]
                    cache.K_csc.nzval[Ri] += cache.Ke[orig_row, orig_col]
                end
            end
        end
    end

    return cache
end

function assemble!(physics::Physics{DKTFormulation,PlateDisplacement,M,Mat}) where {M<:AbstractMesh,Mat<:AbstractMaterial}
    cache = DKTAssemblyCache(physics)
    return _assemble_dkt!(physics, cache)
end

function _assemble_dkt!(
    physics::Physics{DKTFormulation,PlateDisplacement,M,Mat},
    cache::DKTAssemblyCache) where {M<:AbstractMesh,Mat<:AbstractMaterial}

    fill!(cache.f, 0.0)
    assemble_elements_dkt!(cache, physics)

    nnodes = length(physics.mesh.nodes)
    ndofs = DKT_DOF_PER_NODE * nnodes

    apply_neumann_forces!(cache.f, physics.bc_neumann, nnodes)

    K = copy(cache.K_csc)
    apply_dirichlet_constraints!(K, cache.f, physics.bc_dirichlet, ndofs)

    return (K, cache.f)
end

# Export functions
export DKTShapeFunctions
export element_stiffness_matrix
export curvature_matrix
