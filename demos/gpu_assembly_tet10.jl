"""
GPU Assembly for Tet10 (Quadratic Tetrahedron)
==============================================

The workhorse element for 3D real simulations!

Following JuliaFEM's established pattern:
- Loop through shape functions (i = 1:nnodes)
- Fill 3x3 blocks directly from derivatives dN[1:3, i]
- No "B-matrix" concept - just derivatives!
- Tensors.jl for strain/stress (SymmetricTensor{2,3})
"""

using CUDA
using LinearAlgebra
using Tensors
using Krylov

# ============================================================================
# Material Model
# ============================================================================

struct LinearElastic
    E::Float64
    ν::Float64
end

@inline λ(mat::LinearElastic) = mat.E * mat.ν / ((1 + mat.ν) * (1 - 2mat.ν))
@inline μ(mat::LinearElastic) = mat.E / (2(1 + mat.ν))

"""
3D linear elastic stress: sigma = lambda*tr(eps)*I + 2*mu*eps
"""
@inline function compute_stress_3d(
    material::LinearElastic,
    eps::SymmetricTensor{2,3,T}
) where T
    lambda_val = T(λ(material))
    mu_val = T(μ(material))
    I = one(eps)
    sigma = lambda_val * tr(eps) * I + 2 * mu_val * eps
    return sigma
end

# ============================================================================
# Tet10 Reference Element (Quadratic Tetrahedron)
# ============================================================================

"""
Tet10 node numbering (ABAQUS convention):
  Vertices: 1-4
  Edge midpoints: 5(1-2), 6(2-3), 7(1-3), 8(1-4), 9(2-4), 10(3-4)
"""
const TET10_REF_COORDS = (
    Vec{3}((0.0, 0.0, 0.0)),  # 1
    Vec{3}((1.0, 0.0, 0.0)),  # 2
    Vec{3}((0.0, 1.0, 0.0)),  # 3
    Vec{3}((0.0, 0.0, 1.0)),  # 4
    Vec{3}((0.5, 0.0, 0.0)),  # 5 (1-2)
    Vec{3}((0.5, 0.5, 0.0)),  # 6 (2-3)
    Vec{3}((0.0, 0.5, 0.0)),  # 7 (1-3)
    Vec{3}((0.0, 0.0, 0.5)),  # 8 (1-4)
    Vec{3}((0.5, 0.0, 0.5)),  # 9 (2-4)
    Vec{3}((0.0, 0.5, 0.5))   # 10 (3-4)
)

"""
Gauss quadrature for Tet10: 4-point scheme
"""
const GAUSS_TET4 = (
    (Vec{3}((0.5854101966249685, 0.1381966011250105, 0.1381966011250105)), 0.25),
    (Vec{3}((0.1381966011250105, 0.5854101966249685, 0.1381966011250105)), 0.25),
    (Vec{3}((0.1381966011250105, 0.1381966011250105, 0.5854101966249685)), 0.25),
    (Vec{3}((0.1381966011250105, 0.1381966011250105, 0.1381966011250105)), 0.25)
)

@inline function tet10_shape_derivatives(xi, eta, zeta)
    """
    Quadratic shape function derivatives for Tet10.
    Returns tuple of 10 Vec{3} (using Tensors.jl).
    """
    lambda = 1 - xi - eta - zeta

    # Vertex nodes (1-4)
    dN1 = Vec{3}((4 * lambda - 1, 4 * lambda - 1, 4 * lambda - 1))
    dN2 = Vec{3}((4 * xi - 1, 0.0, 0.0))
    dN3 = Vec{3}((0.0, 4 * eta - 1, 0.0))
    dN4 = Vec{3}((0.0, 0.0, 4 * zeta - 1))

    # Edge midpoints (5-10)
    dN5 = Vec{3}((4 * (1 - 2 * xi - eta - zeta), -4 * xi, -4 * xi))
    dN6 = Vec{3}((4 * eta, 4 * xi, 0.0))
    dN7 = Vec{3}((-4 * eta, 4 * (1 - xi - 2 * eta - zeta), -4 * eta))
    dN8 = Vec{3}((-4 * zeta, -4 * zeta, 4 * (1 - xi - eta - 2 * zeta)))
    dN9 = Vec{3}((4 * zeta, 0.0, 4 * xi))
    dN10 = Vec{3}((0.0, 4 * zeta, 4 * eta))

    return (dN1, dN2, dN3, dN4, dN5, dN6, dN7, dN8, dN9, dN10)
end

@inline function compute_jacobian_tet10(dN_dxi::NTuple{10,Vec{3,T}}, X::NTuple{10,Vec{3,T}}) where T
    """
    Jacobian using tensor products: J = Σ_i dN_i ⊗ X_i
    """
    return sum(dN_dxi[i] ⊗ X[i] for i in 1:10)
end

# ============================================================================
# JuliaFEM Pattern: Loop Through Shape Functions, Fill 3x3 Blocks
# ============================================================================

@inline function compute_strain_from_displacements(
    dN_dx::NTuple{10,Vec{3,T}},
    u::NTuple{10,Vec{3,T}}
) where T
    """
    Compute strain using Tensors.jl tensor products.

    Displacement gradient: ∇u = Σ_i dN_i ⊗ u_i
    Strain (small): ε = 1/2 (∇u + ∇uᵀ) = sym(∇u)
    """
    # Displacement gradient via tensor products
    gradu = sum(dN_dx[i] ⊗ u[i] for i in 1:10)

    # Symmetric part (strain)
    return symmetric(gradu)
end

@inline function compute_nodal_forces_from_stress(
    dN_dx::NTuple{10,Vec{3,T}},
    sigma::SymmetricTensor{2,3,T}
) where T
    """
    Compute nodal forces using Tensors.jl.

    Force at node i: f_i = dN_i · σ
    """
    return ntuple(i -> dN_dx[i] ⋅ sigma, Val(10))
end

# ============================================================================
# GPU Kernel
# ============================================================================

function tet10_residual_kernel!(
    r_global::CuDeviceVector{T},
    u_global::CuDeviceVector{T},
    elem_nodes::CuDeviceMatrix{Int32},
    node_coords::CuDeviceMatrix{T},
    E::T,
    ν::T
) where T
    """
    Element-parallel GPU kernel for Tet10 elasticity.

    Pattern:
    1. Get 10 node coordinates and 30 DOFs
    2. Loop over 4 Gauss points
    3. Compute shape derivatives dN/dξ
    4. Compute Jacobian and physical derivatives dN/dx
    5. Compute strain from derivatives (no B-matrix!)
    6. Compute stress from material model
    7. Compute nodal forces from stress (loop through shape functions)
    8. Atomic scatter to global residual
    """

    elem_id = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if elem_id > size(elem_nodes, 1)
        return
    end

    material = LinearElastic(E, ν)

    # Get element nodes (1-indexed to 10 nodes)
    nodes = ntuple(i -> elem_nodes[elem_id, i], Val(10))

    # Element coordinates (10 nodes × 3 coordinates)
    X = ntuple(Val(10)) do i
        Vec{3}((node_coords[nodes[i], 1],
            node_coords[nodes[i], 2],
            node_coords[nodes[i], 3]))
    end

    # Element displacements (10 nodes × 3 DOFs)
    u = ntuple(Val(10)) do i
        Vec{3}((u_global[3*nodes[i]-2],
            u_global[3*nodes[i]-1],
            u_global[3*nodes[i]]))
    end

    # Accumulate element residual (10 forces as Vec{3})
    r_elem = [zero(Vec{3,T}) for _ in 1:10]

    # Integration loop (4 Gauss points)
    for (xez_tuple, w) in GAUSS_TET4
        xi, eta, zeta = xez_tuple[1], xez_tuple[2], xez_tuple[3]

        # Shape function derivatives in reference coordinates
        dN_dxi = tet10_shape_derivatives(xi, eta, zeta)

        # Jacobian using tensor products: J = Σ dN_i ⊗ X_i
        J = compute_jacobian_tet10(dN_dxi, X)
        detJ = det(J)
        invJ = inv(J)

        # Physical derivatives: dN_dx = invJ · dN_dxi (tensor contraction)
        dN_dx = ntuple(i -> invJ ⋅ dN_dxi[i], Val(10))

        # Compute strain: ε = sym(∇u) where ∇u = Σ dN_i ⊗ u_i
        eps = compute_strain_from_displacements(dN_dx, u)

        # Compute stress
        sigma = compute_stress_3d(material, eps)

        # Compute nodal forces: f_i = dN_i · σ
        f_contrib = compute_nodal_forces_from_stress(dN_dx, sigma)

        # Accumulate with quadrature weight
        for i in 1:10
            r_elem[i] += f_contrib[i] * (w * detJ)
        end
    end

    # Atomic scatter (10 nodes × 3 components)
    for i in 1:10
        CUDA.@atomic r_global[3*nodes[i]-2] += r_elem[i][1]
        CUDA.@atomic r_global[3*nodes[i]-1] += r_elem[i][2]
        CUDA.@atomic r_global[3*nodes[i]] += r_elem[i][3]
    end

    return nothing
end

# ============================================================================
# Assembly & Solver Wrappers
# ============================================================================

function compute_residual_gpu!(
    r_gpu::CuVector{T},
    u_gpu::CuVector{T},
    elem_nodes_gpu::CuMatrix{Int32},
    coords_gpu::CuMatrix{T},
    E::T,
    ν::T
) where T
    n_elements = size(elem_nodes_gpu, 1)
    threads = 256
    blocks = cld(n_elements, threads)

    fill!(r_gpu, zero(T))

    @cuda threads = threads blocks = blocks tet10_residual_kernel!(
        r_gpu, u_gpu, elem_nodes_gpu, coords_gpu, E, ν
    )
    CUDA.synchronize()

    return nothing
end

function compute_Jv_gpu!(
    Jv_gpu::CuVector{T},
    u_gpu::CuVector{T},
    v_gpu::CuVector{T},
    r0_gpu::CuVector{T},
    elem_nodes_gpu::CuMatrix{Int32},
    coords_gpu::CuMatrix{T},
    E::T,
    ν::T,
    ε::T=T(1e-7)
) where T
    u_perturbed = u_gpu .+ ε .* v_gpu
    r_perturbed = CUDA.zeros(T, length(u_gpu))
    compute_residual_gpu!(r_perturbed, u_perturbed, elem_nodes_gpu, coords_gpu, E, ν)
    Jv_gpu .= (r_perturbed .- r0_gpu) ./ ε
    return nothing
end

struct GPUMatrixFreeOperator{T}
    u::CuVector{T}
    r0::CuVector{T}
    elem_nodes::CuMatrix{Int32}
    coords::CuMatrix{T}
    E::T
    ν::T
    n::Int
end

Base.size(op::GPUMatrixFreeOperator) = (op.n, op.n)

function LinearAlgebra.mul!(Jv, op::GPUMatrixFreeOperator{T}, v) where T
    v_gpu = CuVector{T}(v)
    Jv_gpu = CuVector{T}(undef, length(v))
    compute_Jv_gpu!(Jv_gpu, op.u, v_gpu, op.r0, op.elem_nodes, op.coords, op.E, op.ν)
    copyto!(Jv, Array(Jv_gpu))
    return Jv
end

function solve_newton_krylov_gpu!(
    u_gpu::CuVector{T},
    elem_nodes_gpu::CuMatrix{Int32},
    coords_gpu::CuMatrix{T},
    E::T,
    ν::T,
    fixed_dofs::Vector{Int};
    max_iter::Int=20,
    tol::T=T(1e-8),
    gmres_tol::T=T(1e-6),
    verbose::Bool=true
) where T
    n_dofs = length(u_gpu)
    r_gpu = CUDA.zeros(T, n_dofs)

    for iter in 1:max_iter
        compute_residual_gpu!(r_gpu, u_gpu, elem_nodes_gpu, coords_gpu, E, ν)

        # Enforce BC
        r_cpu_temp = Array(r_gpu)
        r_cpu_temp[fixed_dofs] .= 0.0
        copyto!(r_gpu, r_cpu_temp)

        r_norm = CUDA.norm(r_gpu)

        if verbose
            println("  Newton iter $iter: ||r|| = $r_norm")
        end

        if r_norm < tol
            if verbose
                println("  ✅ Converged in $iter iterations")
            end
            return iter
        end

        op = GPUMatrixFreeOperator(u_gpu, r_gpu, elem_nodes_gpu, coords_gpu, E, ν, n_dofs)
        r_cpu = Array(-r_gpu)
        du_cpu, stats = gmres(op, r_cpu, atol=gmres_tol, rtol=0.0, verbose=0)

        if !stats.solved
            @warn "GMRES did not converge at iteration $iter"
        end

        du_cpu[fixed_dofs] .= 0.0
        du_gpu = CuVector{T}(du_cpu)
        u_gpu .+= du_gpu
    end

    @warn "Newton did not converge in $max_iter iterations"
    return max_iter
end

# ============================================================================
# Test Mesh Generation
# ============================================================================

function generate_single_tet10_mesh()
    """
    Single Tet10 element for testing.
    """
    # 10 nodes: 4 vertices + 6 edge midpoints
    coords = [
        0.0 0.0 0.0;
        1.0 0.0 0.0;
        0.0 1.0 0.0;
        0.0 0.0 1.0;
        0.5 0.0 0.0;
        0.5 0.5 0.0;
        0.0 0.5 0.0;
        0.0 0.0 0.5;
        0.5 0.0 0.5;
        0.0 0.5 0.5
    ]

    connectivity = reshape(Int32[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 1, 10)

    return coords, connectivity
end

# ============================================================================
# Main Demo
# ============================================================================

function main()
    println("\n" * "="^70)
    println("GPU Assembly: Tet10 (Quadratic Tetrahedron)")
    println("="^70)

    E, ν = 200e9, 0.3

    coords, connectivity = generate_single_tet10_mesh()
    n_dofs = 3 * size(coords, 1)

    println("\n📐 Mesh: 1 Tet10 element, 10 nodes, 30 DOFs")
    println("🔧 Material: E=$(E/1e9) GPa, ν=$ν")
    println("✅ Pattern: Loop through shape functions, fill 3x3 blocks")
    println("✅ No B-matrix concept - just derivatives!")

    # Boundary conditions: Fix node 1 (DOFs 1,2,3)
    fixed_dofs = [1, 2, 3]

    # Initial displacement: Small perturbation
    u0 = randn(n_dofs) * 1e-6
    u0[fixed_dofs] .= 0.0
    u0[4] = 0.001  # Pull node 2 in x-direction

    println("🔒 BC: Node 1 fixed, node 2 displaced 1mm in x")

    # Transfer to GPU
    elem_nodes_gpu = CuArray{Int32}(connectivity)
    coords_gpu = CuArray{Float64}(coords)
    u_gpu = CuArray{Float64}(u0)

    println("\n🚀 Starting GPU Newton-Krylov (Tet10)...")

    n_iter = solve_newton_krylov_gpu!(
        u_gpu, elem_nodes_gpu, coords_gpu, E, ν, fixed_dofs,
        max_iter=20, tol=1e-8, gmres_tol=1e-6, verbose=true
    )

    u_final = Array(u_gpu)

    println("\n📊 Results:")
    println("  Iterations: $n_iter")
    println("  ||u||: $(norm(u_final))")
    println("\n✅ TET10 POC COMPLETE!")
    println("="^70 * "\n")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
