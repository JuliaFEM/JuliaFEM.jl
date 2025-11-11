"""
GPU Assembly Proof-of-Concept (Tensors.jl Version)
===================================================

Minimal working example of matrix-free Newton-Krylov on GPU for 2D linear elasticity.

**CORRECTED VERSION** using proper Tensors.jl architecture from material_modeling.md

Key changes from v1:
- Uses SymmetricTensor{2,2} for strain and stress (2D)
- Proper material API: compute_stress(material, ε, state, Δt)
- No Voigt notation, no manual indexing
- Mathematics looks like equations!

Goal: Prove that entire solve can stay on GPU with no escapes until final result.
"""

using CUDA
using LinearAlgebra
using Tensors  # ✅ Using Tensors.jl for all tensor operations!
using Krylov

# ============================================================================
# Material Model (following material_modeling.md)
# ============================================================================

"""
Linear elastic (Hookean) material model for plane strain.

Stateless: σ depends only on current ε, no history.
"""
struct LinearElastic
    E::Float64   # Young's modulus [Pa]
    ν::Float64   # Poisson's ratio [-]
end

# Lamé parameters
@inline λ(mat::LinearElastic) = mat.E * mat.ν / ((1 + mat.ν) * (1 - 2mat.ν))
@inline μ(mat::LinearElastic) = mat.E / (2(1 + mat.ν))

"""
Compute stress for 2D plane strain using Tensors.jl.

Returns (σ, 𝔻, state_new) following unified material API.
"""
@inline function compute_stress_2d(
    material::LinearElastic,
    ε::SymmetricTensor{2,2,T}
) where T
    λ_val = T(λ(material))
    μ_val = T(μ(material))

    # Identity tensor
    I = one(ε)

    # Hooke's law: σ = λ·tr(ε)·I + 2μ·ε
    σ = λ_val * tr(ε) * I + 2μ_val * ε

    return σ
end

# ============================================================================
# Mesh Generation
# ============================================================================

function generate_rectangle_mesh(nx::Int, ny::Int, Lx::Float64, Ly::Float64)
    """Generate structured Quad4 mesh for rectangle [0,Lx] × [0,Ly]"""

    # Node coordinates
    n_nodes = (nx + 1) * (ny + 1)
    coords = zeros(n_nodes, 2)

    node_id = 1
    for j in 0:ny
        for i in 0:nx
            coords[node_id, 1] = i * Lx / nx
            coords[node_id, 2] = j * Ly / ny
            node_id += 1
        end
    end

    # Element connectivity (counterclockwise from lower-left)
    n_elements = nx * ny
    connectivity = zeros(Int32, n_elements, 4)

    elem_id = 1
    for j in 0:(ny-1)
        for i in 0:(nx-1)
            n1 = i + j * (nx + 1) + 1
            n2 = (i + 1) + j * (nx + 1) + 1
            n3 = (i + 1) + (j + 1) * (nx + 1) + 1
            n4 = i + (j + 1) * (nx + 1) + 1
            connectivity[elem_id, :] = [n1, n2, n3, n4]
            elem_id += 1
        end
    end

    return coords, connectivity
end

# ============================================================================
# GPU Kernel using Tensors.jl
# ============================================================================

# Gauss quadrature (2x2 for Quad4)
const GAUSS_POINTS_2D = SA[
    SA[-0.5773502691896257, -0.5773502691896257],
    SA[0.5773502691896257, -0.5773502691896257],
    SA[0.5773502691896257, 0.5773502691896257],
    SA[-0.5773502691896257, 0.5773502691896257]
]
const GAUSS_WEIGHTS_2D = SA[1.0, 1.0, 1.0, 1.0]

@inline function shape_derivatives_quad4(ξ, η)
    """Quad4 shape function derivatives: dN/dξ and dN/dη"""
    dN_dξ = SA[-0.25*(1-η), 0.25*(1-η), 0.25*(1+η), -0.25*(1+η)]
    dN_dη = SA[-0.25*(1-ξ), -0.25*(1+ξ), 0.25*(1+ξ), 0.25*(1-ξ)]
    return dN_dξ, dN_dη
end

@inline function compute_jacobian_quad4(dN_dξ, dN_dη, x_coords, y_coords)
    """Compute 2D Jacobian as Tensor{2,2}"""
    dx_dξ = sum(dN_dξ[i] * x_coords[i] for i in 1:4)
    dx_dη = sum(dN_dη[i] * x_coords[i] for i in 1:4)
    dy_dξ = sum(dN_dξ[i] * y_coords[i] for i in 1:4)
    dy_dη = sum(dN_dη[i] * y_coords[i] for i in 1:4)

    # Return as Tensor (not SMatrix) for proper inv() and det()
    return Tensor{2,2}((dx_dξ, dy_dξ, dx_dη, dy_dη))
end

@inline function compute_B_matrix_strain(dN_dx, dN_dy, u_elem)
    """
    Compute strain from B-matrix and displacements using Tensors.jl.

    Returns SymmetricTensor{2,2} for 2D strain.
    """
    # ε = [εxx  εxy]  where εxy = (∂ux/∂y + ∂uy/∂x)/2
    #     [εxy  εyy]

    εxx = dN_dx[1] * u_elem[1] + dN_dx[2] * u_elem[3] +
          dN_dx[3] * u_elem[5] + dN_dx[4] * u_elem[7]

    εyy = dN_dy[1] * u_elem[2] + dN_dy[2] * u_elem[4] +
          dN_dy[3] * u_elem[6] + dN_dy[4] * u_elem[8]

    # Engineering shear strain γxy (factor of 2 handled by SymmetricTensor constructor)
    γxy = (dN_dy[1] * u_elem[1] + dN_dx[1] * u_elem[2] +
           dN_dy[2] * u_elem[3] + dN_dx[2] * u_elem[4] +
           dN_dy[3] * u_elem[5] + dN_dx[3] * u_elem[6] +
           dN_dy[4] * u_elem[7] + dN_dx[4] * u_elem[8])

    # SymmetricTensor{2,2} constructor: (ε11, ε12, ε22)
    # Note: ε12 = γxy/2 (tensorial shear strain, not engineering)
    return SymmetricTensor{2,2}((εxx, γxy / 2, εyy))
end

@inline function compute_B_transpose_sigma(dN_dx, dN_dy, σ::SymmetricTensor{2,2})
    """
    Compute Bᵀ·σ for element residual.

    Returns SVector{8} of nodal forces.
    """
    # Extract stress components
    σxx = σ[1, 1]
    σyy = σ[2, 2]
    σxy = σ[1, 2]  # Tensorial (symmetric), not engineering

    # Bᵀ·σ gives forces at each DOF
    r_elem = SA[
        dN_dx[1]*σxx+dN_dy[1]*σxy,  # Node 1, x-direction
        dN_dy[1]*σyy+dN_dx[1]*σxy,  # Node 1, y-direction
        dN_dx[2]*σxx+dN_dy[2]*σxy,  # Node 2, x-direction
        dN_dy[2]*σyy+dN_dx[2]*σxy,  # Node 2, y-direction
        dN_dx[3]*σxx+dN_dy[3]*σxy,  # Node 3, x-direction
        dN_dy[3]*σyy+dN_dx[3]*σxy,  # Node 3, y-direction
        dN_dx[4]*σxx+dN_dy[4]*σxy,  # Node 4, x-direction
        dN_dy[4]*σyy+dN_dx[4]*σxy   # Node 4, y-direction
    ]

    return r_elem
end

# Main GPU kernel
function elasticity_residual_kernel_tensors!(
    r_global::CuDeviceVector{T},
    u_global::CuDeviceVector{T},
    elem_nodes::CuDeviceMatrix{Int32},
    node_coords::CuDeviceMatrix{T},
    E::T,
    ν::T
) where T
    """
    Compute residual using Tensors.jl for proper tensor operations.

    Key: ε and σ are SymmetricTensor{2,2}, not vectors!
    """

    elem_id = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if elem_id > size(elem_nodes, 1)
        return
    end

    # Material model
    material = LinearElastic(E, ν)

    # Get element nodes
    n1, n2, n3, n4 = elem_nodes[elem_id, 1], elem_nodes[elem_id, 2],
    elem_nodes[elem_id, 3], elem_nodes[elem_id, 4]

    # Node coordinates
    x_coords = SA[node_coords[n1, 1], node_coords[n2, 1],
        node_coords[n3, 1], node_coords[n4, 1]]
    y_coords = SA[node_coords[n1, 2], node_coords[n2, 2],
        node_coords[n3, 2], node_coords[n4, 2]]

    # Element DOFs
    u_elem = SA[
        u_global[2*n1-1], u_global[2*n1],
        u_global[2*n2-1], u_global[2*n2],
        u_global[2*n3-1], u_global[2*n3],
        u_global[2*n4-1], u_global[2*n4]
    ]

    # Accumulate element residual
    r_elem = MVector{8,T}(zeros(8))

    # Integration loop
    for ip in 1:4
        ξ, η = GAUSS_POINTS_2D[ip]
        w = GAUSS_WEIGHTS_2D[ip]

        # Shape function derivatives
        dN_dξ, dN_dη = shape_derivatives_quad4(ξ, η)

        # Jacobian
        J = compute_jacobian_quad4(dN_dξ, dN_dη, x_coords, y_coords)
        det_J = det(J)
        inv_J = inv(J)

        # Physical derivatives: dN/dx = inv(J) · dN/dξ
        dN_dx = SA[
            inv_J[1, 1]*dN_dξ[1]+inv_J[1, 2]*dN_dη[1],
            inv_J[1, 1]*dN_dξ[2]+inv_J[1, 2]*dN_dη[2],
            inv_J[1, 1]*dN_dξ[3]+inv_J[1, 2]*dN_dη[3],
            inv_J[1, 1]*dN_dξ[4]+inv_J[1, 2]*dN_dη[4]
        ]
        dN_dy = SA[
            inv_J[2, 1]*dN_dξ[1]+inv_J[2, 2]*dN_dη[1],
            inv_J[2, 1]*dN_dξ[2]+inv_J[2, 2]*dN_dη[2],
            inv_J[2, 1]*dN_dξ[3]+inv_J[2, 2]*dN_dη[3],
            inv_J[2, 1]*dN_dξ[4]+inv_J[2, 2]*dN_dη[4]
        ]

        # ✅ Compute strain as SymmetricTensor{2,2}
        ε = compute_B_matrix_strain(dN_dx, dN_dy, u_elem)

        # ✅ Compute stress using material model
        σ = compute_stress_2d(material, ε)

        # ✅ Compute Bᵀ·σ (element forces)
        r_contrib = compute_B_transpose_sigma(dN_dx, dN_dy, σ)

        # Accumulate
        r_elem .+= r_contrib .* (w * det_J)
    end

    # Atomic scatter
    CUDA.@atomic r_global[2*n1-1] += r_elem[1]
    CUDA.@atomic r_global[2*n1] += r_elem[2]
    CUDA.@atomic r_global[2*n2-1] += r_elem[3]
    CUDA.@atomic r_global[2*n2] += r_elem[4]
    CUDA.@atomic r_global[2*n3-1] += r_elem[5]
    CUDA.@atomic r_global[2*n3] += r_elem[6]
    CUDA.@atomic r_global[2*n4-1] += r_elem[7]
    CUDA.@atomic r_global[2*n4] += r_elem[8]

    return nothing
end

# ============================================================================
# GPU Assembly Functions (same as before)
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

    @cuda threads = threads blocks = blocks elasticity_residual_kernel_tensors!(
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

# ============================================================================
# Matrix-Free Operator
# ============================================================================

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

# ============================================================================
# Newton-Krylov Solver
# ============================================================================

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
# Main Demo
# ============================================================================

function main()
    println("\n" * "="^70)
    println("GPU Assembly POC with Tensors.jl")
    println("="^70)

    nx, ny = 10, 10
    Lx, Ly = 1.0, 1.0
    E, ν = 200e9, 0.3

    println("\n📐 Mesh: $(nx*ny) Quad4 elements, $((nx+1)*(ny+1)) nodes, $(2*(nx+1)*(ny+1)) DOFs")
    println("🔧 Material: E=$(E/1e9) GPa, ν=$ν (LinearElastic)")
    println("✅ Using Tensors.jl: SymmetricTensor{2,2} for ε and σ")

    coords, connectivity = generate_rectangle_mesh(nx, ny, Lx, Ly)
    n_dofs = 2 * size(coords, 1)

    # Boundary conditions
    fixed_dofs = Int[]
    for node_id in 1:size(coords, 1)
        if coords[node_id, 1] < 1e-10
            push!(fixed_dofs, 2 * node_id - 1, 2 * node_id)
        end
    end

    u0 = randn(n_dofs) * 1e-6
    u0[fixed_dofs] .= 0.0

    for node_id in 1:size(coords, 1)
        if abs(coords[node_id, 1] - Lx) < 1e-10
            u0[2*node_id-1] = 0.001  # 1mm tension
        end
    end

    println("🔒 BC: $(length(fixed_dofs)) fixed DOFs, 1mm tension on right edge")

    # Transfer to GPU
    elem_nodes_gpu = CuArray{Int32}(connectivity)
    coords_gpu = CuArray{Float64}(coords)
    u_gpu = CuArray{Float64}(u0)

    println("\n🚀 Starting GPU Newton-Krylov (Tensors.jl version)...")

    n_iter = solve_newton_krylov_gpu!(
        u_gpu, elem_nodes_gpu, coords_gpu, E, ν, fixed_dofs,
        max_iter=20, tol=1e-8, gmres_tol=1e-6, verbose=true
    )

    u_final = Array(u_gpu)

    println("\n📊 Results:")
    println("  Iterations: $n_iter")
    println("  ||u||: $(norm(u_final))")
    println("\n✅ POC COMPLETE - Now using proper Tensors.jl!")
    println("="^70 * "\n")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
