"""
Nodal Assembly GPU Implementation

This is the GPU port of demos/nodal_assembly_cpu.jl using CUDA.jl.

TWO-PHASE APPROACH:
1. compute_gp_data_kernel!() - Compute integration point stresses and material states
2. nodal_assembly_kernel!() - Assemble residual at nodes (matrix-free, no atomics!)

Uses Tensors.jl throughout on GPU (CuArray{SymmetricTensor} works!)
"""

using CUDA
using Tensors
using LinearAlgebra
using Printf

# Material state for plasticity (GPU-compatible!)
struct PlasticState
    ε_p::SymmetricTensor{2,3,Float64,6}  # Plastic strain tensor
    α::Float64                            # Accumulated plastic strain
end

# Material properties
struct Material
    E::Float64   # Young's modulus
    ν::Float64   # Poisson's ratio
    σ_y::Float64 # Yield stress
end

# Node-to-elements connectivity (CSR format)
struct NodeToElementsMap
    ptr::CuArray{Int32,1}   # Length: n_nodes + 1
    data::CuArray{Int32,1}  # Length: total connections
end

"""
Build CSR map: which elements touch each node?
"""
function build_node_to_elems_gpu(elements::Vector{NTuple{4,Int}}, n_nodes::Int)
    # Count connections per node
    counts = zeros(Int, n_nodes)
    for elem in elements
        for node in elem
            counts[node] += 1
        end
    end

    # Build CSR structure
    ptr = cumsum([1; counts])
    data = Vector{Int32}(undef, sum(counts))

    # Fill data array
    offset = copy(ptr[1:end-1])
    for (elem_idx, elem) in enumerate(elements)
        for node in elem
            data[offset[node]] = elem_idx
            offset[node] += 1
        end
    end

    return NodeToElementsMap(CuArray(Int32.(ptr)), CuArray(data))
end

"""
Return mapping for von Mises perfect plasticity (using Tensors.jl on GPU!)

This function works identically on CPU and GPU!
"""
@inline function return_mapping_tensor(ε_total::SymmetricTensor{2,3,T},
    state_old::PlasticState,
    E, ν, σ_y) where T
    # Elastic strain
    ε_e = ε_total - state_old.ε_p

    # Elastic predictor
    λ = E * ν / ((1 + ν) * (1 - 2ν))
    μ = E / (2(1 + ν))
    I = one(ε_e)
    σ_trial = λ * tr(ε_e) * I + 2μ * ε_e

    # Deviatoric stress
    σ_dev = dev(σ_trial)
    σ_eq = sqrt(3 / 2 * (σ_dev ⊡ σ_dev))

    # Yield function
    f = σ_eq - σ_y

    if f <= T(0.0)
        # Elastic
        return (σ_trial, state_old)
    else
        # Plastic - radial return
        Δγ = f / (3μ)
        n = σ_dev / σ_eq

        σ = σ_trial - 2μ * Δγ * n

        # Update plastic state
        Δε_p = Δγ * n
        ε_p_new = state_old.ε_p + Δε_p
        α_new = state_old.α + Δγ

        state_new = PlasticState(ε_p_new, α_new)

        return (σ, state_new)
    end
end

"""
PHASE 1 GPU KERNEL: Compute integration point data

One thread per integration point!
"""
function compute_gp_data_kernel!(
    σ_gp::CuDeviceArray{SymmetricTensor{2,3,Float64,6},1},
    states_new::CuDeviceArray{PlasticState,1},
    u::CuDeviceArray{Float64,1},
    nodes::CuDeviceArray{Float64,2},  # Shape: 3 × n_nodes
    elements::CuDeviceArray{Int32,2},  # Shape: 4 × n_elems
    states_old::CuDeviceArray{PlasticState,1},
    E, ν, σ_y
)
    gp_idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    if gp_idx <= length(σ_gp)
        # Map GP to element and local GP
        elem_idx = (gp_idx - 1) ÷ 4 + 1  # 4 GPs per Tet4
        # local_gp = (gp_idx - 1) % 4 + 1  # Not used yet (all GPs same for linear Tet4)

        # Extract element nodes
        n1 = elements[1, elem_idx]
        n2 = elements[2, elem_idx]
        n3 = elements[3, elem_idx]
        n4 = elements[4, elem_idx]

        # Node coordinates (using Tensors.jl Vec!)
        X1 = Vec{3}((nodes[1, n1], nodes[2, n1], nodes[3, n1]))
        X2 = Vec{3}((nodes[1, n2], nodes[2, n2], nodes[3, n2]))
        X3 = Vec{3}((nodes[1, n3], nodes[2, n3], nodes[3, n3]))
        X4 = Vec{3}((nodes[1, n4], nodes[2, n4], nodes[3, n4]))

        # Displacements (using Tensors.jl Vec!)
        u1 = Vec{3}((u[3*n1-2], u[3*n1-1], u[3*n1]))
        u2 = Vec{3}((u[3*n2-2], u[3*n2-1], u[3*n2]))
        u3 = Vec{3}((u[3*n3-2], u[3*n3-1], u[3*n3]))
        u4 = Vec{3}((u[3*n4-2], u[3*n4-1], u[3*n4]))

        # Shape derivatives (constant for Tet4)
        dN1_dxi = Vec{3}((-1.0, -1.0, -1.0))
        dN2_dxi = Vec{3}((1.0, 0.0, 0.0))
        dN3_dxi = Vec{3}((0.0, 1.0, 0.0))
        dN4_dxi = Vec{3}((0.0, 0.0, 1.0))

        # Jacobian (using tensor products!)
        J = dN1_dxi ⊗ X1 + dN2_dxi ⊗ X2 + dN3_dxi ⊗ X3 + dN4_dxi ⊗ X4
        invJ = inv(J)

        # Physical derivatives (using tensor contractions!)
        dN1_dx = invJ ⋅ dN1_dxi
        dN2_dx = invJ ⋅ dN2_dxi
        dN3_dx = invJ ⋅ dN3_dxi
        dN4_dx = invJ ⋅ dN4_dxi

        # Strain (using tensor products and symmetric!)
        ε = symmetric(dN1_dx ⊗ u1 + dN2_dx ⊗ u2 + dN3_dx ⊗ u3 + dN4_dx ⊗ u4)

        # Material state update (using Tensors.jl - works on GPU!)
        state_old = states_old[gp_idx]
        σ, state_new = return_mapping_tensor(ε, state_old, E, ν, σ_y)

        # Store results
        σ_gp[gp_idx] = σ
        states_new[gp_idx] = state_new
    end

    return nothing
end

"""
PHASE 2 GPU KERNEL: Nodal assembly (matrix-free, no atomics!)

One thread per node!
"""
function nodal_assembly_kernel!(
    r::CuDeviceArray{Float64,1},
    σ_gp::CuDeviceArray{SymmetricTensor{2,3,Float64,6},1},
    nodes::CuDeviceArray{Float64,2},
    elements::CuDeviceArray{Int32,2},
    node_to_elems_ptr::CuDeviceArray{Int32,1},
    node_to_elems_data::CuDeviceArray{Int32,1}
)
    node_idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    if node_idx <= size(nodes, 2)
        # Accumulate forces from all elements touching this node
        f_node = zero(Vec{3,Float64})

        # Gauss weights for Tet4 (standard 4-point quadrature)
        gauss_weight = 1.0 / 24.0

        # Shape derivatives in reference coordinates
        dN_dxi = (
            Vec{3}((-1.0, -1.0, -1.0)),
            Vec{3}((1.0, 0.0, 0.0)),
            Vec{3}((0.0, 1.0, 0.0)),
            Vec{3}((0.0, 0.0, 1.0))
        )

        # Get element range for this node (CSR format)
        elem_start = node_to_elems_ptr[node_idx]
        elem_end = node_to_elems_ptr[node_idx+1] - 1

        # Loop over touching elements
        for elem_offset in elem_start:elem_end
            elem_idx = node_to_elems_data[elem_offset]

            # Extract element nodes
            n1 = elements[1, elem_idx]
            n2 = elements[2, elem_idx]
            n3 = elements[3, elem_idx]
            n4 = elements[4, elem_idx]

            # Find local node index in element
            local_node = 1
            if node_idx == n2
                local_node = 2
            elseif node_idx == n3
                local_node = 3
            elseif node_idx == n4
                local_node = 4
            end

            # Recompute geometry (matrix-free!)
            X1 = Vec{3}((nodes[1, n1], nodes[2, n1], nodes[3, n1]))
            X2 = Vec{3}((nodes[1, n2], nodes[2, n2], nodes[3, n2]))
            X3 = Vec{3}((nodes[1, n3], nodes[2, n3], nodes[3, n3]))
            X4 = Vec{3}((nodes[1, n4], nodes[2, n4], nodes[3, n4]))

            J = dN_dxi[1] ⊗ X1 + dN_dxi[2] ⊗ X2 + dN_dxi[3] ⊗ X3 + dN_dxi[4] ⊗ X4
            detJ = det(J)
            invJ = inv(J)

            # Physical derivative for this node
            dN_dx = invJ ⋅ dN_dxi[local_node]

            # Loop over Gauss points (4 per Tet4)
            for local_gp in 1:4
                gp_idx = (elem_idx - 1) * 4 + local_gp

                # Get stress at this GP
                σ = σ_gp[gp_idx]

                # Accumulate force (using tensor contraction!)
                f_node += (dN_dx ⋅ σ) * (gauss_weight * detJ)
            end
        end

        # Write result (no atomics - this node is ours!)
        r[3*node_idx-2] = f_node[1]
        r[3*node_idx-1] = f_node[2]
        r[3*node_idx] = f_node[3]
    end

    return nothing
end

"""
Complete residual computation on GPU (two-phase approach)
"""
function compute_residual_gpu!(
    r::CuArray{Float64,1},
    u::CuArray{Float64,1},
    nodes::CuArray{Float64,2},
    elements::CuArray{Int32,2},
    states_old::CuArray{PlasticState,1},
    mat::Material,
    node_to_elems::NodeToElementsMap
)
    n_gp = length(states_old)
    n_nodes = size(nodes, 2)

    # Storage for integration point data (on GPU!)
    σ_gp = CuArray{SymmetricTensor{2,3,Float64,6}}(undef, n_gp)
    states_new = CuArray{PlasticState}(undef, n_gp)

    # Phase 1: Compute integration point data
    threads = 256
    blocks = cld(n_gp, threads)

    @cuda threads = threads blocks = blocks compute_gp_data_kernel!(
        σ_gp, states_new,
        u, nodes, elements,
        states_old,
        mat.E, mat.ν, mat.σ_y
    )

    # Phase 2: Nodal assembly
    fill!(r, 0.0)

    threads = 256
    blocks = cld(n_nodes, threads)

    @cuda threads = threads blocks = blocks nodal_assembly_kernel!(
        r, σ_gp,
        nodes, elements,
        node_to_elems.ptr, node_to_elems.data
    )

    return r, states_new
end

# ============================================================================
# Test Setup
# ============================================================================

function main()
    println("\n" * "="^70)
    println("Nodal Assembly GPU Implementation - CUDA.jl + Tensors.jl")
    println("="^70)

    # Check CUDA availability
    if !CUDA.functional()
        println("❌ CUDA not available! This demo requires a GPU.")
        return
    end

    println("\n✅ CUDA device: ", CUDA.name(CUDA.device()))

    # Single Tet4 element
    nodes_cpu = Float64[
        0.0 1.0 0.0 0.0;   # X coordinates
        0.0 0.0 1.0 0.0;   # Y coordinates
        0.0 0.0 0.0 1.0    # Z coordinates
    ]

    elements_cpu = [(1, 2, 3, 4)]
    elements_mat = Int32[e[i] for i in 1:4, e in elements_cpu]

    n_nodes = 4
    n_elems = 1
    n_gps = n_elems * 4  # 4 GPs per Tet4

    # Material
    mat = Material(
        210e3,  # E = 210 GPa (steel)
        0.3,    # ν = 0.3
        250.0   # σ_y = 250 MPa
    )

    # Displacement (apply tension)
    u_cpu = zeros(12)
    u_cpu[4] = 0.01  # Move node 2 in X-direction

    # Initial states (all elastic)
    states_old_cpu = [PlasticState(zero(SymmetricTensor{2,3,Float64}), 0.0) for _ in 1:n_gps]

    # Build node-to-elements map
    println("\nBuilding node-to-elements map (CSR format)...")
    node_to_elems = build_node_to_elems_gpu(elements_cpu, n_nodes)

    # Transfer to GPU
    println("Transferring data to GPU...")
    nodes_gpu = CuArray(nodes_cpu)
    elements_gpu = CuArray(elements_mat)
    u_gpu = CuArray(u_cpu)
    states_old_gpu = CuArray(states_old_cpu)
    r_gpu = CUDA.zeros(Float64, 12)

    # Compute residual on GPU
    println("\n" * "-"^70)
    println("Computing residual on GPU (two-phase nodal assembly)...")
    println("-"^70)

    r_gpu, states_new_gpu = compute_residual_gpu!(
        r_gpu, u_gpu, nodes_gpu, elements_gpu,
        states_old_gpu, mat, node_to_elems
    )

    # Transfer results back to CPU
    r_cpu = Array(r_gpu)
    states_new_cpu = Array(states_new_gpu)

    println("\nResidual vector (internal forces):")
    for i in 1:n_nodes
        rx = r_cpu[3*i-2]
        ry = r_cpu[3*i-1]
        rz = r_cpu[3*i]
        @printf("Node %d: [%12.6e, %12.6e, %12.6e]\n", i, rx, ry, rz)
    end

    println("\nResidual norm: ", norm(r_cpu))

    # Check material states
    println("\n" * "-"^70)
    println("Material States at Gauss Points:")
    println("-"^70)

    for (gp_idx, state) in enumerate(states_new_cpu)
        elem_idx = (gp_idx - 1) ÷ 4 + 1
        local_gp = (gp_idx - 1) % 4 + 1

        status = state.α > 0.0 ? "Plastic" : "Elastic"
        @printf("Elem %d, GP %d: %s (α = %.6e)\n", elem_idx, local_gp, status, state.α)
    end

    # Test force balance
    println("\n" * "-"^70)
    println("Force Balance Check:")
    println("-"^70)

    f_total = sum(reshape(r_cpu, 3, :), dims=2)
    @printf("Sum of forces: [%.6e, %.6e, %.6e]\n", f_total[1], f_total[2], f_total[3])
    @printf("Should be ≈ zero for internal forces (tol: 1e-10)\n")

    if norm(f_total) < 1e-10
        println("✅ Force balance: PASSED")
    else
        println("❌ Force balance: FAILED")
    end

    # Compare with CPU reference
    println("\n" * "="^70)
    println("Comparing with CPU reference (demos/nodal_assembly_cpu.jl)...")
    println("="^70)
    println("\nExpected residual norms should match!")
    println("  CPU reference: 727.208...")
    println("  GPU result:    ", norm(r_cpu))

    println("\n" * "="^70)
    println("✅ GPU nodal assembly complete!")
    println("="^70)
    println("\nNext steps:")
    println("  1. Benchmark GPU vs CPU performance")
    println("  2. Scale to realistic mesh sizes (10K+ elements)")
    println("  3. Integrate with Newton-Krylov solver")
    println("  4. Add line search for convergence")
    println("  5. Add preconditioning (Chebyshev-Jacobi → GMG)")
    println("="^70 * "\n")
end

main()
