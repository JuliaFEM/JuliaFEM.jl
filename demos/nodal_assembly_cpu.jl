"""
Nodal Assembly CPU Reference Implementation

This implements the TWO-PHASE nodal assembly approach that will be ported to GPU:

Phase 1: Compute integration point data (stresses, material states)
Phase 2: Nodal assembly (matrix-free, no atomics)

Uses Tensors.jl throughout for natural tensor operations.
"""

using Tensors
using LinearAlgebra
using Printf

# Material state for plasticity
mutable struct PlasticState
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
    ptr::Vector{Int}   # Length: n_nodes + 1
    data::Vector{Int}  # Length: total connections
end

"""
Build CSR map: which elements touch each node?
"""
function build_node_to_elems(elements::Vector{NTuple{4,Int}}, n_nodes::Int)
    # Count connections per node
    counts = zeros(Int, n_nodes)
    for elem in elements
        for node in elem
            counts[node] += 1
        end
    end

    # Build CSR structure
    ptr = cumsum([1; counts])
    data = Vector{Int}(undef, sum(counts))

    # Fill data array
    offset = copy(ptr[1:end-1])
    for (elem_idx, elem) in enumerate(elements)
        for node in elem
            data[offset[node]] = elem_idx
            offset[node] += 1
        end
    end

    return NodeToElementsMap(ptr, data)
end

"""
Tet4 shape function derivatives in reference coordinates (constant!)
"""
function tet4_shape_derivatives()
    return (
        Vec{3}((-1.0, -1.0, -1.0)),  # dN1/dξ
        Vec{3}((1.0, 0.0, 0.0)),     # dN2/dξ
        Vec{3}((0.0, 1.0, 0.0)),     # dN3/dξ
        Vec{3}((0.0, 0.0, 1.0))      # dN4/dξ
    )
end

"""
Return mapping for von Mises perfect plasticity (using Tensors.jl!)
"""
function return_mapping_tensor(ε_total::SymmetricTensor{2,3,T},
    state_old::PlasticState,
    mat::Material) where T
    # Elastic strain
    ε_e = ε_total - state_old.ε_p

    # Elastic predictor
    λ = mat.E * mat.ν / ((1 + mat.ν) * (1 - 2mat.ν))
    μ = mat.E / (2(1 + mat.ν))
    I = one(ε_e)
    σ_trial = λ * tr(ε_e) * I + 2μ * ε_e

    # Deviatoric stress
    σ_dev = dev(σ_trial)
    σ_eq = sqrt(3 / 2 * (σ_dev ⊡ σ_dev))

    # Yield function
    f = σ_eq - mat.σ_y

    if f <= 0.0
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
Phase 1: Compute integration point data (stresses and material states)

One "thread" per integration point (in CPU version, just a loop)
"""
function compute_gp_data!(
    σ_gp::Vector{SymmetricTensor{2,3,Float64,6}},
    states_new::Vector{PlasticState},
    u::Vector{Float64},
    nodes::Matrix{Float64},  # Shape: 3 × n_nodes
    elements::Vector{NTuple{4,Int}},
    states_old::Vector{PlasticState},
    mat::Material
)
    n_elems = length(elements)
    n_gps_per_elem = 4  # 4 Gauss points for Tet4

    dN_dxi = tet4_shape_derivatives()

    for elem_idx in 1:n_elems
        # Extract element nodes
        n1, n2, n3, n4 = elements[elem_idx]

        X1 = Vec{3}((nodes[1, n1], nodes[2, n1], nodes[3, n1]))
        X2 = Vec{3}((nodes[1, n2], nodes[2, n2], nodes[3, n2]))
        X3 = Vec{3}((nodes[1, n3], nodes[2, n3], nodes[3, n3]))
        X4 = Vec{3}((nodes[1, n4], nodes[2, n4], nodes[3, n4]))

        u1 = Vec{3}((u[3*n1-2], u[3*n1-1], u[3*n1]))
        u2 = Vec{3}((u[3*n2-2], u[3*n2-1], u[3*n2]))
        u3 = Vec{3}((u[3*n3-2], u[3*n3-1], u[3*n3]))
        u4 = Vec{3}((u[3*n4-2], u[3*n4-1], u[3*n4]))

        # Jacobian (using tensor products!)
        J = dN_dxi[1] ⊗ X1 + dN_dxi[2] ⊗ X2 + dN_dxi[3] ⊗ X3 + dN_dxi[4] ⊗ X4
        invJ = inv(J)

        # Physical derivatives
        dN1_dx = invJ ⋅ dN_dxi[1]
        dN2_dx = invJ ⋅ dN_dxi[2]
        dN3_dx = invJ ⋅ dN_dxi[3]
        dN4_dx = invJ ⋅ dN_dxi[4]

        # Loop over Gauss points (for Tet4, same strain at all GPs since linear)
        # In real code, would have different GP locations
        for local_gp in 1:n_gps_per_elem
            gp_idx = (elem_idx - 1) * n_gps_per_elem + local_gp

            # Strain (using tensor products!)
            ε = symmetric(dN1_dx ⊗ u1 + dN2_dx ⊗ u2 + dN3_dx ⊗ u3 + dN4_dx ⊗ u4)

            # Material state update
            state_old = states_old[gp_idx]
            σ, state_new = return_mapping_tensor(ε, state_old, mat)

            # Store results
            σ_gp[gp_idx] = σ
            states_new[gp_idx] = state_new
        end
    end
end

"""
Phase 2: Nodal assembly (matrix-free, no atomics!)

One "thread" per node (in CPU version, just a loop)
"""
function nodal_assembly!(
    r::Vector{Float64},
    σ_gp::Vector{SymmetricTensor{2,3,Float64,6}},
    nodes::Matrix{Float64},
    elements::Vector{NTuple{4,Int}},
    node_to_elems::NodeToElementsMap
)
    n_nodes = size(nodes, 2)
    n_gps_per_elem = 4

    dN_dxi = tet4_shape_derivatives()

    # Gauss weights for Tet4 (standard 4-point quadrature)
    gauss_weights = (1 / 24, 1 / 24, 1 / 24, 1 / 24)

    fill!(r, 0.0)

    for node_idx in 1:n_nodes
        f_node = zero(Vec{3,Float64})

        # Get elements touching this node (CSR traversal)
        elem_start = node_to_elems.ptr[node_idx]
        elem_end = node_to_elems.ptr[node_idx+1] - 1

        # Loop over touching elements
        for elem_offset in elem_start:elem_end
            elem_idx = node_to_elems.data[elem_offset]
            elem_nodes = elements[elem_idx]

            # Find local node index in element
            local_node = findfirst(==(node_idx), elem_nodes)
            @assert local_node !== nothing "Node not found in element!"

            # Recompute geometry (matrix-free approach!)
            n1, n2, n3, n4 = elem_nodes
            X1 = Vec{3}((nodes[1, n1], nodes[2, n1], nodes[3, n1]))
            X2 = Vec{3}((nodes[1, n2], nodes[2, n2], nodes[3, n2]))
            X3 = Vec{3}((nodes[1, n3], nodes[2, n3], nodes[3, n3]))
            X4 = Vec{3}((nodes[1, n4], nodes[2, n4], nodes[3, n4]))

            J = dN_dxi[1] ⊗ X1 + dN_dxi[2] ⊗ X2 + dN_dxi[3] ⊗ X3 + dN_dxi[4] ⊗ X4
            detJ = det(J)
            invJ = inv(J)

            # Physical derivative for this node
            dN_dx = invJ ⋅ dN_dxi[local_node]

            # Loop over Gauss points
            for local_gp in 1:n_gps_per_elem
                gp_idx = (elem_idx - 1) * n_gps_per_elem + local_gp

                # Get stress at this GP
                σ = σ_gp[gp_idx]

                # Gauss weight
                w = gauss_weights[local_gp]

                # Accumulate force (using tensor contraction!)
                f_node += (dN_dx ⋅ σ) * (w * detJ)
            end
        end

        # Write result (in GPU version, no atomics needed - this node is ours!)
        r[3*node_idx-2] = f_node[1]
        r[3*node_idx-1] = f_node[2]
        r[3*node_idx] = f_node[3]
    end
end

"""
Complete residual computation (two-phase approach)
"""
function compute_residual!(
    r::Vector{Float64},
    u::Vector{Float64},
    nodes::Matrix{Float64},
    elements::Vector{NTuple{4,Int}},
    states_old::Vector{PlasticState},
    mat::Material,
    node_to_elems::NodeToElementsMap
)
    n_gp = length(states_old)

    # Storage for integration point data
    σ_gp = Vector{SymmetricTensor{2,3,Float64,6}}(undef, n_gp)
    states_new = Vector{PlasticState}(undef, n_gp)

    # Phase 1: Compute integration point data
    compute_gp_data!(σ_gp, states_new, u, nodes, elements, states_old, mat)

    # Phase 2: Nodal assembly
    nodal_assembly!(r, σ_gp, nodes, elements, node_to_elems)

    return r, states_new
end

# ============================================================================
# Test Setup
# ============================================================================

function main()
    println("\n" * "="^70)
    println("Nodal Assembly CPU Reference - Using Tensors.jl")
    println("="^70)

    # Single Tet4 element
    nodes = Float64[
        0.0 1.0 0.0 0.0;   # X coordinates
        0.0 0.0 1.0 0.0;   # Y coordinates
        0.0 0.0 0.0 1.0    # Z coordinates
    ]

    elements = [(1, 2, 3, 4)]
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
    u = zeros(12)
    u[4] = 0.01  # Move node 2 in X-direction

    # Initial states (all elastic)
    states_old = [PlasticState(zero(SymmetricTensor{2,3,Float64}), 0.0) for _ in 1:n_gps]

    # Build node-to-elements map
    println("\nBuilding node-to-elements map (CSR format)...")
    node_to_elems = build_node_to_elems(elements, n_nodes)

    println("CSR ptr: ", node_to_elems.ptr)
    println("CSR data: ", node_to_elems.data)

    # Verify each node touches exactly 1 element
    for node_idx in 1:n_nodes
        elem_start = node_to_elems.ptr[node_idx]
        elem_end = node_to_elems.ptr[node_idx+1] - 1
        n_touching = elem_end - elem_start + 1
        touching_elems = node_to_elems.data[elem_start:elem_end]
        println("Node $node_idx touches $n_touching element(s): $touching_elems")
    end

    # Compute residual (two-phase approach)
    println("\n" * "-"^70)
    println("Computing residual (two-phase nodal assembly)...")
    println("-"^70)

    r = zeros(12)
    r, states_new = compute_residual!(r, u, nodes, elements, states_old, mat, node_to_elems)

    println("\nResidual vector (internal forces):")
    for i in 1:n_nodes
        rx = r[3*i-2]
        ry = r[3*i-1]
        rz = r[3*i]
        @printf("Node %d: [%12.6e, %12.6e, %12.6e]\n", i, rx, ry, rz)
    end

    println("\nResidual norm: ", norm(r))

    # Check material states
    println("\n" * "-"^70)
    println("Material States at Gauss Points:")
    println("-"^70)

    for (gp_idx, state) in enumerate(states_new)
        elem_idx = (gp_idx - 1) ÷ 4 + 1
        local_gp = (gp_idx - 1) % 4 + 1

        status = state.α > 0.0 ? "Plastic" : "Elastic"
        @printf("Elem %d, GP %d: %s (α = %.6e)\n", elem_idx, local_gp, status, state.α)
    end

    # Test force balance (should sum to zero for internal forces)
    println("\n" * "-"^70)
    println("Force Balance Check:")
    println("-"^70)

    f_total = sum(reshape(r, 3, :), dims=2)
    @printf("Sum of forces: [%.6e, %.6e, %.6e]\n", f_total[1], f_total[2], f_total[3])
    @printf("Should be ≈ zero for internal forces (tol: 1e-10)\n")

    if norm(f_total) < 1e-10
        println("✅ Force balance: PASSED")
    else
        println("❌ Force balance: FAILED")
    end

    println("\n" * "="^70)
    println("✅ Nodal assembly CPU reference complete!")
    println("="^70)
    println("\nNext step: Port this to GPU with CUDA.jl")
    println("  Phase 1: @cuda compute_gp_data_kernel!(...)")
    println("  Phase 2: @cuda nodal_assembly_kernel!(...)")
    println("="^70 * "\n")
end

main()
