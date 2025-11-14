# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
CPU Backend for Elasticity

Uses traditional element assembly with iterative CG solver.
Works directly with immutable Elements and Physics structure.
"""

using LinearAlgebra
using SparseArrays
using Tensors

"""
    ElasticityDataCPU <: AbstractElasticityData

CPU backend data using element assembly structures.

Fields:
- `assembly`: ElementAssemblyData for global system
- `n_nodes`: Number of nodes
- `n_dofs`: Number of DOFs (3 * n_nodes for 3D)
"""
struct ElasticityDataCPU <: AbstractElasticityData
    assembly::ElementAssemblyData{Float64}
    n_nodes::Int
    n_dofs::Int
end

"""
    initialize_backend(::CPU, physics::Physics{ElasticityPhysicsType}, time::Float64)

Initialize CPU backend from Physics problem using NEW immutable Element API.

Assembles the global system directly from immutable elements without
using the old Problem/update! API.
"""
function initialize_backend(::CPU, physics::Physics{ElasticityPhysicsType}, time::Float64)
    # Determine problem size from elements
    if length(physics.body_elements) == 0
        error("No elements in physics problem!")
    end

    # Get all unique nodes from elements
    all_nodes = Set{Int}()
    for element in physics.body_elements
        conn = get_connectivity(element)
        union!(all_nodes, conn)
    end

    n_nodes = maximum(all_nodes)
    n_dofs = physics.dimension * n_nodes

    # Create assembly data
    assembly = ElementAssemblyData(n_dofs, Float64)

    # Assemble each element
    for (elem_id, element) in enumerate(physics.body_elements)
        # Get element connectivity and DOF indices
        conn = get_connectivity(element)
        conn_int = NTuple{length(conn),Int64}(conn)  # Convert UInt64 → Int64
        gdofs = get_dof_indices(conn_int, physics.dimension)

        # Compute element stiffness using proper integration and basis functions
        K_local = compute_element_stiffness(element, time)

        # Create contribution and scatter to global
        contrib = ElementContribution(elem_id, collect(gdofs), K_local,
            zeros(length(gdofs)), zeros(length(gdofs)))
        scatter_to_global!(assembly, contrib)
    end

    # Apply external forces (if any body forces in elements)
    # TODO: Implement body force extraction from element fields

    # Compute residual
    compute_residual!(assembly)

    # Apply Dirichlet boundary conditions
    bc = physics.bc_dirichlet
    if length(bc.node_ids) > 0
        fixed_dofs = Int[]
        prescribed_values = Float64[]

        for (node, components, values) in zip(bc.node_ids, bc.components, bc.values)
            for (comp, val) in zip(components, values)
                dof = physics.dimension * (node - 1) + comp
                push!(fixed_dofs, dof)
                push!(prescribed_values, val)
            end
        end

        apply_dirichlet_bc!(assembly, fixed_dofs, prescribed_values)
    end

    return ElasticityDataCPU(assembly, n_nodes, n_dofs)
end

"""
    compute_element_stiffness(element, time)

Compute element stiffness matrix using NEW API: topology/integration + basis evaluation.

USES NEW API:
- integration_points(Gauss{order}(), topology) for quadrature
- get_basis_derivatives(topology, basis, xi) for shape function gradients
- Tensors.jl for all math (NO B-matrix!)

Implementation follows golden standard: docs/src/book/multigpu_nodal_assembly.md
Uses 4th-order elasticity tensor with double contractions (NO Voigt notation!)
"""
function compute_element_stiffness(element::Element{N,NIP,F,B}, time::Float64) where {N,NIP,F,B}
    # Get element properties from fields
    X = element.fields.geometry  # Vector{Vec{3}} of node coordinates
    E = element.fields.youngs_modulus
    ν = element.fields.poissons_ratio

    # Material parameters (Lamé constants)
    λ = E * ν / ((1 + ν) * (1 - 2ν))
    μ = E / (2(1 + ν))

    # 4th-order elasticity tensor (Tensors.jl, symmetric in all index pairs)
    # C_ijkl = λ δ_ij δ_kl + μ (δ_ik δ_jl + δ_il δ_jk)
    δ(i, j) = i == j ? 1.0 : 0.0
    C_ijkl = [(λ * δ(i, j) * δ(k, l) + μ * (δ(i, k) * δ(j, l) + δ(i, l) * δ(j, k)))
              for i in 1:3, j in 1:3, k in 1:3, l in 1:3]
    C = Tensor{4,3}(tuple(C_ijkl...))

    # Extract topology and basis from element type parameter B
    # B is Lagrange{Topology, Order}
    topology_type = extract_topology_type(B)
    # Create topology instance - use N from element (8 for Hex8, etc.)
    topology = topology_type{N}()
    basis = B()

    # NEW API: integration points from topology module
    ips = integration_points(Gauss{2}(), topology)

    # Initialize element stiffness as 3×3 blocks (Tensors.jl approach)
    K_blocks = [[zero(Tensor{2,3}) for _ in 1:N] for _ in 1:N]

    # Integrate over element
    for ip in ips
        ξ = Vec{3}(ip.ξ)
        w = ip.weight

        # NEW API: Basis function derivatives (shape function gradients in reference coords)
        dN_dξ = get_basis_derivatives(topology, basis, ξ)  # Returns NTuple{N, Vec{3}}

        # Jacobian transformation: J_ij = ∑_k X_k^i ∂N_k/∂ξ^j
        # Build Jacobian as Tensor{2,3} (3×3 matrix)
        J = zero(Tensor{2,3})
        for k in 1:N
            # Outer product: X[k] ⊗ dN_dξ[k] gives 3×3 tensor
            J += X[k] ⊗ dN_dξ[k]
        end
        detJ = det(J)
        J_inv = inv(J)

        # Shape derivatives in physical coordinates: ∂N_i/∂x = J^{-T} ⋅ ∂N_i/∂ξ
        dN_dx = tuple([J_inv ⋅ dN_dξ[i] for i in 1:N]...)

        # Assemble stiffness blocks using Tensors.jl (NO B-matrix!)
        # K_ij^{αβ} = ∫ (∂N_i/∂x_γ) C_{αβγδ} (∂N_j/∂x_δ) detJ dξ
        for i in 1:N, j in 1:N
            # Gradient tensors: ∂N/∂x as Vec{3}
            grad_i = dN_dx[i]  # Vec{3}
            grad_j = dN_dx[j]  # Vec{3}

            # Compute stiffness contribution: K_ij^{αβ} += (∂N_i/∂x_γ) C_{αβγδ} (∂N_j/∂x_δ) detJ w
            # Use double contraction over γ and δ indices
            K_contrib = zero(Tensor{2,3})
            for α in 1:3, β in 1:3
                stiffness_component = 0.0
                for γ in 1:3, δ in 1:3
                    stiffness_component += grad_i[γ] * C[α, β, γ, δ] * grad_j[δ]
                end
                # Construct 3×3 tensor contribution (only αβ component nonzero)
                e_α = basevec(Val{3}(), α)  # Unit vector in direction α
                e_β = basevec(Val{3}(), β)  # Unit vector in direction β
                K_contrib += stiffness_component * (e_α ⊗ e_β)
            end
            K_blocks[i][j] += K_contrib * detJ * w
        end
    end

    # Convert blocked Tensor{2,3} format to standard Float64 matrix
    ndofs = 3 * N
    K_e = zeros(ndofs, ndofs)
    for i in 1:N, j in 1:N
        for α in 1:3, β in 1:3
            K_e[3*(i-1)+α, 3*(j-1)+β] = K_blocks[i][j][α, β]
        end
    end

    return K_e
end

# Helper: Unit basis vector
@inline basevec(::Val{3}, i::Int) = Vec{3}(ntuple(j -> j == i ? 1.0 : 0.0, 3))

# Helper to extract topology TYPE from Lagrange{T, O} (returns TYPE, not instance)
extract_topology_type(::Type{Lagrange{T,O}}) where {T,O} = T

"""
    solve_backend!(data::ElasticityDataCPU, physics::Physics{ElasticityPhysicsType}; kwargs...)

Solve elasticity problem using CPU backend with CG solver.

Returns: (u, iterations, residual)
"""
function solve_backend!(data::ElasticityDataCPU, physics::Physics{ElasticityPhysicsType};
    tol=1e-6, max_iter=1000,
    newton_tol=1e-6, max_newton=20, max_cg_per_newton=50)

    # For now, linear elasticity only (no Newton iterations)
    # TODO: Add Newton-Raphson for nonlinear problems

    # Conjugate Gradient solver
    function cg_solve(A::ElementAssemblyData, b::Vector{Float64};
        tol=1e-8, max_iter=1000)

        n = length(b)
        x = zeros(n)
        r = b - matrix_vector_product(A, x)

        # Early exit if already converged
        r_norm = norm(r)
        if r_norm < tol
            return x, 0, r_norm
        end

        p = copy(r)
        rsold = dot(r, r)

        for iter in 1:max_iter
            Ap = matrix_vector_product(A, p)
            α = rsold / dot(p, Ap)
            x .+= α .* p
            r .-= α .* Ap
            rsnew = dot(r, r)

            if sqrt(rsnew) < tol
                return x, iter, sqrt(rsnew)
            end

            β = rsnew / rsold
            p .= r .+ β .* p
            rsold = rsnew
        end

        return x, max_iter, sqrt(rsold)
    end

    # Solve
    u, iterations, residual = cg_solve(data.assembly, data.assembly.r_global,
        tol=tol, max_iter=max_iter)

    return (u, iterations, residual)
end
