"""
Complete Newton-Krylov-Anderson Reference Implementation (CPU)
==============================================================

Shows the full nonlinear solver pipeline:
1. Newton iteration (outer loop)
2. GMRES for linear solve (inner loop, matrix-free)
3. Anderson acceleration (outer loop acceleration)
4. Perfect plasticity with state variables

This is the REFERENCE. Once it works, we port to GPU.
"""

using LinearAlgebra
using Tensors
using Printf

# ============================================================================
# Material: Perfect Plasticity with State Variables
# ============================================================================

struct VonMisesPlasticity
    E::Float64      # Young's modulus
    ν::Float64      # Poisson's ratio
    σ_y::Float64    # Yield stress
end

# Material constants
λ(mat::VonMisesPlasticity) = mat.E * mat.ν / ((1 + mat.ν) * (1 - 2mat.ν))
μ(mat::VonMisesPlasticity) = mat.E / (2(1 + mat.ν))

# Plastic state per integration point
mutable struct PlasticState
    ε_p::SymmetricTensor{2,3,Float64,6}  # Plastic strain
    α::Float64                             # Accumulated plastic strain
end

PlasticState() = PlasticState(zero(SymmetricTensor{2,3,Float64}), 0.0)

"""
Compute stress with return mapping (radial return algorithm).
Returns (σ, state_new, plastic_loading).
"""
function compute_stress_with_plasticity(material::VonMisesPlasticity,
    ε_total::SymmetricTensor{2,3,Float64},
    state_old::PlasticState)
    # Trial elastic strain
    ε_e_trial = ε_total - state_old.ε_p

    # Trial stress (elastic predictor)
    λ_val = λ(material)
    μ_val = μ(material)
    I = one(ε_e_trial)
    σ_trial = λ_val * tr(ε_e_trial) * I + 2 * μ_val * ε_e_trial

    # Deviatoric stress
    σ_dev = dev(σ_trial)
    σ_eq = sqrt(3 / 2 * (σ_dev ⊡ σ_dev))  # von Mises equivalent stress

    # Check yield condition
    f_trial = σ_eq - material.σ_y

    if f_trial <= 0.0
        # Elastic loading - no plasticity
        return (σ_trial, state_old, false)
    else
        # Plastic loading - return mapping
        Δγ = f_trial / (3 * μ_val)  # Plastic multiplier (for perfect plasticity)

        # Return mapping
        n = σ_dev / σ_eq  # Flow direction (normal to yield surface)
        σ_new = σ_trial - 2 * μ_val * Δγ * n

        # Update plastic strain
        Δε_p = Δγ * n
        ε_p_new = state_old.ε_p + Δε_p
        α_new = state_old.α + Δγ

        state_new = PlasticState(ε_p_new, α_new)

        return (σ_new, state_new, true)
    end
end

# ============================================================================
# Tet4: Linear Tetrahedron (simpler than Tet10 for GPU proof-of-concept)
# ============================================================================

# Gauss quadrature: 1-point for Tet4 (sufficient for linear element)
const GAUSS_TET4_1PT = (
    (Vec{3}((0.25, 0.25, 0.25)), 1.0 / 6.0),  # Weight = volume of reference tet
)

"""
Shape function derivatives for Tet4 (constant in reference element).
Returns tuple of 4 Vec{3}.
"""
function tet4_shape_derivatives()
    # For reference Tet4: vertices at (0,0,0), (1,0,0), (0,1,0), (0,0,1)
    # dN/dξ are constant:
    dN1 = Vec{3}((-1.0, -1.0, -1.0))
    dN2 = Vec{3}((1.0, 0.0, 0.0))
    dN3 = Vec{3}((0.0, 1.0, 0.0))
    dN4 = Vec{3}((0.0, 0.0, 1.0))
    return (dN1, dN2, dN3, dN4)
end

# ============================================================================
# Element Residual Assembly (Matrix-Free)
# ============================================================================

"""
Assemble element residual for Tet4.
This is what gets called inside the matrix-free operator.
"""
function assemble_element_residual!(r_elem, X, u, material, states)
    # Shape derivatives (constant for Tet4)
    dN_dxi = tet4_shape_derivatives()

    # Zero residual
    fill!(r_elem, zero(Vec{3,Float64}))

    # Integration loop (1 point for Tet4)
    for (gp_idx, (xivec, w)) in enumerate(GAUSS_TET4_1PT)
        # Jacobian: J = Σ dN_i ⊗ X_i
        J = sum(dN_dxi[i] ⊗ X[i] for i in 1:4)
        detJ = det(J)
        invJ = inv(J)

        # Physical derivatives: dN/dx = J^-1 · dN/dξ
        dN_dx = ntuple(i -> invJ ⋅ Vec{3}(dN_dxi[i]), Val(4))

        # Strain: ε = sym(∇u)
        eps = symmetric(sum(dN_dx[i] ⊗ u[i] for i in 1:4))

        # Stress with plasticity
        state_old = states[gp_idx]
        (sigma, state_new, plastic) = compute_stress_with_plasticity(material, eps, state_old)
        states[gp_idx] = state_new  # Update state

        # Nodal forces: f_i = dN_i · σ
        f_contrib = ntuple(i -> dN_dx[i] ⋅ sigma, Val(4))

        # Accumulate
        for i in 1:4
            r_elem[i] += f_contrib[i] * (w * detJ)
        end
    end

    return r_elem
end

"""
Global residual assembly (loop over elements).
"""
function assemble_global_residual!(r, u, mesh, material, element_states)
    fill!(r, 0.0)

    for (elem_idx, elem) in enumerate(mesh.elements)
        # Extract element data
        X = ntuple(i -> mesh.nodes[elem[i]], Val(4))
        u_elem = ntuple(i -> Vec{3}((u[3*elem[i]-2], u[3*elem[i]-1], u[3*elem[i]])), Val(4))

        # Element residual
        r_elem = [zero(Vec{3,Float64}) for _ in 1:4]
        assemble_element_residual!(r_elem, X, u_elem, material, element_states[elem_idx])

        # Scatter to global
        for (i, node) in enumerate(elem)
            r[3*node-2] += r_elem[i][1]
            r[3*node-1] += r_elem[i][2]
            r[3*node] += r_elem[i][3]
        end
    end

    return r
end

# ============================================================================
# Matrix-Free Operator for GMRES
# ============================================================================

"""
Matrix-free Jacobian-vector product: J·v ≈ (R(u+ε·v) - R(u))/ε
"""
struct MatrixFreeJacobian
    u::Vector{Float64}      # Current solution
    r::Vector{Float64}      # Current residual R(u)
    mesh::Any               # Mesh data
    material::Any           # Material
    element_states::Vector  # Plastic states (one vector per element)
    ε::Float64              # Finite difference step

    # Temporary storage
    u_pert::Vector{Float64}
    r_pert::Vector{Float64}
end

function MatrixFreeJacobian(u, r, mesh, material, element_states, ε=1e-7)
    u_pert = similar(u)
    r_pert = similar(r)
    return MatrixFreeJacobian(u, r, mesh, material, element_states, ε, u_pert, r_pert)
end

"""
Apply J·v using finite differences.
"""
function apply_jacobian!(result, J::MatrixFreeJacobian, v)
    # Perturbed solution: u + ε·v
    @. J.u_pert = J.u + J.ε * v

    # Need to copy states for perturbation (don't modify original)
    states_pert = [copy(states) for states in J.element_states]

    # Perturbed residual: R(u + ε·v)
    assemble_global_residual!(J.r_pert, J.u_pert, J.mesh, J.material, states_pert)

    # Finite difference: (R(u+ε·v) - R(u))/ε
    @. result = (J.r_pert - J.r) / J.ε

    return result
end

# ============================================================================
# GMRES Solver (Matrix-Free)
# ============================================================================

"""
Simple GMRES implementation (matrix-free).
Solves J·du = -r for Newton correction.
"""
function gmres_solve!(du, J::MatrixFreeJacobian, r, max_iter=50, tol=1e-6)
    n = length(r)

    # Krylov subspace basis
    V = [zeros(n) for _ in 1:max_iter+1]
    H = zeros(max_iter + 1, max_iter)

    # Initial residual
    fill!(du, 0.0)
    V[1] .= -r  # We're solving J·du = -r
    β = norm(V[1])
    V[1] ./= β

    # Givens rotations storage
    g = zeros(max_iter + 1)
    g[1] = β
    c = zeros(max_iter)
    s = zeros(max_iter)

    for j in 1:max_iter
        # Arnoldi: w = J·v_j
        w = zeros(n)
        apply_jacobian!(w, J, V[j])

        # Modified Gram-Schmidt
        for i in 1:j
            H[i, j] = dot(w, V[i])
            w .-= H[i, j] .* V[i]
        end
        H[j+1, j] = norm(w)

        if H[j+1, j] > 1e-14
            V[j+1] .= w ./ H[j+1, j]
        end

        # Apply previous Givens rotations
        for i in 1:j-1
            temp = c[i] * H[i, j] + s[i] * H[i+1, j]
            H[i+1, j] = -s[i] * H[i, j] + c[i] * H[i+1, j]
            H[i, j] = temp
        end

        # Compute new Givens rotation
        ρ = sqrt(H[j, j]^2 + H[j+1, j]^2)
        c[j] = H[j, j] / ρ
        s[j] = H[j+1, j] / ρ
        H[j, j] = ρ
        H[j+1, j] = 0.0

        # Update residual norm
        g[j+1] = -s[j] * g[j]
        g[j] = c[j] * g[j]

        residual_norm = abs(g[j+1])

        if residual_norm < tol * β
            # Back-solve upper triangular system
            y = zeros(j)
            for i in j:-1:1
                y[i] = g[i]
                for k in i+1:j
                    y[i] -= H[i, k] * y[k]
                end
                y[i] /= H[i, i]
            end

            # Form solution: du = V * y
            for i in 1:j
                du .+= y[i] .* V[i]
            end

            @printf("    GMRES converged in %d iterations (res: %.2e)\n", j, residual_norm)
            return true
        end
    end

    @printf("    GMRES did NOT converge after %d iterations\n", max_iter)
    return false
end

# ============================================================================
# Anderson Acceleration
# ============================================================================

"""
Anderson acceleration for Newton iterations.
Mixes previous iterates to accelerate convergence.
"""
mutable struct AndersonAccelerator
    m::Int                          # Mixing depth
    X::Vector{Vector{Float64}}      # Previous iterates
    F::Vector{Vector{Float64}}      # Previous residuals
    iter::Int                       # Current iteration
end

function AndersonAccelerator(n::Int, m::Int=5)
    X = [zeros(n) for _ in 1:m]
    F = [zeros(n) for _ in 1:m]
    return AndersonAccelerator(m, X, F, 0)
end

"""
Apply Anderson mixing to compute next iterate.
"""
function anderson_step!(acc::AndersonAccelerator, u_new, f_new)
    acc.iter += 1

    if acc.iter == 1
        # First iteration - no mixing
        return copy(u_new)
    end

    # Number of previous iterates to use
    k = min(acc.iter - 1, acc.m)

    # Store current iterate
    idx = mod1(acc.iter, acc.m)
    acc.X[idx] .= u_new
    acc.F[idx] .= f_new

    if k == 1
        # Not enough history - just return current
        return copy(u_new)
    end

    # Build ΔF matrix (differences of residuals)
    ΔF = zeros(length(f_new), k - 1)
    for i in 1:k-1
        idx_curr = mod1(acc.iter - i + 1, acc.m)
        idx_prev = mod1(acc.iter - i, acc.m)
        ΔF[:, i] .= acc.F[idx_curr] .- acc.F[idx_prev]
    end

    # Solve least-squares: min ||ΔF·θ - f_new||
    θ = ΔF \ f_new

    # Mixed iterate: u = u_new - Σ θ_i (u_{k-i} - u_{k-i-1})
    u_mixed = copy(u_new)
    for i in 1:k-1
        idx_curr = mod1(acc.iter - i + 1, acc.m)
        idx_prev = mod1(acc.iter - i, acc.m)
        u_mixed .-= θ[i] .* (acc.X[idx_curr] .- acc.X[idx_prev])
    end

    return u_mixed
end

# ============================================================================
# Newton Solver with Anderson Acceleration
# ============================================================================

"""
Newton solver with GMRES and Anderson acceleration.
This is the COMPLETE PIPELINE.
"""
function solve_newton_krylov_anderson!(u, mesh, material, element_states;
    max_iter=20, tol=1e-6, anderson_depth=5)

    println("\n" * "="^70)
    println("Newton-Krylov-Anderson Solver")
    println("="^70)

    n = length(u)
    r = zeros(n)
    du = zeros(n)

    # Anderson accelerator
    anderson = AndersonAccelerator(n, anderson_depth)

    for iter in 1:max_iter
        # Assemble residual
        assemble_global_residual!(r, u, mesh, material, element_states)

        r_norm = norm(r)
        @printf("Newton iter %2d: ||r|| = %.6e\n", iter, r_norm)

        if r_norm < tol
            println("✅ Converged!")
            return true
        end

        # Matrix-free Jacobian
        J = MatrixFreeJacobian(u, r, mesh, material, element_states)

        # GMRES solve: J·du = -r
        gmres_solve!(du, J, r)

        # Line search parameter (simple version)
        α = 1.0
        u_new = u .+ α .* du

        # Anderson acceleration (mix with previous iterates)
        if anderson_depth > 0
            u_new = anderson_step!(anderson, u_new, r)
        end

        # Update solution
        u .= u_new
    end

    println("❌ Did NOT converge after $max_iter iterations")
    return false
end

# ============================================================================
# Test Problem
# ============================================================================

# Simple mesh structure
struct SimpleMesh
    nodes::Vector{Vec{3,Float64}}
    elements::Vector{NTuple{4,Int}}
end

function create_test_mesh()
    # Single Tet4 element
    nodes = [
        Vec{3}((0.0, 0.0, 0.0)),
        Vec{3}((1.0, 0.0, 0.0)),
        Vec{3}((0.0, 1.0, 0.0)),
        Vec{3}((0.0, 0.0, 1.0))
    ]
    elements = [(1, 2, 3, 4)]

    return SimpleMesh(nodes, elements)
end

function main()
    println("\n" * "="^70)
    println("Newton-Krylov-Anderson Reference Implementation (CPU)")
    println("="^70)

    # Mesh
    mesh = create_test_mesh()
    n_nodes = length(mesh.nodes)
    n_dofs = 3 * n_nodes

    # Material (perfect plasticity)
    material = VonMisesPlasticity(
        200e9,    # E = 200 GPa
        0.3,      # ν = 0.3
        250e6     # σ_y = 250 MPa
    )

    println("\n📦 Problem Setup:")
    println("  Elements: $(length(mesh.elements))")
    println("  Nodes: $n_nodes")
    println("  DOFs: $n_dofs")
    println("  Material: VonMises plasticity (σ_y = $(material.σ_y/1e6) MPa)")

    # Initial guess (small displacement to trigger plasticity)
    u = zeros(n_dofs)
    u[4] = 0.002  # Node 2, x-direction (2mm - should exceed elastic limit)

    # Initialize plastic states (one vector per element, one state per gauss point)
    element_states = [[PlasticState() for _ in 1:length(GAUSS_TET4_1PT)]
                      for _ in 1:length(mesh.elements)]

    # Solve
    converged = solve_newton_krylov_anderson!(u, mesh, material, element_states,
        max_iter=20, tol=1e-6, anderson_depth=3)

    println("\n📊 Final Results:")
    println("  Converged: $converged")
    println("  u (first 6 DOFs): $(u[1:6])")

    # Check plastic state
    for (elem_idx, states) in enumerate(element_states)
        for (gp_idx, state) in enumerate(states)
            if state.α > 1e-10
                @printf("  Element %d, GP %d: Plastic (α = %.6e)\n",
                    elem_idx, gp_idx, state.α)
            end
        end
    end

    println("\n✅ CPU REFERENCE COMPLETE!")
    println("="^70)
end

# Run
main()
