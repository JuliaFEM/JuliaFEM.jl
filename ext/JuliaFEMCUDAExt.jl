# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
JuliaFEMCUDAExt - GPU Backend Extension

Package extension that provides GPU acceleration for JuliaFEM.
Automatically loaded when user does 'using CUDA'.

Pure GPU implementation with:
- All assembly in CUDA kernels
- Zero CPU-GPU transfer during solve
- Matrix-free conjugate gradient
"""
module JuliaFEMCUDAExt

using CUDA
using Tensors
using LinearAlgebra

# Import parent package
using JuliaFEM
using JuliaFEM: Element, FieldProblem, get_connectivity
using JuliaFEM: Physics, ElasticityPhysicsType, DirichletBC, NeumannBC
using JuliaFEM: AbstractElasticityData, initialize_backend, solve_backend!, GPU

"""
ElasticityDataGPU <: AbstractElasticityData

GPU-resident data structure (all arrays on device).

Note: Internal type, not exported to users!
"""
mutable struct ElasticityDataGPU <: AbstractElasticityData
    # Geometry (nodes + connectivity)
    nodes::CuArray{Float64,2}           # 3 × n_nodes
    elements::CuArray{Int32,2}          # 4 × n_elements (Tet4)
    n_nodes::Int
    n_elements::Int

    # Material properties (per element)
    E::CuArray{Float64,1}               # Young's modulus per element
    ν::CuArray{Float64,1}               # Poisson's ratio per element

    # Boundary conditions (device)
    is_fixed::CuArray{Bool,1}           # n_dofs (true = constrained)
    prescribed::CuArray{Float64,1}      # n_dofs (value if constrained)

    # Surface loads (Neumann)
    surface_nodes::CuArray{Int32,2}     # 3 × n_surface_tri (Tri3 connectivity)
    surface_traction::CuArray{Float64,2} # 3 × n_surface_tri (traction per element)

    # Node-to-elements map (CSR format)
    node_to_elem_ptr::CuArray{Int32,1}  # n_nodes+1 (CSR pointers)
    node_to_elem_data::CuArray{Int32,1} # (element indices for each node)

    # Solution and working arrays
    u::CuArray{Float64,1}               # Solution vector (n_dofs)
    f_ext::CuArray{Float64,1}           # External forces (n_dofs)
end

"""
Physics{Elasticity} - GPU-resident elasticity problem

NOTE: Physics struct is now defined in physics_api.jl (backend-agnostic)
This module implements the GPU backend.
"""

# Alias for backward compatibility with internal code
const GPUElasticityData = ElasticityDataGPU

# ============================================================================
# Helper functions
# ============================================================================

"""
Initialize GPU data from Physics{Elasticity}

Transfer all CPU data to GPU:
1. Extract coordinates from elements
2. Build connectivity arrays
3. Extract material properties
4. Build BC flag arrays
5. Construct node-to-elements map
"""
function initialize_gpu_data!(physics::Physics{ElasticityPhysicsType}, time::Float64=0.0)
    @info "Initializing GPU data for $(physics.name)..."

    # 1. Extract geometry from elements
    n_elements = length(physics.body_elements)
    @assert n_elements > 0 "No body elements in physics!"

    # Assume Tet4 for now (can generalize later)
    first_el = physics.body_elements[1]
    conn = get_connectivity(first_el)
    nnodes_per_elem = length(conn)
    @assert nnodes_per_elem == 4 "Only Tet4 supported currently"

    # Build node set and renumber
    node_set = Set{Int}()
    for el in physics.body_elements
        for node in get_connectivity(el)
            push!(node_set, node)
        end
    end
    n_nodes = length(node_set)
    node_list = sort(collect(node_set))
    node_map = Dict(node => i for (i, node) in enumerate(node_list))

    @info "  Nodes: $n_nodes, Elements: $n_elements"

    # 2. Extract coordinates (assume first element has all nodes for now - FIXME)
    # Better: iterate all elements, extract unique nodes
    nodes_cpu = zeros(3, n_nodes)
    for el in physics.body_elements
        # Direct field access for immutable API (geometry is constant, not time-dependent)
        X = el.fields.geometry  # 3×4 matrix for Tet4
        conn = get_connectivity(el)
        for (local_idx, global_node) in enumerate(conn)
            renumbered = node_map[global_node]
            nodes_cpu[:, renumbered] = X[:, local_idx]
        end
    end

    # 3. Build connectivity (renumbered)
    elements_cpu = zeros(Int32, 4, n_elements)
    for (i, el) in enumerate(physics.body_elements)
        conn = get_connectivity(el)
        for (j, node) in enumerate(conn)
            elements_cpu[j, i] = node_map[node]
        end
    end

    # 4. Extract material properties
    E_cpu = zeros(Float64, n_elements)
    ν_cpu = zeros(Float64, n_elements)
    for (i, el) in enumerate(physics.body_elements)
        # Direct field access for immutable API (material props are constant)
        E_cpu[i] = el.fields.youngs_modulus
        ν_cpu[i] = el.fields.poissons_ratio
    end

    # 5. Build BC flag arrays
    n_dofs = 3 * n_nodes
    is_fixed_cpu = fill(false, n_dofs)
    prescribed_cpu = zeros(n_dofs)

    bc = physics.bc_dirichlet
    for (i, node_id) in enumerate(bc.node_ids)
        if haskey(node_map, node_id)
            renumbered = node_map[node_id]
            for (comp_idx, comp) in enumerate(bc.components[i])
                dof = 3 * (renumbered - 1) + comp
                is_fixed_cpu[dof] = true
                prescribed_cpu[dof] = bc.values[i][comp_idx]
            end
        end
    end

    # 6. Build surface load data
    n_surface = length(physics.bc_neumann.surface_elements)
    surface_nodes_cpu = zeros(Int32, 3, n_surface)
    surface_traction_cpu = zeros(Float64, 3, n_surface)

    for (i, surf_el) in enumerate(physics.bc_neumann.surface_elements)
        conn = get_connectivity(surf_el)
        @assert length(conn) == 3 "Only Tri3 surface elements supported"
        for (j, node) in enumerate(conn)
            if haskey(node_map, node)
                surface_nodes_cpu[j, i] = node_map[node]
            else
                error("Surface element references node $node not in body mesh")
            end
        end
        traction = physics.bc_neumann.traction[i]
        surface_traction_cpu[:, i] = [traction[1], traction[2], traction[3]]
    end

    # 7. Build node-to-elements map (CSR)
    node_to_elems = [Int32[] for _ in 1:n_nodes]
    for (el_idx, el) in enumerate(physics.body_elements)
        for node in get_connectivity(el)
            renumbered = node_map[node]
            push!(node_to_elems[renumbered], el_idx)
        end
    end

    ptr_cpu = zeros(Int32, n_nodes + 1)
    ptr_cpu[1] = 1
    for i in 1:n_nodes
        ptr_cpu[i+1] = ptr_cpu[i] + length(node_to_elems[i])
    end
    data_cpu = vcat(node_to_elems...)

    # 8. Upload to GPU
    @info "  Uploading to GPU..."
    gpu_data = ElasticityDataGPU(
        CuArray(nodes_cpu),
        CuArray(elements_cpu),
        n_nodes,
        n_elements,
        CuArray(E_cpu),
        CuArray(ν_cpu),
        CuArray(is_fixed_cpu),
        CuArray(prescribed_cpu),
        CuArray(surface_nodes_cpu),
        CuArray(surface_traction_cpu),
        CuArray(ptr_cpu),
        CuArray(data_cpu),
        CUDA.zeros(Float64, n_dofs),
        CUDA.zeros(Float64, n_dofs)
    )

    @info "  GPU initialization complete!"

    return gpu_data  # Return instead of storing in physics object!
end

"""
CUDA KERNEL: Compute element stresses at integration points
"""
function compute_element_stresses_kernel!(
    σ_gp::CuDeviceArray{SymmetricTensor{2,3,Float64,6},1},
    u::CuDeviceArray{Float64,1},
    nodes::CuDeviceArray{Float64,2},
    elements::CuDeviceArray{Int32,2},
    E_vec::CuDeviceArray{Float64,1},
    ν_vec::CuDeviceArray{Float64,1}
)
    gp_idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    n_elements = size(elements, 2)
    n_gp_per_elem = 4  # Tet4 has 4 Gauss points

    if gp_idx <= n_elements * n_gp_per_elem
        elem_idx = div(gp_idx - 1, n_gp_per_elem) + 1

        # Material
        E = E_vec[elem_idx]
        ν = ν_vec[elem_idx]

        # Extract element nodes
        n1 = elements[1, elem_idx]
        n2 = elements[2, elem_idx]
        n3 = elements[3, elem_idx]
        n4 = elements[4, elem_idx]

        # Node coordinates
        X1 = Vec{3}((nodes[1, n1], nodes[2, n1], nodes[3, n1]))
        X2 = Vec{3}((nodes[1, n2], nodes[2, n2], nodes[3, n2]))
        X3 = Vec{3}((nodes[1, n3], nodes[2, n3], nodes[3, n3]))
        X4 = Vec{3}((nodes[1, n4], nodes[2, n4], nodes[3, n4]))

        # Displacements
        u1 = Vec{3}((u[3*n1-2], u[3*n1-1], u[3*n1]))
        u2 = Vec{3}((u[3*n2-2], u[3*n2-1], u[3*n2]))
        u3 = Vec{3}((u[3*n3-2], u[3*n3-1], u[3*n3]))
        u4 = Vec{3}((u[3*n4-2], u[3*n4-1], u[3*n4]))

        # Shape derivatives (constant for Tet4)
        dN1_dxi = Vec{3}((-1.0, -1.0, -1.0))
        dN2_dxi = Vec{3}((1.0, 0.0, 0.0))
        dN3_dxi = Vec{3}((0.0, 1.0, 0.0))
        dN4_dxi = Vec{3}((0.0, 0.0, 1.0))

        # Jacobian
        J = dN1_dxi ⊗ X1 + dN2_dxi ⊗ X2 + dN3_dxi ⊗ X3 + dN4_dxi ⊗ X4
        invJ = inv(J)

        # Physical derivatives
        dN1_dx = invJ ⋅ dN1_dxi
        dN2_dx = invJ ⋅ dN2_dxi
        dN3_dx = invJ ⋅ dN3_dxi
        dN4_dx = invJ ⋅ dN4_dxi

        # Strain
        ε = symmetric(dN1_dx ⊗ u1 + dN2_dx ⊗ u2 + dN3_dx ⊗ u3 + dN4_dx ⊗ u4)

        # Stress (Hooke's law)
        λ = E * ν / ((1 + ν) * (1 - 2ν))
        μ = E / (2(1 + ν))
        I = one(ε)
        σ = λ * tr(ε) * I + 2μ * ε

        σ_gp[gp_idx] = σ
    end

    return nothing
end

"""
CUDA KERNEL: Nodal assembly (internal forces from stresses)
"""
function nodal_assembly_kernel!(
    r::CuDeviceArray{Float64,1},
    σ_gp::CuDeviceArray{SymmetricTensor{2,3,Float64,6},1},
    nodes::CuDeviceArray{Float64,2},
    elements::CuDeviceArray{Int32,2},
    node_to_elems_ptr::CuDeviceArray{Int32,1},
    node_to_elems_data::CuDeviceArray{Int32,1},
    E_vec::CuDeviceArray{Float64,1},
    ν_vec::CuDeviceArray{Float64,1}
)
    node_idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    n_nodes = size(nodes, 2)

    if node_idx <= n_nodes
        f_node = zero(Vec{3,Float64})
        gauss_weight = 1.0 / 24.0

        dN_dxi = (
            Vec{3}((-1.0, -1.0, -1.0)),
            Vec{3}((1.0, 0.0, 0.0)),
            Vec{3}((0.0, 1.0, 0.0)),
            Vec{3}((0.0, 0.0, 1.0))
        )

        elem_start = node_to_elems_ptr[node_idx]
        elem_end = node_to_elems_ptr[node_idx+1] - 1

        for elem_offset in elem_start:elem_end
            elem_idx = node_to_elems_data[elem_offset]

            n1 = elements[1, elem_idx]
            n2 = elements[2, elem_idx]
            n3 = elements[3, elem_idx]
            n4 = elements[4, elem_idx]

            local_node = 1
            if node_idx == n2
                local_node = 2
            elseif node_idx == n3
                local_node = 3
            elseif node_idx == n4
                local_node = 4
            end

            X1 = Vec{3}((nodes[1, n1], nodes[2, n1], nodes[3, n1]))
            X2 = Vec{3}((nodes[1, n2], nodes[2, n2], nodes[3, n2]))
            X3 = Vec{3}((nodes[1, n3], nodes[2, n3], nodes[3, n3]))
            X4 = Vec{3}((nodes[1, n4], nodes[2, n4], nodes[3, n4]))

            J = dN_dxi[1] ⊗ X1 + dN_dxi[2] ⊗ X2 + dN_dxi[3] ⊗ X3 + dN_dxi[4] ⊗ X4
            detJ = det(J)
            invJ = inv(J)

            dN_dx = invJ ⋅ dN_dxi[local_node]

            for local_gp in 1:4
                gp_idx = (elem_idx - 1) * 4 + local_gp
                σ = σ_gp[gp_idx]
                f_node += (dN_dx ⋅ σ) * (gauss_weight * detJ)
            end
        end

        r[3*node_idx-2] = f_node[1]
        r[3*node_idx-1] = f_node[2]
        r[3*node_idx] = f_node[3]
    end

    return nothing
end

"""
CUDA KERNEL: Apply surface traction (Neumann BC)

Integrate traction over surface elements, add to external force.
"""
function apply_surface_traction_kernel!(
    f_ext::CuDeviceArray{Float64,1},
    surface_nodes::CuDeviceArray{Int32,2},
    surface_traction::CuDeviceArray{Float64,2},
    nodes::CuDeviceArray{Float64,2}
)
    surf_idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    n_surface = size(surface_nodes, 2)

    if surf_idx <= n_surface
        # Get surface element nodes (Tri3)
        n1 = surface_nodes[1, surf_idx]
        n2 = surface_nodes[2, surf_idx]
        n3 = surface_nodes[3, surf_idx]

        # Coordinates
        X1 = Vec{3}((nodes[1, n1], nodes[2, n1], nodes[3, n1]))
        X2 = Vec{3}((nodes[1, n2], nodes[2, n2], nodes[3, n2]))
        X3 = Vec{3}((nodes[1, n3], nodes[2, n3], nodes[3, n3]))

        # Surface area (for Tri3)
        v1 = X2 - X1
        v2 = X3 - X1
        area = 0.5 * norm(v1 × v2)

        # Traction vector
        t = Vec{3}((surface_traction[1, surf_idx],
            surface_traction[2, surf_idx],
            surface_traction[3, surf_idx]))

        # Distribute equally to nodes (lumped load)
        force_per_node = (area / 3.0) * t

        # Atomic add to force vector (allows parallel writes)
        CUDA.@atomic f_ext[3*n1-2] += force_per_node[1]
        CUDA.@atomic f_ext[3*n1-1] += force_per_node[2]
        CUDA.@atomic f_ext[3*n1] += force_per_node[3]

        CUDA.@atomic f_ext[3*n2-2] += force_per_node[1]
        CUDA.@atomic f_ext[3*n2-1] += force_per_node[2]
        CUDA.@atomic f_ext[3*n2] += force_per_node[3]

        CUDA.@atomic f_ext[3*n3-2] += force_per_node[1]
        CUDA.@atomic f_ext[3*n3-1] += force_per_node[2]
        CUDA.@atomic f_ext[3*n3] += force_per_node[3]
    end

    return nothing
end

"""
CUDA KERNEL: Apply Dirichlet BC to residual

Zero out fixed DOFs in residual vector.
"""
function apply_dirichlet_kernel!(
    r::CuDeviceArray{Float64,1},
    is_fixed::CuDeviceArray{Bool,1}
)
    dof = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    n_dofs = length(r)

    if dof <= n_dofs && is_fixed[dof]
        r[dof] = 0.0
    end

    return nothing
end

"""
Compute residual: R = f_int(u) - f_ext

All on GPU, returns CuArray.
"""
function compute_residual_gpu!(
    gpu_data::GPUElasticityData,
    u::CuArray{Float64,1}
)
    n_gp = gpu_data.n_elements * 4
    n_nodes = gpu_data.n_nodes

    # Phase 1: Compute stresses
    σ_gp = CuArray{SymmetricTensor{2,3,Float64,6}}(undef, n_gp)
    threads = 256
    blocks = cld(n_gp, threads)
    @cuda threads = threads blocks = blocks compute_element_stresses_kernel!(
        σ_gp, u, gpu_data.nodes, gpu_data.elements, gpu_data.E, gpu_data.ν
    )

    # Phase 2: Nodal assembly (internal forces)
    f_int = CUDA.zeros(Float64, 3 * n_nodes)
    threads = 256
    blocks = cld(n_nodes, threads)
    @cuda threads = threads blocks = blocks nodal_assembly_kernel!(
        f_int, σ_gp, gpu_data.nodes, gpu_data.elements,
        gpu_data.node_to_elem_ptr, gpu_data.node_to_elem_data,
        gpu_data.E, gpu_data.ν
    )

    # Residual: R = f_int - f_ext
    r = f_int - gpu_data.f_ext

    # Apply Dirichlet BC (zero out fixed DOFs)
    threads = 256
    blocks = cld(length(r), threads)
    @cuda threads = threads blocks = blocks apply_dirichlet_kernel!(
        r, gpu_data.is_fixed
    )

    return r
end

"""
Matrix-free operator: K * u (for CG solver)

Computes stiffness operator action without forming K matrix.

For linear elasticity: K*u = f_int(u)
For nonlinear: K(u_current)*v = ∂f_int/∂u|_{u_current} * v (tangent operator)
"""
function stiffness_operator_gpu(
    gpu_data::GPUElasticityData,
    u::CuArray{Float64,1}
)
    # For linear elasticity: K*u = f_int(u)
    return compute_residual_gpu!(gpu_data, u)
end

"""
Matrix-free tangent operator for Newton-Krylov: K(u_current) * v

Computes action of tangent stiffness at u_current on vector v.
For linear elasticity, K is constant so K(u_current)*v = K*v = f_int(v).
For nonlinear (future), this would linearize around u_current.
"""
function tangent_operator_gpu(
    gpu_data::GPUElasticityData,
    u_current::CuArray{Float64,1},
    v::CuArray{Float64,1}
)
    # For linear elasticity: K independent of u_current
    # Just compute K*v directly
    return compute_residual_gpu!(gpu_data, v)
end

"""
    cg_solve_matfree_gpu!(Δu, rhs, gpu_data, u_current; tol, max_iter)

Matrix-free CG solver for Newton-Krylov framework.

Solves K(u_current)*Δu = rhs for the Newton update Δu.

# Arguments
- `Δu`: Solution vector (modified in-place, should be initialized to zeros)
- `rhs`: Right-hand side (-R for Newton)
- `gpu_data`: GPU data structure
- `u_current`: Current solution state (for tangent linearization)
- `tol`: Convergence tolerance for CG
- `max_iter`: Maximum CG iterations

# Returns
- `(iterations, residual)`: Number of CG iterations and final residual norm
"""
function cg_solve_matfree_gpu!(
    Δu::CuArray{Float64,1},
    rhs::CuArray{Float64,1},
    gpu_data::GPUElasticityData,
    u_current::CuArray{Float64,1};
    tol=1e-6,
    max_iter=1000
)
    n_dofs = length(Δu)

    # Tangent operator: K(u_current) * v
    K_op(v) = tangent_operator_gpu(gpu_data, u_current, v)

    # Initial residual: r = rhs - K*Δu (Δu = 0 initially)
    r = rhs - K_op(Δu)

    # Apply BC to r
    threads = 256
    blocks = cld(n_dofs, threads)
    @cuda threads = threads blocks = blocks apply_dirichlet_kernel!(r, gpu_data.is_fixed)

    p = copy(r)
    r_dot_r = dot(r, r)

    for iter in 1:max_iter
        Ap = K_op(p)

        # Apply BC to Ap
        @cuda threads = threads blocks = blocks apply_dirichlet_kernel!(Ap, gpu_data.is_fixed)

        alpha = r_dot_r / dot(p, Ap)
        Δu .+= alpha .* p
        r .-= alpha .* Ap

        r_dot_r_new = dot(r, r)

        if sqrt(r_dot_r_new) < tol
            return iter, sqrt(r_dot_r_new)
        end

        beta = r_dot_r_new / r_dot_r
        p .= r .+ beta .* p
        r_dot_r = r_dot_r_new
    end

    return max_iter, sqrt(r_dot_r)
end

"""
GPU Conjugate Gradient solver (matrix-free) - LEGACY

Old interface for linear problems. Use `solve_newton_krylov_gpu!` for new code.
"""
function cg_solve_gpu!(
    gpu_data::GPUElasticityData;
    tol=1e-6,
    max_iter=1000
)
    n_dofs = length(gpu_data.f_ext)
    u = gpu_data.u
    b = gpu_data.f_ext

    # Initial residual
    r = b - stiffness_operator_gpu(gpu_data, u)

    # Apply BC to r
    threads = 256
    blocks = cld(n_dofs, threads)
    @cuda threads = threads blocks = blocks apply_dirichlet_kernel!(
        r, gpu_data.is_fixed
    )

    p = copy(r)
    r_dot_r = dot(r, r)

    for iter in 1:max_iter
        Ap = stiffness_operator_gpu(gpu_data, p)

        # Apply BC to Ap
        @cuda threads = threads blocks = blocks apply_dirichlet_kernel!(
            Ap, gpu_data.is_fixed
        )

        alpha = r_dot_r / dot(p, Ap)
        u .+= alpha .* p
        r .-= alpha .* Ap

        r_dot_r_new = dot(r, r)

        if sqrt(r_dot_r_new) < tol
            @info "  CG converged in $iter iterations (residual: $(sqrt(r_dot_r_new)))"
            return iter, sqrt(r_dot_r_new)
        end

        beta = r_dot_r_new / r_dot_r
        p .= r .+ beta .* p
        r_dot_r = r_dot_r_new
    end

    @warn "CG did not converge in $max_iter iterations"
    return max_iter, sqrt(r_dot_r)
end

# ============================================================================
# Inexact Newton-Krylov Solver (Simultaneous Newton + CG)
# ============================================================================

"""
    solve_newton_krylov_gpu!(gpu_data, physics; kwargs...)

Inexact Newton-Krylov solver with adaptive forcing terms (Eisenstat-Walker).

Solves nonlinear elasticity problem using Newton iterations with inexact
linear solves. The key insight: don't fully converge the linear system!
Use adaptive tolerance that couples Newton and Krylov iterations.

For LINEAR problems: Converges in 1 Newton iteration (becomes pure CG).
For NONLINEAR problems: Adaptively adjusts CG tolerance per Newton step.

# Arguments
- `gpu_data`: GPU data structure
- `physics`: Physics problem definition

# Keyword Arguments
- `newton_tol=1e-6`: Newton convergence tolerance (residual norm)
- `max_newton=20`: Maximum Newton iterations
- `max_cg_per_newton=50`: Maximum CG iterations per Newton step
- `forcing_power=0.5`: Eisenstat-Walker parameter (η = ||R||^α, α ∈ [0.5, 1])
- `forcing_max=0.9`: Maximum forcing term (prevents too loose solves)
- `linear_solver=:cg`: Linear solver type (:cg only for now)

# Returns
- `(u, newton_iters, total_cg_iters, residual, history)`
  - `u`: Solution vector (CuArray)
  - `newton_iters`: Number of Newton iterations
  - `total_cg_iters`: Total CG iterations across all Newton steps
  - `residual`: Final residual norm
  - `history`: Vector of (cg_iters, R_norm, η) tuples for each Newton step

# Example
```julia
result = solve_newton_krylov_gpu!(
    gpu_data, physics;
    newton_tol=1e-6,
    max_newton=20,
    max_cg_per_newton=50
)

# result.history shows the simultaneous solving:
# [(5, 0.12, 0.35), (8, 0.034, 0.18), (12, 0.009, 0.09), ...]
#  ↑   ↑     ↑
#  CG  |R|   forcing term η
```
"""
function solve_newton_krylov_gpu!(
    gpu_data::GPUElasticityData,
    physics::Physics{ElasticityPhysicsType};
    newton_tol=1e-6,
    max_newton=20,
    max_cg_per_newton=50,
    forcing_power=0.5,
    forcing_max=0.9,
    linear_solver=:cg
)
    n_dofs = length(gpu_data.u)
    u = gpu_data.u  # Current solution (modified in-place)

    total_cg_iters = 0
    history = Tuple{Int,Float64,Float64}[]  # (cg_iters, R_norm, η)

    @info "Starting inexact Newton-Krylov solve..."
    @info "  Newton tolerance: $newton_tol"
    @info "  Max Newton iterations: $max_newton"
    @info "  Max CG per Newton: $max_cg_per_newton"
    @info "  Forcing power: $forcing_power (Eisenstat-Walker)"

    # For LINEAR problems: Directly solve K*u = f_ext (single CG solve)
    # Check if this is linear by seeing if initial residual is just f_ext
    R_initial = copy(gpu_data.f_ext)
    threads = 256
    blocks = cld(n_dofs, threads)
    @cuda threads = threads blocks = blocks apply_dirichlet_kernel!(R_initial, gpu_data.is_fixed)
    R_initial_norm = sqrt(dot(R_initial, R_initial))

    @info "Initial residual norm: $R_initial_norm (u=0)"
    @info "Detecting problem type..."

    # If initial guess is zero and we're solving linear elasticity,
    # just do one CG solve: K*u = f_ext
    if true  # Always treat as potentially linear for now
        @info "Solving K*u = f_ext (linear solve)..."

        η_k = min(forcing_max, R_initial_norm^forcing_power)
        linear_tol = η_k * R_initial_norm

        @info "  CG tolerance: $linear_tol (η=$η_k)"

        # Solve K*u = f_ext
        cg_iters, cg_residual = cg_solve_matfree_gpu!(
            u,  # Solution (starts at zero)
            gpu_data.f_ext,  # RHS
            gpu_data,
            CUDA.zeros(Float64, n_dofs);  # Linearization point (doesn't matter for linear)
            tol=linear_tol,
            max_iter=max_cg_per_newton
        )

        total_cg_iters += cg_iters
        push!(history, (cg_iters, R_initial_norm, η_k))

        @info "  CG: $cg_iters iterations (residual: $(round(cg_residual, sigdigits=3)))"

        # Apply BC to solution
        @cuda threads = threads blocks = blocks apply_dirichlet_kernel!(u, gpu_data.is_fixed)

        # Verify convergence
        f_int = compute_residual_gpu!(gpu_data, u)
        R_final = gpu_data.f_ext - f_int
        @cuda threads = threads blocks = blocks apply_dirichlet_kernel!(R_final, gpu_data.is_fixed)
        R_final_norm = sqrt(dot(R_final, R_final))

        @info "Final residual norm: $R_final_norm"

        if R_final_norm < newton_tol
            @info "✓ Converged in 1 Newton iteration (linear problem)"
            return (Array(u), 1, total_cg_iters, R_final_norm, history)
        else
            @info "Residual still large - continuing with Newton iterations..."
        end
    end

    # Nonlinear iterations (if needed)
    for newton_iter in 2:max_newton  # Start from 2 since we did one solve above
        # 1. Compute residual: R = f_ext - f_int(u)
        # For linear elasticity: f_int(u) = K*u
        f_int = compute_residual_gpu!(gpu_data, u)
        R = gpu_data.f_ext - f_int

        # Apply Dirichlet BC to residual
        threads = 256
        blocks = cld(n_dofs, threads)
        @cuda threads = threads blocks = blocks apply_dirichlet_kernel!(R, gpu_data.is_fixed)

        R_norm = sqrt(dot(R, R))

        @info "Newton iteration $newton_iter: ||R|| = $(R_norm)"

        # Check convergence
        if R_norm < newton_tol
            @info "✓ Newton converged in $newton_iter iterations!"
            @info "  Total CG iterations: $total_cg_iters"
            if newton_iter > 1
                avg_cg = total_cg_iters / newton_iter
                @info "  Average CG per Newton: $(round(avg_cg, digits=1))"
            end

            # For linear problems, just one CG solve
            if newton_iter == 1 && length(history) == 0
                # Haven't done any CG yet - solve now
                η_k = min(forcing_max, R_norm^forcing_power)
                linear_tol = η_k * R_norm

                @info "  LINEAR problem detected: solving K*u = R directly"
                @info "  CG tolerance: $linear_tol"

                Δu = CUDA.zeros(Float64, n_dofs)
                cg_iters, cg_residual = cg_solve_matfree_gpu!(
                    Δu, R, gpu_data, u;
                    tol=linear_tol,
                    max_iter=max_cg_per_newton
                )

                u .= Δu  # For linear: u = K^{-1} R
                total_cg_iters += cg_iters
                push!(history, (cg_iters, R_norm, η_k))

                @info "  CG: $cg_iters iterations (residual: $(round(cg_residual, sigdigits=3)))"

                # Apply BC to solution
                @cuda threads = threads blocks = blocks apply_dirichlet_kernel!(u, gpu_data.is_fixed)

                return (Array(u), newton_iter, total_cg_iters, cg_residual, history)
            end

            return (Array(u), newton_iter, total_cg_iters, R_norm, history)
        end

        # 2. Adaptive forcing term (Eisenstat-Walker)
        η_k = min(forcing_max, R_norm^forcing_power)
        linear_tol = η_k * R_norm

        @info "  Forcing term η = $(round(η_k, digits=3)) → linear tol = $(round(linear_tol, sigdigits=3))"

        # 3. Solve K(u)*Δu = -R inexactly (matrix-free CG)
        Δu = CUDA.zeros(Float64, n_dofs)

        cg_iters, cg_residual = cg_solve_matfree_gpu!(
            Δu,
            -R,
            gpu_data,
            u;
            tol=linear_tol,
            max_iter=max_cg_per_newton
        )

        total_cg_iters += cg_iters
        push!(history, (cg_iters, R_norm, η_k))

        @info "  CG: $cg_iters iterations (residual: $(round(cg_residual, sigdigits=3)))"

        # 4. Update solution
        u .+= Δu

        # Apply Dirichlet BC to solution
        @cuda threads = threads blocks = blocks apply_dirichlet_kernel!(u, gpu_data.is_fixed)
    end  # for newton_iter

    # Did not converge
    @warn "Newton did not converge in $max_newton iterations"
    @info "  Final residual: $(history[end][2])"
    @info "  Total CG iterations: $total_cg_iters"

    return (Array(u), max_newton, total_cg_iters, history[end][2], history)
end  # function solve_newton_krylov_gpu!

# ============================================================================
# Backend Interface Implementation
# ============================================================================

"""
    JuliaFEM.initialize_backend(::GPU, physics::Physics{ElasticityPhysicsType}, time::Float64)

Initialize GPU backend data from physics problem.

Implements AbstractBackend interface.
"""
function JuliaFEM.initialize_backend(::GPU, physics::Physics{ElasticityPhysicsType}, time::Float64)
    return initialize_gpu_data!(physics, time)
end

"""
    JuliaFEM.solve_backend!(data::ElasticityDataGPU, physics; tol, max_iter)

Solve elasticity problem on GPU using inexact Newton-Krylov.

Uses adaptive forcing terms (Eisenstat-Walker) to solve Newton and CG
iterations SIMULTANEOUSLY rather than in nested loops.

For linear problems: Converges in 1 Newton iteration (becomes pure CG).
For nonlinear problems: Adaptively adjusts linear solve tolerance.

Implements AbstractBackend interface.

Returns: (u, newton_iterations, total_cg_iterations, residual, history)
"""
function JuliaFEM.solve_backend!(data::ElasticityDataGPU, physics::Physics{ElasticityPhysicsType};
    tol=1e-6, max_iter=1000, newton_tol=1e-6, max_newton=20, max_cg_per_newton=50)
    # Compute external forces (Neumann BC)
    @info "Computing external forces..."
    fill!(data.f_ext, 0.0)

    n_surface = size(data.surface_nodes, 2)
    if n_surface > 0
        threads = 256
        blocks = cld(n_surface, threads)
        @cuda threads = threads blocks = blocks apply_surface_traction_kernel!(
            data.f_ext,
            data.surface_nodes,
            data.surface_traction,
            data.nodes
        )
    end

    # Solve with inexact Newton-Krylov
    @info "Solving with inexact Newton-Krylov..."
    u, newton_iters, total_cg_iters, residual, history = solve_newton_krylov_gpu!(
        data, physics;
        newton_tol=newton_tol,
        max_newton=max_newton,
        max_cg_per_newton=max_cg_per_newton
    )

    # Return in the format expected by backend interface
    # Include Newton-Krylov statistics
    return (u, newton_iters, total_cg_iters, residual, history)
end

# ============================================================================
# Backward Compatibility
# ============================================================================

"""
    solve_elasticity_gpu!(physics::Physics{ElasticityPhysicsType}; time, tol, max_iter)

Backward compatibility wrapper for direct GPU solve calls.

**Deprecated:** Use `solve!(physics; backend=GPU())` instead.
"""
function solve_elasticity_gpu!(physics::Physics{ElasticityPhysicsType}; time=0.0, tol=1e-6, max_iter=1000)
    # Use the unified solve! interface
    return JuliaFEM.solve!(physics; backend=GPU(), time=time, tol=tol, max_iter=max_iter)
end

end  # module JuliaFEMCUDAExt
