"""
Main solver: Elasticity on GPU

Solves linear elasticity using:
- Two-phase nodal assembly (no atomics)
- Matrix-free conjugate gradient
- GPU-resident throughout
"""
function solve_elasticity_gpu(physics::ElasticityPhysics; tol=1e-6, max_iter=1000)

module GPUElasticity

export solve_elasticity_gpu, ElasticityPhysics, ElasticMaterial

using CUDA
using Tensors
using LinearAlgebra
using Printf

# Re-export mesh reader
include("gmsh_reader.jl")
using .GmshReader
export read_gmsh_mesh, GmshMesh, get_surface_nodes

"""
Elastic material properties
"""
struct ElasticMaterial
    E::Float64   # Young's modulus [Pa]
    ν::Float64   # Poisson's ratio [-]
end

"""
Elasticity physics definition
"""
struct ElasticityPhysics
    mesh::GmshMesh
    material::ElasticMaterial
    fixed_nodes::Vector{Int}        # Dirichlet BC (fixed displacement)
    pressure_nodes::Vector{Int}     # Neumann BC (pressure load)
    pressure_value::Float64         # Pressure magnitude [Pa]
end

"""
Node-to-elements connectivity (CSR format)
"""
struct NodeToElementsMap
    ptr::CuArray{Int32,1}
    data::CuArray{Int32,1}
end

"""
Build CSR map: which elements touch each node?
"""
function build_node_to_elems_gpu(elements::Matrix{Int}, n_nodes::Int)
    # Count connections per node
    counts = zeros(Int, n_nodes)
    for elem_idx in 1:size(elements, 2)
        for i in 1:4
            node = elements[i, elem_idx]
            counts[node] += 1
        end
    end

    # Build CSR structure
    ptr = cumsum([1; counts])
    data = Vector{Int32}(undef, sum(counts))

    # Fill data array
    offset = copy(ptr[1:end-1])
    for elem_idx in 1:size(elements, 2)
        for i in 1:4
            node = elements[i, elem_idx]
            data[offset[node]] = elem_idx
            offset[node] += 1
        end
    end

    return NodeToElementsMap(CuArray(Int32.(ptr)), CuArray(data))
end

"""
PHASE 1 GPU KERNEL: Compute element stiffness contributions at integration points

For LINEAR ELASTICITY (no plasticity), we don't need state variables.
Just compute stresses from strains using Hooke's law.
"""
function compute_element_stresses_kernel!(
    σ_gp::CuDeviceArray{SymmetricTensor{2,3,Float64,6},1},
    u::CuDeviceArray{Float64,1},
    nodes::CuDeviceArray{Float64,2},
    elements::CuDeviceArray{Int32,2},
    E, ν
)
    gp_idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    if gp_idx <= length(σ_gp)
        # Map GP to element
        elem_idx = (gp_idx - 1) ÷ 4 + 1  # 4 GPs per Tet4

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

        # Strain (small strain assumption)
        ε = symmetric(dN1_dx ⊗ u1 + dN2_dx ⊗ u2 + dN3_dx ⊗ u3 + dN4_dx ⊗ u4)

        # Stress (Hooke's law)
        λ = E * ν / ((1 + ν) * (1 - 2ν))
        μ = E / (2(1 + ν))
        I = one(ε)
        σ = λ * tr(ε) * I + 2μ * ε

        # Store result
        σ_gp[gp_idx] = σ
    end

    return nothing
end

"""
PHASE 2 GPU KERNEL: Nodal assembly (matrix-free, no atomics!)
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
        # Accumulate forces
        f_node = zero(Vec{3,Float64})

        # Gauss weight for Tet4
        gauss_weight = 1.0 / 24.0

        # Shape derivatives
        dN_dxi = (
            Vec{3}((-1.0, -1.0, -1.0)),
            Vec{3}((1.0, 0.0, 0.0)),
            Vec{3}((0.0, 1.0, 0.0)),
            Vec{3}((0.0, 0.0, 1.0))
        )

        # Get element range for this node
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

            # Find local node index
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
                σ = σ_gp[gp_idx]

                # Accumulate force
                f_node += (dN_dx ⋅ σ) * (gauss_weight * detJ)
            end
        end

        # Write result (no atomics!)
        r[3*node_idx-2] = f_node[1]
        r[3*node_idx-1] = f_node[2]
        r[3*node_idx] = f_node[3]
    end

    return nothing
end

"""
Compute residual on GPU (internal forces)
"""
function compute_residual_gpu!(
    r::CuArray{Float64,1},
    u::CuArray{Float64,1},
    nodes::CuArray{Float64,2},
    elements::CuArray{Int32,2},
    node_to_elems::NodeToElementsMap,
    E, ν
)
    n_gp = size(elements, 2) * 4
    n_nodes = size(nodes, 2)

    # Phase 1: Compute stresses at GPs
    σ_gp = CuArray{SymmetricTensor{2,3,Float64,6}}(undef, n_gp)

    threads = 256
    blocks = cld(n_gp, threads)
    @cuda threads = threads blocks = blocks compute_element_stresses_kernel!(
        σ_gp, u, nodes, elements, E, ν
    )

    # Phase 2: Nodal assembly
    fill!(r, 0.0)

    threads = 256
    blocks = cld(n_nodes, threads)
    @cuda threads = threads blocks = blocks nodal_assembly_kernel!(
        r, σ_gp, nodes, elements,
        node_to_elems.ptr, node_to_elems.data
    )

    return r
end

"""
Apply pressure load to top surface (Neumann BC)
"""
function apply_pressure_load!(
    f::CuArray{Float64,1},
    pressure_nodes::Vector{Int},
    mesh::GmshMesh,
    pressure::Float64
)
    # Simple uniform distribution (should integrate properly over surface)
    # For now, divide pressure equally among nodes

    f_cpu = Array(f)
    n_pressure_nodes = length(pressure_nodes)

    # Estimate surface area (assuming uniform Z = height)
    surface_area = (maximum(mesh.nodes[1, :]) - minimum(mesh.nodes[1, :])) *
                   (maximum(mesh.nodes[2, :]) - minimum(mesh.nodes[2, :]))

    # Total force
    total_force = pressure * surface_area
    force_per_node = total_force / n_pressure_nodes

    # Apply in Z direction (negative, pointing down)
    for node in pressure_nodes
        f_cpu[3*node] += -force_per_node  # Z component
    end

    copyto!(f, f_cpu)

    return f
end

"""
Apply Dirichlet boundary conditions (fixed nodes)
"""
function apply_dirichlet_bc!(
    K_op::Function,
    f::CuArray{Float64,1},
    fixed_nodes::Vector{Int}
)
    # Zero out DOFs
    f_cpu = Array(f)
    for node in fixed_nodes
        f_cpu[3*node-2] = 0.0  # X
        f_cpu[3*node-1] = 0.0  # Y
        f_cpu[3*node] = 0.0    # Z
    end
    copyto!(f, f_cpu)

    # Return modified operator that zeros fixed DOFs
    function K_bc(u)
        r = K_op(u)
        r_cpu = Array(r)
        for node in fixed_nodes
            r_cpu[3*node-2] = 0.0
            r_cpu[3*node-1] = 0.0
            r_cpu[3*node] = 0.0
        end
        copyto!(r, r_cpu)
        return r
    end

    return K_bc
end

"""
Conjugate Gradient solver (GPU)
"""
function cg_solve_gpu!(
    x::CuArray{Float64,1},
    A_op::Function,
    b::CuArray{Float64,1};
    tol=1e-6,
    max_iter=1000
)
    n = length(x)

    # Initial residual
    r = b - A_op(x)
    p = copy(r)
    rsold = dot(r, r)

    println("\nConjugate Gradient solver:")
    println("  Initial residual: $(sqrt(rsold))")

    for iter in 1:max_iter
        Ap = A_op(p)
        alpha = rsold / dot(p, Ap)

        x .+= alpha .* p
        r .-= alpha .* Ap

        rsnew = dot(r, r)

        if iter % 10 == 0 || iter == 1
            @printf("  Iter %4d: ||r|| = %.6e\n", iter, sqrt(rsnew))
        end

        if sqrt(rsnew) < tol
            println("  ✅ Converged in $iter iterations")
            return x, iter
        end

        beta = rsnew / rsold
        p .= r .+ beta .* p
        rsold = rsnew
    end

    println("  ❌ Did not converge in $max_iter iterations")
    return x, max_iter
end

"""
Solve linear elasticity problem on GPU
"""
function solve_elasticity_gpu(problem::ElasticityProblem; tol=1e-6, max_iter=1000)
    println("\n" * "="^70)
    println("GPU Linear Elasticity Solver")
    println("="^70)

    # Check CUDA
    if !CUDA.functional()
        error("CUDA not available!")
    end
    println("GPU: ", CUDA.name(CUDA.device()))

    # Extract mesh data
    mesh = physics.mesh
    n_nodes = size(mesh.nodes, 2)
    n_elems = size(mesh.elements, 2)
    n_dofs = 3 * n_nodes

    println("\nMesh:")
    println("  Nodes:    $n_nodes")
    println("  Elements: $n_elems")
    println("  DOFs:     $n_dofs")

    println("\nBoundary conditions:")
    println("  Fixed nodes:    $(length(physics.fixed_nodes))")
    println("  Pressure nodes: $(length(physics.pressure_nodes))")
    println("  Pressure value: $(physics.pressure_value) Pa")

    println("\nMaterial:")
    println("  Young's modulus: $(physics.material.E) Pa")
    println("  Poisson's ratio: $(physics.material.ν)")

    # Transfer to GPU
    println("\nTransferring data to GPU...")
    nodes_gpu = CuArray(mesh.nodes)
    elements_gpu = CuArray(Int32.(mesh.elements))

    # Build CSR map
    println("Building node-to-elements map...")
    node_to_elems = build_node_to_elems_gpu(mesh.elements, n_nodes)

    # Initial guess
    u_gpu = CUDA.zeros(Float64, n_dofs)

    # External force (pressure load)
    f_gpu = CUDA.zeros(Float64, n_dofs)
    apply_pressure_load!(f_gpu, physics.pressure_nodes, mesh, physics.pressure_value)

    println("External force norm: $(norm(Array(f_gpu)))")

    # Define stiffness operator K(u) = internal forces
    E = physics.material.E
    ν = physics.material.ν

    function K_op(u)
        r = CUDA.zeros(Float64, n_dofs)
        compute_residual_gpu!(r, u, nodes_gpu, elements_gpu, node_to_elems, E, ν)
        return r
    end

    # Apply Dirichlet BC
    K_bc = apply_dirichlet_bc!(K_op, f_gpu, physics.fixed_nodes)

    # Solve: K * u = f
    println("\n" * "-"^70)
    println("Solving linear system...")
    println("-"^70)

    u_gpu, n_iter = cg_solve_gpu!(u_gpu, K_bc, f_gpu, tol=tol, max_iter=max_iter)

    # Transfer back to CPU
    u_cpu = Array(u_gpu)

    println("\n" * "="^70)
    println("Solution statistics:")
    println("="^70)
    println("  Max displacement: $(maximum(abs.(u_cpu))) m")
    println("  CG iterations:    $n_iter")

    # Compute final residual
    r_final = K_bc(u_gpu) - f_gpu
    println("  Final residual:   $(norm(Array(r_final)))")

    println("\n" * "="^70)
    println("✅ GPU elasticity solver complete!")
    println("="^70)

    return u_cpu
end

end  # module
