# Shared helper functions for kernel allocation tests

using JuliaFEM
using Tensors
using LinearAlgebra

"""
    create_test_mesh() -> Mesh{8, Hex8}

Create a single Hex8 element mesh for testing.
"""
function create_test_mesh()
    # Single Hex8 element with 8 nodes
    # Node coordinates for a unit cube
    X = Vec{3,Float64}[
        Vec{3}((0.0, 0.0, 0.0)),  # Node 1
        Vec{3}((1.0, 0.0, 0.0)),  # Node 2
        Vec{3}((1.0, 1.0, 0.0)),  # Node 3
        Vec{3}((0.0, 1.0, 0.0)),  # Node 4
        Vec{3}((0.0, 0.0, 1.0)),  # Node 5
        Vec{3}((1.0, 0.0, 1.0)),  # Node 6
        Vec{3}((1.0, 1.0, 1.0)),  # Node 7
        Vec{3}((0.0, 1.0, 1.0)),  # Node 8
    ]

    # Single element connectivity (must be UInt32)
    connectivity = [NTuple{8,UInt32}((1, 2, 3, 4, 5, 6, 7, 8))]

    return Mesh{8,Hex8}(X, connectivity)
end

"""
    create_test_kernel() -> ContinuumKernel

Create a ContinuumKernel with LinearElastic material for testing.
"""
function create_test_kernel()
    # Material properties: Steel-like
    E = 200.0e9  # Young's modulus (Pa)
    ν = 0.3      # Poisson's ratio

    # Create linear elastic material
    material = LinearElastic(E, ν)

    # Create continuum formulation (3D elasticity)
    formulation = ContinuumFormulation{FullThreeD}()

    # Create kernel (field is optional, defaults to Displacement{3})
    kernel = ContinuumKernel(formulation, material)

    return kernel
end

"""
    create_material_state(kernel::AbstractKernel, mesh::AbstractMesh)

Create initial material state for all elements (for stateless materials, returns nothing).
"""
function create_material_state(kernel::AbstractKernel, mesh::AbstractMesh)
    # For stateless materials (LinearElastic, NeoHookean), state is nothing
    if !needs_state(kernel.material)
        return nothing
    end

    # For stateful materials, initialize state per element
    # This would need to be implemented for plasticity tests
    error("Stateful material state initialization not yet implemented")
end
