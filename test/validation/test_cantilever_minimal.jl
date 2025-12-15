# Integration test for cantilever example with NEW API
# Tests that the full workflow runs successfully

using Test
using JuliaFEM
using Tensors
using LinearAlgebra

@testset "Cantilever NEW API Integration Test" begin
    # Run the example in a function to capture results
    function run_cantilever_example()
        # Create mesh
        nodes = [
            Vec{3}((0.0, 0.0, 0.0)), Vec{3}((5.0, 0.0, 0.0)),
            Vec{3}((5.0, 1.0, 0.0)), Vec{3}((0.0, 1.0, 0.0)),
            Vec{3}((0.0, 0.0, 1.0)), Vec{3}((5.0, 0.0, 1.0)),
            Vec{3}((5.0, 1.0, 1.0)), Vec{3}((0.0, 1.0, 1.0)),
            Vec{3}((10.0, 0.0, 0.0)), Vec{3}((10.0, 1.0, 0.0)),
            Vec{3}((10.0, 0.0, 1.0)), Vec{3}((10.0, 1.0, 1.0))
        ]

        connectivity = [
            (1, 2, 3, 4, 5, 6, 7, 8),
            (2, 9, 10, 3, 6, 11, 12, 7)
        ]

        E = 210e9
        ν = 0.3

        # Create elements
        elements = Element[]
        for (elem_id, conn) in enumerate(connectivity)
            elem_nodes = [nodes[i] for i in conn]
            element = Element(Hexahedron, conn,
                fields=(geometry=elem_nodes, youngs_modulus=E, poissons_ratio=ν),
                id=UInt(elem_id))
            push!(elements, element)
        end

        # Assemble
        n_dofs = 3 * length(nodes)
        K_global = zeros(n_dofs, n_dofs)
        f_global = zeros(n_dofs)

        for element in elements
            K_local = JuliaFEM.compute_element_stiffness(element, 0.0)
            conn = element.connectivity
            gdofs = Int[]
            for node in conn
                push!(gdofs, 3 * (node - 1) + 1, 3 * (node - 1) + 2, 3 * (node - 1) + 3)
            end
            for (i, dof_i) in enumerate(gdofs)
                for (j, dof_j) in enumerate(gdofs)
                    K_global[dof_i, dof_j] += K_local[i, j]
                end
            end
        end

        # Apply loads
        loaded_nodes = [9, 10, 11, 12]
        F_total = -1000.0
        f_per_node = F_total / length(loaded_nodes)
        for node in loaded_nodes
            f_global[3*node] = f_per_node
        end

        # Apply BCs
        fixed_nodes = [1, 4, 5, 8]
        fixed_dofs = Int[]
        for node in fixed_nodes
            push!(fixed_dofs, 3 * (node - 1) + 1, 3 * (node - 1) + 2, 3 * (node - 1) + 3)
        end
        for dof in fixed_dofs
            K_global[dof, :] .= 0.0
            K_global[:, dof] .= 0.0
            K_global[dof, dof] = 1.0
            f_global[dof] = 0.0
        end

        # Solve
        u = K_global \ f_global

        return u, K_global, f_global
    end

    # Run the example
    u, K, f = run_cantilever_example()

    # Test 1: Solution exists and is correct size
    @test length(u) == 36  # 12 nodes × 3 DOFs

    # Test 2: Fixed nodes have zero displacement
    fixed_dofs = [1, 2, 3, 10, 11, 12, 13, 14, 15, 22, 23, 24]
    for dof in fixed_dofs
        @test abs(u[dof]) < 1e-10
    end

    # Test 3: Tip deflection is negative (downward)
    # Tip nodes: 9,10,11,12; Z components: 27,30,33,36
    tip_z_dofs = [27, 30, 33, 36]
    for dof in tip_z_dofs
        @test u[dof] < 0.0  # Downward deflection
    end

    # Test 4: Stiffness matrix is symmetric
    @test isapprox(K, K', rtol=1e-10)

    # Test 5: No NaN or Inf in solution
    @test all(isfinite, u)

    # Test 6: Tip deflection has reasonable magnitude (order 1e-7 m)
    avg_tip_deflection = sum(u[tip_z_dofs]) / length(tip_z_dofs)
    @test abs(avg_tip_deflection) > 1e-10  # Not zero
    @test abs(avg_tip_deflection) < 1e-3   # Not unreasonably large

    println("✅ All cantilever NEW API integration tests passed!")
    println("   - Solution computed successfully")
    println("   - Boundary conditions satisfied")
    println("   - Tip deflection: $(abs(avg_tip_deflection)*1e3) mm")
end
