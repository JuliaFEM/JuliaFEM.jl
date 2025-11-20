# Test get_dof_mapping! function

@testset "get_dof_mapping!" begin
    kernel = create_test_kernel()
    mesh = create_test_mesh()

    # Allocate DOF buffer (8 nodes * 3 DOFs = 24)
    dofs = zeros(Int, 24)

    # Test DOF mapping for element 1
    get_dof_mapping!(dofs, kernel, 1, mesh)

    # Verify correct DOF indices (1-indexed)
    # For element 1 with nodes [1,2,3,4,5,6,7,8]:
    # Node 1: DOFs [1,2,3], Node 2: DOFs [4,5,6], etc.
    expected_dofs = collect(1:24)
    @test dofs == expected_dofs

    # Test zero allocations (warm-up call first)
    get_dof_mapping!(dofs, kernel, 1, mesh)
    allocs = @allocated get_dof_mapping!(dofs, kernel, 1, mesh)
    @test allocs == 0
end
