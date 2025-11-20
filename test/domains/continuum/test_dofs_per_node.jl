# Test dofs_per_node function

@testset "dofs_per_node" begin
    kernel = create_test_kernel()

    # Test that it returns 3 for 3D displacement field
    @test dofs_per_node(kernel) == 3

    # Test zero allocations
    allocs = @allocated dofs_per_node(kernel)
    @test allocs == 0
end
