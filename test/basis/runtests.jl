# Test suite for basis functions module
using Test

@testset "Basis Functions" begin
    include("test_partition_of_unity.jl")
    include("test_svector_returns.jl")
    include("test_derivatives.jl")
    include("test_interpolation.jl")
    include("test_all_elements.jl")
    include("test_kronecker_delta.jl")
    include("test_type_stability.jl")
    include("test_numerical_accuracy.jl")
    include("test_gradient_properties.jl")
end
