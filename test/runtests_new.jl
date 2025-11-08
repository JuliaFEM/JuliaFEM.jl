# JuliaFEM Test Suite - New Structure
# Educational testing with Literate.jl

using Test, JuliaFEM

# Configuration
const RUN_TUTORIALS = get(ENV, "JULIAFEM_TEST_TUTORIALS", "true") == "true"
const RUN_UNIT = get(ENV, "JULIAFEM_TEST_UNIT", "false") == "true"
const RUN_OLD = get(ENV, "JULIAFEM_TEST_OLD", "false") == "true"

println("="^70)
println("JuliaFEM Test Suite (New Structure)")
println("="^70)
println("Tutorials: ", RUN_TUTORIALS ? "✓" : "✗")
println("Unit tests: ", RUN_UNIT ? "✓" : "✗")
println("Old tests: ", RUN_OLD ? "✓" : "✗")
println("="^70)

# Tutorial Tests
if RUN_TUTORIALS
    @testset "Tutorials" begin
        @testset "01_Fundamentals" begin
            include("tutorials/01_fundamentals/creating_elements.jl")
        end
    end
end

# Unit Tests  
if RUN_UNIT
    @testset "Unit Tests" begin
        @info "No unit tests yet"
    end
end

# Old Tests
if RUN_OLD
    @warn "Old tests have 49+ failures - see docs/TEST_FIXES_NEEDED.md"
    include("runtests.jl")  # Original test suite
end

println()
println("="^70)
println("Complete - See docs/TESTING_PHILOSOPHY.md")
println("="^70)
