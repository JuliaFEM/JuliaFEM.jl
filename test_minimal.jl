#!/usr/bin/env julia
# Minimal test to debug what's broken

using JuliaFEM

println("✓ JuliaFEM loaded")

# Test 1: Can we create an element?
try
    element = Element(Seg2, (1, 2))
    println("✓ Element created: ", typeof(element))
    println("  length: ", length(element))
catch e
    println("✗ Element creation failed: ", e)
end

# Test 2: Can we create a Problem?
try
    problem = Problem(Dirichlet, "test", 1, "temperature")
    println("✓ Problem created: ", typeof(problem))
catch e
    println("✗ Problem creation failed: ", e)
end

# Test 3: Can we update element fields?
try
    element = Element(Seg2, (1, 2))
    X = Dict(1 => [0.0, 0.0], 2 => [6.0, 0.0])
    update!(element, "geometry", X)
    println("✓ Element update successful")
catch e
    println("✗ Element update failed: ", e)
    println("  Stacktrace:")
    for (exc, bt) in Base.catch_stack()
        showerror(stdout, exc, bt)
        println()
    end
end

println("\n=== Summary ===")
println("If all checks pass, the core API works!")
