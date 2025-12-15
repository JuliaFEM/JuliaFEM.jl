# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
Test microkernel utilities with Element{K,P,S}.

Tests helper functions that work with existing Element system.
No redundant FormulationSpec - just use Element{K,P,S} directly!
"""

using Test
using JuliaFEM

@testset "Field Type Extraction from Element{K,P,S}" begin
    # Define multi-field specification
    S = @DOFSet{T::DOF{Temperature,Vertex}, u::DOF{Displacement{3},Vertex}}
    
    @test fieldnames(S) == (:T, :u)
    
    # Extract field types for dispatch
    field_T = field_type_for_dispatch(S, :T)
    field_u = field_type_for_dispatch(S, :u)
    
    @test field_T isa Temperature
    @test field_u isa Displacement{3}
end

@testset "Type Aliases Work" begin
    # Using the convenience alias
    elem_type = Element{Tet4, Lagrange{1}, ThermoelasticityFields}
    
    S = dof_type(elem_type)
    @test fieldnames(S) == (:T, :u)
end

println("✓ All microkernel utilities tests passed!")
println("  - Field type extraction from Element{K,P,S} works")
println("  - Type aliases work")
println("  - No redundant abstractions - just use Element directly!")
