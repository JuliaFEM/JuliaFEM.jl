"""
# Modal Analysis - NEW API (Test-Driven Development)

**What:** Shows how modal (eigenvalue) analysis SHOULD work with the NEW API

**Why:**
- **Eigenvalue problems** - Different solver type (not Newton-Krylov)
- **Natural frequencies** - Validates dynamic behavior
- **Geometric stiffness** - Prestress effects on frequencies
- **Multiple physics** - Works for elasticity AND heat transfer
- **Mass matrix required** - First time we need M, not just K

**NEW API Concepts:**
1. **ModalSolver** - Eigenvalue solver (not LinearSolver or NewtonSolver)
2. **EigenvalueProblem** - K φ = λ M φ (generalized eigenvalue problem)
3. **Geometric stiffness** - K_g from initial stress/displacement
4. **Modal shapes** - φ_i (eigenvectors = mode shapes)
5. **Natural frequencies** - ω_i = √λ_i

**Test Problems:**

## Test 1: Single Tet4 Element
- 4-node tetrahedron with 3 nodes fixed
- Validates basic eigenvalue computation
- Tests with and without geometric stiffness

## Test 2: Heat Transfer Modal Analysis
- Two Quad4 elements (plane heat)
- Eigenvalues of thermal diffusion operator
- Validates modal analysis works for temperature field

## Test 3: Vibrating Beam (Future)
- Fixed-fixed beam
- Compare natural frequencies to analytical solution
- ω_i = λ_i² √(EI/ρA)

**Expected Behavior (when implemented):**
✅ Eigenvalues computed correctly
✅ Mode shapes orthogonal (φ_i^T M φ_j = δ_ij)
✅ Geometric stiffness affects frequencies
✅ Works for both elasticity and heat transfer
✅ Sparse eigenvalue solver (only need first k modes)

**Status:** 🚧 VISIONARY TEST - Implementation in progress
"""

using Test
using JuliaFEM
using Tensors
using LinearAlgebra
using Statistics

@testset "Modal Analysis - NEW API (TDD)" begin

    # =============================================================================
    # SINGLE ELEMENT: TET4 WITHOUT GEOMETRIC STIFFNESS
    # =============================================================================

    @testset "Single Tet4 Modal Analysis (Visionary)" begin
        @test_skip begin  # Skip until implemented

            # Geometry
            X = Dict(
                1 => Vec(2.0, 3.0, 4.0),
                2 => Vec(6.0, 3.0, 2.0),
                3 => Vec(2.0, 5.0, 1.0),
                4 => Vec(4.0, 3.0, 6.0)
            )

            # Initial displacement (for geometric stiffness later)
            u0 = Dict(
                1 => Vec(0.0, 0.0, 0.0),
                2 => Vec(0.0, 0.0, 0.0),
                3 => Vec(0.0, 0.0, 0.0),
                4 => Vec(0.25, 0.25, 0.25)
            )

            # Material
            material = LinearElastic(E=96.0, ν=1 / 3, ρ=420.0)

            # Physics
            elastic_physics = ContinuumPhysics{Displacement}(
                material=material,
                formulation=FullThreeD(),
                finite_strain=false,
                geometric_stiffness=false    # No K_g initially
            )

            # Create element
            tet_element = create_element(
                Tet4, (1, 2, 3, 4),
                geometry=X,
                displacement=u0
            )

            # Domain
            domain = Domain(
                name="TET",
                elements=[tet_element],
                physics=elastic_physics
            )

            # Boundary conditions (3 nodes fixed)
            bc_fixed = DirichletBC(
                name="FIXED_FACE",
                nodes=[1, 2, 3],
                dof=:displacement,
                values=[0.0, 0.0, 0.0]
            )

            # NEW: EigenvalueProblem (not static or transient)
            problem = EigenvalueProblem(
                domains=[domain],
                boundary_conditions=[bc_fixed]
            )

            # NEW: ModalSolver
            solver = ModalSolver(
                n_modes=2,              # Number of modes to compute
                which=:SM,              # Smallest Magnitude (or :LM for largest)
                method=:arpack          # ARPACK, KrylovKit, or direct
            )

            # Solve: K φ = λ M φ
            solution = solve!(problem, solver)

            # Extract eigenvalues and eigenvectors
            λ = solution.eigenvalues     # [λ_1, λ_2]
            φ = solution.eigenvectors    # [φ_1, φ_2]

            # Validate eigenvalues (without geometric stiffness)
            @test isapprox(λ, [4 / 3, 1 / 3], rtol=1e-6)

            # Validate orthogonality: φ_i^T M φ_j = δ_ij
            for i in 1:2
                for j in 1:2
                    orthogonality = φ[i]' * solution.mass_matrix * φ[j]
                    expected = (i == j) ? 1.0 : 0.0
                    @test isapprox(orthogonality, expected, atol=1e-6)
                end
            end

        end
    end

    # =============================================================================
    # GEOMETRIC STIFFNESS EFFECT
    # =============================================================================

    @testset "Tet4 with Geometric Stiffness (Visionary)" begin
        @test_skip begin

            # Same setup as before, but enable geometric stiffness
            elastic_physics = ContinuumPhysics{Displacement}(
                material=LinearElastic(E=96.0, ν=1 / 3, ρ=420.0),
                formulation=FullThreeD(),
                finite_strain=false,
                geometric_stiffness=true     # NEW: Enable K_g
            )

            problem = EigenvalueProblem(
                domains=[domain],
                boundary_conditions=[bc_fixed],
                initial_displacement=u0       # Required for K_g
            )

            solver = ModalSolver(n_modes=2, which=:SM)
            solution = solve!(problem, solver)

            λ = solution.eigenvalues

            # With geometric stiffness: K_eff = K + K_g(u0)
            # Eigenvalues should be different!
            @test isapprox(λ, [5 / 3, 2 / 3], rtol=1e-6)

            # Geometric stiffness changes natural frequencies
            @test λ[1] > 4 / 3  # Stiffened by prestress
            @test λ[2] > 1 / 3

        end
    end

    # =============================================================================
    # HEAT TRANSFER MODAL ANALYSIS
    # =============================================================================

    @testset "Heat Transfer Modal (Visionary)" begin
        @test_skip begin

            # Geometry: Two quad elements stacked vertically
            X = Dict(
                1 => Vec(0.0, 0.0),
                2 => Vec(1.0, 0.0),
                3 => Vec(1.0, 3.0),
                4 => Vec(0.0, 3.0),
                5 => Vec(0.0, 3.0),
                6 => Vec(1.0, 3.0),
                7 => Vec(1.0, 9.0),
                8 => Vec(0.0, 9.0)
            )

            # Material (thermal)
            thermal_material = ThermalMaterial(
                conductivity=36.0,
                density=6.0,
                specific_heat=1.0
            )

            # Physics
            thermal_physics = ContinuumPhysics{Temperature}(
                material=thermal_material,
                formulation=PlaneHeat()
            )

            # Elements
            elem1 = create_element(Quad4, (1, 2, 3, 4), geometry=X)
            elem2 = create_element(Quad4, (4, 3, 7, 8), geometry=X)

            # Domain
            domain = Domain(
                name="THERMAL",
                elements=[elem1, elem2],
                physics=thermal_physics
            )

            # Boundary conditions (fixed temperature at ends)
            bc_bottom = DirichletBC(
                nodes=[1, 2],
                dof=:temperature,
                value=0.0
            )

            bc_top = DirichletBC(
                nodes=[7, 8],
                dof=:temperature,
                value=0.0
            )

            # Eigenvalue problem for heat equation
            # M ∂T/∂t + K T = 0
            # Modal: K φ = λ M φ
            problem = EigenvalueProblem(
                domains=[domain],
                boundary_conditions=[bc_bottom, bc_top]
            )

            solver = ModalSolver(n_modes=1, which=:SM)
            solution = solve!(problem, solver)

            λ = solution.eigenvalues

            # First eigenvalue should be 1.0 (analytical)
            @test isapprox(λ[1], 1.0, rtol=1e-6)

        end
    end

    # =============================================================================
    # NATURAL FREQUENCIES FROM EIGENVALUES
    # =============================================================================

    @testset "Natural Frequency Computation (Visionary)" begin
        @test_skip begin

            # Solve modal problem
            solution = solve_modal_problem()

            # Extract eigenvalues
            λ = solution.eigenvalues  # [rad²/s²]

            # Natural frequencies: ω = √λ [rad/s]
            ω = sqrt.(λ)

            # Convert to Hz: f = ω / (2π)
            f = ω ./ (2π)

            # Validate units and values
            @test all(λ .≥ 0)  # Eigenvalues non-negative
            @test all(ω .≥ 0)  # Frequencies non-negative
            @test all(f .≥ 0)

            # For simple beam: f_1 ≈ 1-10 Hz (typical)
            @test 1.0 < f[1] < 100.0

        end
    end

    # =============================================================================
    # MODE SHAPE VISUALIZATION (FUTURE)
    # =============================================================================

    @testset "Mode Shape Properties (Visionary)" begin
        @test_skip begin

            solution = solve_modal_problem()

            φ = solution.eigenvectors  # Mode shapes
            M = solution.mass_matrix
            K = solution.stiffness_matrix
            λ = solution.eigenvalues

            # Property 1: Orthogonality w.r.t. mass matrix
            # φ_i^T M φ_j = δ_ij
            for i in 1:length(φ)
                for j in 1:length(φ)
                    orth_M = φ[i]' * M * φ[j]
                    expected = (i == j) ? 1.0 : 0.0
                    @test isapprox(orth_M, expected, atol=1e-6)
                end
            end

            # Property 2: Eigenvalue equation
            # K φ_i = λ_i M φ_i
            for i in 1:length(φ)
                Kφ = K * φ[i]
                λMφ = λ[i] * (M * φ[i])
                @test Kφ ≈ λMφ rtol = 1e-6
            end

            # Property 3: Mode shapes normalized
            # φ_i^T M φ_i = 1
            for i in 1:length(φ)
                norm_M = φ[i]' * M * φ[i]
                @test isapprox(norm_M, 1.0, rtol=1e-6)
            end

        end
    end

    # =============================================================================
    # MASS MATRIX ASSEMBLY
    # =============================================================================

    @testset "Mass Matrix Assembly (Visionary)" begin
        # Pseudo-code showing mass matrix assembly

        println("\n" * "="^70)
        println("MASS MATRIX ASSEMBLY (NODAL APPROACH)")
        println("="^70)

        mass_assembly_pseudo = """
        # Similar to stiffness, but uses density ρ and N^T N
        for node_i in nodes
            for elem in node_to_elements[node_i]
                for node_j in elem.nodes
                    # Mass matrix block (3×3 for displacement)
                    M_ij = ∫_Ω ρ N_i N_j dΩ
                    
                    # For displacement (3D):
                    # M_ij = m_ij * I_3×3  (often lumped)
                    
                    # Consistent mass (full integration)
                    M_block = compute_mass_block(elem, node_i, node_j, ρ)
                    
                    # Or lumped mass (diagonal only)
                    if lumped
                        M_block = (i == j) ? diag(M_block) : 0
                    end
                    
                    M_nodal[node_i, node_j] += M_block
                end
            end
        end
        """

        println(mass_assembly_pseudo)
        println("="^70)
        println("✓ Mass matrix M = ∫_Ω ρ N^T N dΩ")
        println("✓ Consistent mass: Full integration (more accurate)")
        println("✓ Lumped mass: Diagonal only (faster, explicit dynamics)")
        println("✓ Nodal assembly works same as stiffness!")
        println("="^70)
    end

    # =============================================================================
    # KEY ARCHITECTURAL INSIGHTS
    # =============================================================================

    println("\n" * "="^70)
    println("MODAL ANALYSIS ARCHITECTURE INSIGHTS (NEW API)")
    println("="^70)
    println("✓ Modal analysis is EigenvalueProblem (K φ = λ M φ)")
    println("✓ Requires both stiffness K AND mass M matrices")
    println("✓ ModalSolver uses sparse eigenvalue methods (ARPACK)")
    println("✓ Geometric stiffness K_g affects natural frequencies")
    println("✓ Works for ANY physics (displacement, temperature, etc.)")
    println("✓ Mode shapes φ orthogonal w.r.t. mass matrix")
    println("✓ Natural frequencies ω = √λ, in Hz: f = ω/(2π)")
    println("✓ Same nodal assembly pattern for M as for K!")
    println("="^70)

end

"""
# IMPLEMENTATION NOTES

## Generalized Eigenvalue Problem

**Mathematical formulation:**

K φ = λ M φ

where:
- K = stiffness matrix (from elastic/thermal energy)
- M = mass matrix (from kinetic/thermal energy)
- φ = eigenvector (mode shape)
- λ = eigenvalue (ω² for vibrations, α for heat)

**For structural vibrations:**
- ω = √λ (natural frequency in rad/s)
- f = ω/(2π) (natural frequency in Hz)
- φ = displacement mode shape

**For heat diffusion:**
- α = λ (thermal diffusivity eigenvalue)
- φ = temperature mode shape

## Stiffness Matrix

**Elasticity:**

K = ∫_Ω B^T C B dΩ

where:
- B = strain-displacement matrix
- C = elasticity tensor

**Heat transfer:**

K = ∫_Ω k ∇N^T ∇N dΩ

where:
- k = thermal conductivity

## Mass Matrix

**Elasticity (consistent):**

M = ∫_Ω ρ N^T N dΩ

where ρ = density.

**Elasticity (lumped):**

M_ii = ∑_elements ∫_Ω_e ρ N_i dΩ_e

(Diagonal only, faster for explicit dynamics)

**Heat transfer:**

M = ∫_Ω ρc N^T N dΩ

where:
- ρ = density
- c = specific heat

## Geometric Stiffness

**Definition:** Stiffness contribution from initial stress state.

K_g = ∫_Ω G^T σ₀ G dΩ

where:
- G = geometric matrix (relates δε to ∇(δu))
- σ₀ = initial stress

**Effect:** Prestress changes natural frequencies:
- Tension → higher frequencies (stiffening)
- Compression → lower frequencies (softening)

**Total effective stiffness:**

K_eff = K + K_g

Then solve: K_eff φ = λ M φ

## Sparse Eigenvalue Solvers

**Problem:** For large systems, computing ALL eigenvalues is expensive.

**Solution:** Sparse eigenvalue methods (compute only k << n eigenvalues).

**Methods:**
1. **ARPACK** - Arnoldi iteration (standard in Julia)
2. **KrylovKit.jl** - Modern Krylov methods
3. **Lanczos** - For symmetric problems (K and M symmetric)

**JuliaFEM approach:**

```julia
using Arpack

# Solve K φ = λ M φ for k smallest eigenvalues
λ, φ = eigs(K, M, nev=k, which=:SM)
```

## Mode Shape Normalization

**Goal:** Normalize eigenvectors for convenience.

**Mass normalization (standard):**

φ_i^T M φ_i = 1

**Advantages:**
- Orthogonality: φ_i^T M φ_j = δ_ij
- Energy interpretation clear
- Modal damping ratios well-defined

**Implementation:**

```julia
function normalize_modes!(φ, M)
    for i in 1:length(φ)
        # Compute φ_i^T M φ_i
        norm_M = φ[i]' * M * φ[i]
        
        # Normalize
        φ[i] ./= sqrt(norm_M)
    end
end
```

## Nodal Assembly for Mass Matrix

```julia
function assemble_mass_nodal!(M_nodal, ρ, elements, node_to_elements)
    Threads.@threads for node_i in 1:n_nodes
        for elem_idx in node_to_elements[node_i]
            elem = elements[elem_idx]
            
            for node_j in elem.nodes
                # Mass matrix block (3×3 for displacement)
                M_ij = compute_mass_block(elem, node_i, node_j, ρ)
                
                # Add to global (no atomic on diagonal if i = j)
                if node_i == node_j
                    M_nodal[node_i, node_i] += M_ij  # Direct write
                else
                    atomic_add!(M_nodal[node_i, node_j], M_ij)
                end
            end
        end
    end
end
```

**Key:** SAME PATTERN as stiffness assembly!

## Next Steps

1. Implement `EigenvalueProblem` type
2. Implement `ModalSolver` with ARPACK integration
3. Implement mass matrix assembly (nodal)
4. Implement geometric stiffness computation
5. Add mode shape normalization
6. Validate against analytical solutions
7. Performance benchmarks (large systems)

"""
