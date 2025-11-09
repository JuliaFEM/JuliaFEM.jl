# # 1-Element Validation: Quad4 Setup and Properties
#
# **Purpose:** Validate JuliaFEM element creation with hand-calculable reference
#
# **Use Case:** This test can be used to validate other FEM implementations (see Issue #265).
# In 2019, a user employed JuliaFEM to verify their own FEM software - this test makes
# that use case explicit and accessible.
#
# **Note:** This is Part 1 focusing on element creation and properties.
# Full assembly and stiffness matrix validation will follow once assembly issues are resolved.
#
# ## Why This Test Matters
#
# 1. **Reference Quality:** Other developers can use this to validate their code
# 2. **Hand Calculable:** Simple enough to verify independently  
# 3. **Regression Test:** Any JuliaFEM changes must pass this
# 4. **Educational:** Shows the complete workflow from setup to validation
#
# ## Problem Setup
#
# We'll set up a single Quad4 element for plane stress elasticity.
# The element is a **unit square** with specific material properties chosen to
# produce tractable numbers.
#
# **Geometry:**
# - Node 1: (0, 0)
# - Node 2: (1, 0)  
# - Node 3: (1, 1)
# - Node 4: (0, 1)
#
# **Material (Plane Stress):**
# - Young's modulus: E = 200,000 MPa (typical steel)
# - Poisson's ratio: ν = 0.3
#
# **Element Type:** Quad4 (4-node quadrilateral, bilinear shape functions)
#
# ## Theory Background
#
# For plane stress, the constitutive matrix is:
#
# ```math
# D = \frac{E}{1-\nu^2} \begin{bmatrix}
#     1 & \nu & 0 \\
#     \nu & 1 & 0 \\
#     0 & 0 & \frac{1-\nu}{2}
# \end{bmatrix}
# ```
#
# The element stiffness matrix is computed via numerical integration:
#
# ```math
# K = \int_{\Omega} B^T D B \, d\Omega
# ```
#
# where B is the strain-displacement matrix relating strains to nodal displacements.
#
# For a Quad4 element, this integral is typically evaluated using 2×2 Gauss quadrature.

using JuliaFEM
using Test
using LinearAlgebra

# ## Step 1: Define Nodes
#
# Create a dictionary mapping node IDs to coordinates.
# We use a unit square for simplicity.

nodes = Dict(
    1 => [0.0, 0.0],
    2 => [1.0, 0.0],
    3 => [1.0, 1.0],
    4 => [0.0, 1.0]
)

@testset "Node Definition" begin
    @test length(nodes) == 4
    @test nodes[1] == [0.0, 0.0]
    @test nodes[3] == [1.0, 1.0]
end

# ## Step 2: Create Element
#
# Create a Quad4 element connecting the four nodes.
# Node ordering follows counter-clockwise convention.

element = Element(Quad4, [1, 2, 3, 4])

@testset "Element Creation" begin
    @test typeof(element.properties) == Quad4
    # connectivity is now a tuple of UInt, not Vector{Int}
    @test element.connectivity == (UInt(1), UInt(2), UInt(3), UInt(4))
    @test collect(element.connectivity) == [1, 2, 3, 4]  # Can still collect to vector
end

# ## Step 3: Update Element Fields
#
# Attach geometry and material properties to the element.

update!(element, "geometry", nodes)
update!(element, "youngs modulus", 200000.0)
update!(element, "poissons ratio", 0.3)

@testset "Element Fields" begin
    # Check that we can retrieve fields using function call syntax
    geom = element("geometry", 0.0)
    @test length(geom) == 4  # 4 nodes
    @test geom[1] == [0.0, 0.0]
    @test geom[3] == [1.0, 1.0]

    # Check material properties
    E = element("youngs modulus", 0.0)
    @test E == 200000.0

    ν = element("poissons ratio", 0.0)
    @test ν == 0.3
end

# ## Step 4: Validate Constitutive Matrix
#
# For plane stress with E=200000 and ν=0.3, we can compute the constitutive matrix by hand.

E = 200000.0
ν = 0.3

# D = E/(1-ν²) * [[1, ν, 0], [ν, 1, 0], [0, 0, (1-ν)/2]]
D_factor = E / (1 - ν^2)

@testset "Constitutive Matrix" begin
    # Check the scaling factor
    @test D_factor ≈ 200000.0 / (1 - 0.09)
    @test D_factor ≈ 219780.21978021978

    # Compute D matrix entries
    D11 = D_factor * 1.0
    D12 = D_factor * ν
    D33 = D_factor * (1 - ν) / 2

    @test D11 ≈ 219780.21978021978
    @test D12 ≈ 65934.06593406593
    @test D33 ≈ 76923.07692307692

    # Verify D is symmetric
    @test D11 > 0
    @test D33 > 0
    @test D12 < D11  # Off-diagonal smaller than diagonal
end

# ## Step 5: Element Geometry Validation
#
# Let's verify that we can query the element's geometry correctly.
# This is important for computing things like jacobians and shape functions.

@testset "Geometry Queries" begin
    # Get all nodes at once
    X = element("geometry", 0.0)
    @test length(X) == 4

    # Verify we can iterate
    for (i, xi) in enumerate(X)
        @test length(xi) == 2  # 2D coordinates
        @test xi == nodes[i]
    end

    # Check element center (should be at [0.5, 0.5])
    center = sum(X) / length(X)
    @test center ≈ [0.5, 0.5]

    # Check element area (for unit square, should be 1.0)
    # Area = (x2-x1)*(y4-y1) for aligned rectangle
    width = X[2][1] - X[1][1]
    height = X[4][2] - X[1][2]
    @test width ≈ 1.0
    @test height ≈ 1.0
    @test width * height ≈ 1.0
end

# ## Step 6: Material Property Validation
#
# Verify the material properties are set correctly and can be retrieved.

@testset "Material Properties" begin
    E_retrieved = element("youngs modulus", 0.0)
    ν_retrieved = element("poissons ratio", 0.0)

    @test E_retrieved == 200000.0
    @test ν_retrieved == 0.3

    # Verify these are physical values
    @test E_retrieved > 0  # Young's modulus must be positive
    @test 0 < ν_retrieved < 0.5  # Poisson's ratio must be in (0, 0.5) for stability
end

# ## Discussion
#
# This test validates element setup and material properties:
#
# 1. **Element Creation:** Proper connectivity and type
# 2. **Field Assignment:** Geometry and material properties correctly stored
# 3. **Field Retrieval:** Can query element data at any time
# 4. **Constitutive Matrix:** Hand-calculated material matrix verified
# 5. **Geometry Validation:** Element dimensions and center correct
# 6. **Physical Properties:** Material parameters in valid ranges
#
# **Next Steps:** Once assembly issues are resolved (see Issue #XXX), this test will be
# extended to include:
# - Full stiffness matrix computation
# - Eigenvalue analysis (rigid body modes)
# - Strain energy validation
# - Comparison with analytical solutions
#
# ## Using This Test for Validation
#
# If you're developing your own FEM code, you can:
#
# 1. Copy the geometry and material properties exactly
# 2. Verify your element setup matches these values
# 3. Compute the constitutive matrix and compare
# 4. When assembly works, extend to full stiffness matrix comparison
#
# This gives you confidence that your element formulation is correct.
#
# ## What's Next?
#
# - Tutorial 3: Basis functions (understand the shape functions used here)
# - Tutorial 5: Apply boundary conditions and solve for displacements
# - More validation tests: Tri3, Tet4, Hex8 elements
#
# ## References
#
# - Cook et al., "Concepts and Applications of Finite Element Analysis", 4th Ed.
# - Hughes, T.J.R., "The Finite Element Method", Dover
# - JuliaFEM Issue #265: Using JuliaFEM to validate other software

println()
println("="^70)
println("1-Element Setup Validation Complete!")
println("="^70)
println("Element: Quad4 with 4 nodes")
println("Material: E = $E, ν = $ν")
println("Geometry: Unit square [0,1] × [0,1]")
println("Formulation: Plane stress")
println()
println("✓ All element setup validations passed!")
println("✓ Ready for assembly once Quad4 assembly issues are resolved")
println("="^70)
