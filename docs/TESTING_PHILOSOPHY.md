# JuliaFEM Testing Philosophy

**Date:** November 9, 2025  
**Goal:** 99% code coverage with educational, fast, well-structured tests  
**Tool:** Literate.jl for test-as-documentation

---

## Core Principles

### 1. Tests as Teaching Material

**Every test should teach something.**

Tests are not just validation - they're the **primary way users learn JuliaFEM**. When someone asks "How do I solve an elasticity problem?", the answer should be: "Look at `test/tutorials/elasticity_basics.jl`"

**Benefits:**

- Users learn by example (better than API docs)
- Tests stay current (if API changes, tests must update)
- Documentation never lies (it's tested code!)
- Newcomers can contribute tests (learning exercise)

### 2. Literate.jl for Test-Driven Documentation

Use Literate.jl to write tests as narrative documents:

```julia
# # Solving Your First Elasticity Problem
#
# This tutorial shows how to solve a simple 2D elasticity problem.
# We'll create a square block, apply boundary conditions, and solve.

using JuliaFEM
using Test

# ## Step 1: Create the Geometry
#
# First, define the nodes of a unit square:

X = Dict(
    1 => [0.0, 0.0],
    2 => [1.0, 0.0],
    3 => [1.0, 1.0],
    4 => [0.0, 1.0]
)

# ## Step 2: Create Elements
#
# Create a single Quad4 element:

element = Element(Quad4, (1, 2, 3, 4))
update!(element, "geometry", X)

# ... and so on
```

**Output:** Same file generates both test (runs in CI) and documentation (builds HTML).

### 3. Structured Test Hierarchy

Organize tests to match learning progression:

```text
test/
├── tutorials/          # Literate.jl files (test + docs)
│   ├── 01_fundamentals/
│   │   ├── creating_elements.jl
│   │   ├── fields_and_updates.jl
│   │   ├── reading_meshes.jl
│   │   └── basis_functions.jl
│   ├── 02_linear_problems/
│   │   ├── elasticity_1d.jl
│   │   ├── elasticity_2d.jl
│   │   ├── heat_transfer.jl
│   │   └── boundary_conditions.jl
│   ├── 03_nonlinear_problems/
│   │   ├── large_deformation.jl
│   │   ├── plasticity.jl
│   │   └── contact_basics.jl
│   ├── 04_advanced/
│   │   ├── contact_2d.jl
│   │   ├── contact_3d.jl
│   │   ├── mortar_methods.jl
│   │   └── friction.jl
│   └── 05_parallel/
│       ├── threading.jl
│       ├── gpu_assembly.jl
│       └── distributed.jl
├── unit/               # Fast unit tests (not Literate)
│   ├── basis/
│   ├── assembly/
│   └── solvers/
├── verification/       # Known analytical solutions
│   ├── timoshenko_beam.jl
│   ├── hertz_contact.jl
│   └── cook_membrane.jl
└── runtests.jl        # Test runner
```

### 4. Fast Tests First

**Test pyramid:**

```text
        /\
       /  \  Integration tests (slow, few)
      /____\
     /      \  Tutorial tests (medium, some)
    /________\
   /          \  Unit tests (fast, many)
  /__________\
```

**Timing targets:**

- Unit tests: < 5 minutes (run during development)
- Tutorial tests: < 15 minutes (run before commits)
- Full suite: < 30 minutes (run in CI)

**Strategy:**

- Use realistic meshes in tutorials (~10 elements: not too trivial, not too slow)
- Test correctness with analytical solutions, not big problems
- Profile and optimize slow tests
- Mark slow tests with `@testset "slow: hertz_contact"` (skip during dev)
- Co-locate mesh files with tests (no shared `/test/meshes/` directory)
- Include mesh generation recipes (show how mesh was created for reproducibility)

### 5. Coverage-Driven Development

**Target: 99% code coverage**

Every function should have:

1. **Happy path test** - normal usage
2. **Edge case tests** - empty input, single element, etc.
3. **Error tests** - what happens with bad input?

**Process:**

1. Write tutorial (covers main API)
2. Check coverage report
3. Add unit tests for uncovered lines
4. Repeat until 99%+

**Tools:**

- Coverage.jl (built into Julia)
- LocalCoverage.jl (for local checks)
- Codecov (in CI, we set this up yesterday)

---

## Test Structure Details

### Tutorial Tests (Literate.jl)

**Template:**

```julia
# # Tutorial Title
#
# Brief description of what this tutorial teaches.
# Prerequisites: what the reader should know first.

using JuliaFEM
using Test

# ## Section 1: Concept Explanation
#
# Explain the concept in prose, with equations if needed:
# 
# The strain-displacement relationship is:
# ```math
# ε = \frac{1}{2}(∇u + ∇u^T)
# ```

# Code demonstrating the concept
element = Element(Quad4, (1,2,3,4))

# ## Section 2: Building Up
#
# Step-by-step construction of a working example

# ... code ...

# ## Section 3: Validation
#
# Test that the result is correct (this is still a test!)

@testset "Elasticity 2D" begin
    @test isapprox(u_computed, u_analytical, rtol=1e-6)
end

# ## Discussion
#
# What did we learn? What can we do next?
# Links to related tutorials.
```

**Generation:**

```julia
using Literate
Literate.markdown("test/tutorials/01_fundamentals/creating_elements.jl", 
                  "docs/src/tutorials/")
Literate.notebook("test/tutorials/01_fundamentals/creating_elements.jl",
                  "docs/notebooks/")
```

### Unit Tests (Fast Validation)

**Purpose:** Test individual functions in isolation  
**Style:** Concise, no narrative  
**Location:** `test/unit/`

```julia
@testset "Basis Functions - Quad4" begin
    @testset "Evaluation at ξ=0, η=0" begin
        N = eval_basis(Quad4, (0.0, 0.0))
        @test N ≈ [0.25, 0.25, 0.25, 0.25]
    end
    
    @testset "Derivatives" begin
        dN = eval_dbasis(Quad4, (0.0, 0.0))
        @test size(dN) == (2, 4)
    end
    
    @testset "Edge case: extreme ξ" begin
        N = eval_basis(Quad4, (1.0, 1.0))
        @test N[3] ≈ 1.0
        @test sum(N) ≈ 1.0
    end
end
```

### Verification Tests (Known Solutions)

**Purpose:** Validate against analytical solutions or published results  
**Style:** Brief explanation + reference  
**Location:** `test/verification/`

```julia
# Timoshenko Beam - Verification Test
#
# Reference: Timoshenko & Goodier, "Theory of Elasticity", 3rd Ed.
# Problem: Cantilever beam with end load
# Analytical solution available for tip displacement

using JuliaFEM, Test

# Problem parameters from reference
L = 10.0  # Length
h = 1.0   # Height
E = 200e3 # Young's modulus
ν = 0.3   # Poisson's ratio
P = 100.0 # End load

# ... setup and solve ...

# Analytical solution
u_tip_analytical = P*L^3 / (3*E*I)

@testset "Timoshenko Beam Verification" begin
    @test isapprox(u_tip_computed, u_tip_analytical, rtol=0.01)
end
```

---

## Implementation Roadmap

### Phase 1: Infrastructure (Week 1) ✅ IN PROGRESS

**Goal:** Set up Literate.jl integration and test structure

**Tasks:**

1. ✅ Create `test/tutorials/` directory structure
2. ✅ Create new test runner (`test/runtests_new.jl`) with environment control
3. ✅ Add Gmsh.jl to test dependencies
4. ✅ Tutorial 1: Creating elements (5 tests passing)
5. ✅ Tutorial 2: Reading Gmsh meshes (72 tests passing, with recipe and co-located .msh)
6. ⏳ Tutorial 4: 1-element validation (priority for Issue #265)
7. ⏳ Tutorial 3: Basis functions
8. ⏳ Add Literate.jl to `docs/Project.toml`
9. ⏳ Update `docs/make.jl` to process tutorials
10. ⏳ Add coverage tools (LocalCoverage.jl)

**Status:** 77/77 tests passing (Tutorial 1: 5, Tutorial 2: 72)

**Deliverable:** Running CI that executes tutorials and reports coverage

### Phase 2: Core Tutorials (Week 2-3)

**Goal:** Write 10-15 fundamental tutorials covering main API

**Priority order:**

1. ✅ Creating elements and updating fields ⭐ (Done: 5 tests passing)
2. ✅ Reading meshes (Gmsh .msh format) ⭐ (Done: 72 tests passing)
3. **1-element validation tests** ⭐⭐ (Next: helps others validate FEM software, see Issue #265)
4. Basis functions and integration ⭐
5. Simple 2D elasticity (10 elements) ⭐
6. Boundary conditions (Dirichlet)
7. Surface loads and tractions
8. Heat transfer basics
9. Assembly process
10. Solving linear systems

**Note on mesh format:** We use **Gmsh** (via Gmsh.jl) instead of ABAQUS because:

- No license required (accessible to everyone)
- Julia native package (Gmsh.jl)
- Modern, actively developed
- Programmatic mesh generation (reproducible)
- Co-located mesh files with test files (self-contained tests)

**Success metric:** 60%+ code coverage from tutorials alone

### Phase 3: Advanced Tutorials (Week 4-5)

**Goal:** Cover advanced features

**Topics:**

1. Nonlinear elasticity (large deformation)
2. Contact mechanics basics (2D)
3. Mortar methods
4. 3D contact
5. Material models

**Success metric:** 80%+ coverage

### Phase 4: Unit Tests (Week 6)

**Goal:** Fill coverage gaps with fast unit tests

**Process:**

1. Generate coverage report: `julia --code-coverage=user test/runtests.jl`
2. Analyze: `using Coverage; LCOV.writefile("coverage.info", process_folder())`
3. Find uncovered lines
4. Write unit tests for each uncovered function
5. Repeat until 99%+

**Success metric:** 99% coverage, < 5 min unit test runtime

### Phase 5: Verification (Week 7)

**Goal:** Validate against known solutions

**Tests:**

1. Timoshenko beam (bending)
2. Hertz contact (2D)
3. Cook's membrane (stress concentration)
4. Patch tests (element validation)
5. Manufactured solutions

**Success metric:** All verification tests pass with < 1% error

### Phase 6: Documentation (Week 8)

**Goal:** Polish and publish

**Tasks:**

1. Review all tutorial narratives
2. Add figures and visualizations
3. Cross-link tutorials
4. Build documentation locally
5. Deploy to GitHub Pages
6. Write README.md guide to tutorials

**Success metric:** Beautiful, usable documentation website

---

## Test Runner Design

### `test/runtests.jl`

```julia
using Test, JuliaFEM

# Determine what to run based on environment
const RUN_UNIT = get(ENV, "JULIAFEM_TEST_UNIT", "true") == "true"
const RUN_TUTORIALS = get(ENV, "JULIAFEM_TEST_TUTORIALS", "true") == "true"
const RUN_SLOW = get(ENV, "JULIAFEM_TEST_SLOW", "false") == "true"
const RUN_VERIFICATION = get(ENV, "JULIAFEM_TEST_VERIFICATION", "true") == "true"

# Fast unit tests (always run)
if RUN_UNIT
    @testset "Unit Tests" begin
        include("unit/basis/test_quad4.jl")
        include("unit/basis/test_seg2.jl")
        # ... more unit tests
    end
end

# Tutorial tests (run in CI, optional locally)
if RUN_TUTORIALS
    @testset "Tutorials" begin
        # These are also documentation!
        include("tutorials/01_fundamentals/creating_elements.jl")
        include("tutorials/01_fundamentals/reading_meshes.jl")
        include("tutorials/02_linear_problems/elasticity_2d.jl")
        # ... more tutorials
    end
end

# Slow integration tests (CI only by default)
if RUN_SLOW
    @testset "Slow Tests" begin
        include("verification/hertz_contact.jl")
        # Large mesh tests
    end
end

# Verification tests
if RUN_VERIFICATION
    @testset "Verification" begin
        include("verification/timoshenko_beam.jl")
        include("verification/cook_membrane.jl")
    end
end
```

**Usage:**

```bash
# During development (fast, < 5 min)
julia --project=. -e 'using Pkg; Pkg.test()'

# Before commit (< 15 min)
JULIAFEM_TEST_TUTORIALS=true julia --project=. test/runtests.jl

# Full CI run (< 30 min)
JULIAFEM_TEST_SLOW=true julia --project=. test/runtests.jl

# Only unit tests (< 2 min)
JULIAFEM_TEST_TUTORIALS=false julia --project=. test/runtests.jl
```

---

## Coverage Workflow

### Local Development

```bash
# 1. Run tests with coverage
julia --project=. --code-coverage=user test/runtests.jl

# 2. Generate coverage report
julia --project=. -e '
using Coverage
coverage = process_folder()
covered = length(filter(c -> c.coverage > 0, coverage))
total = length(coverage)
println("Coverage: $(round(100*covered/total, digits=2))%")
LCOV.writefile("coverage.info", coverage)
'

# 3. View in browser (requires genhtml from lcov package)
genhtml coverage.info -o coverage/
firefox coverage/index.html
```

### GitHub Actions CI

Already set up yesterday! Codecov will:

- Track coverage over time
- Comment on PRs with coverage changes
- Show which lines are uncovered
- Badge in README.md

---

## Writing Style Guide

### For Tutorials (Literate.jl)

**Do:**

- ✅ Explain **why**, not just **what**
- ✅ Use realistic examples (~10 elements, not too simple or too complex)
- ✅ Include mathematical notation where helpful
- ✅ Show output/results
- ✅ Link to related tutorials
- ✅ Test the actual result (still a test!)
- ✅ Co-locate mesh files with test files (self-contained)
- ✅ Include mesh generation recipe (show how mesh was created)

**Don't:**

- ❌ Assume prior knowledge (explain or link)
- ❌ Use large meshes (slow tests)
- ❌ Skip explanation (this is documentation!)
- ❌ Test implementation details (test behavior)

### For Unit Tests

**Do:**

- ✅ Test one thing per `@testset`
- ✅ Use descriptive test names
- ✅ Test edge cases (empty, single, many)
- ✅ Test error conditions (`@test_throws`)
- ✅ Be concise (no prose needed)

**Don't:**

- ❌ Mix multiple concepts in one test
- ❌ Use real-world complex examples
- ❌ Duplicate tutorial content

---

## Migration Plan for Existing Tests

### Current State (56 test files)

Many old tests are:

- Poorly documented
- Using outdated API
- Slow (large meshes)
- Not structured for learning

### Migration Strategy

**Don't delete old tests immediately!** Instead:

1. **Categorize:** Is it tutorial material, unit test, or verification?
2. **Rewrite:** Create new version following philosophy
3. **Verify:** Ensure new test covers same functionality
4. **Archive:** Move old test to `test/archive/` with note
5. **Delete:** After new tests run successfully in CI

**Example:**

```text
test/test_elasticity_2d_linear_with_surface_load.jl  (old)
  → test/tutorials/02_linear_problems/elasticity_2d.jl  (new, Literate)
  → test/archive/test_elasticity_2d_linear_with_surface_load.jl.old  (keep for reference)
  → delete after 1 month if no issues
```

---

## Success Metrics

### Quantitative

- **Coverage:** 99%+ by end of Phase 4
- **Speed:** < 5 min unit tests, < 30 min full suite
- **Count:** 15+ tutorials, 100+ unit tests

### Qualitative

- **Can a new user learn JuliaFEM from tutorials alone?** (Ask someone!)
- **Are tutorials referenced in issue discussions?** ("See tutorial X")
- **Do contributors write tests first?** (TDD culture)
- **Is documentation always up-to-date?** (Literate.jl ensures it)

---

## Special: 1-Element Validation Tests

### Motivation (Issue #265)

In 2019, JuliaFEM was used to **validate another FEM software**. A user computed a reference solution with JuliaFEM and compared it against their own implementation. This is a powerful use case we should embrace!

### Design Philosophy

**Goal:** Create 1-element tests that can be:

1. **Hand-calculated** - Simple enough to verify by hand
2. **Exact** - Integer or simple fractional results (no floating-point ambiguity)
3. **Reference-quality** - Others can use to validate their FEM code
4. **Educational** - Show the math, not just code

### Example Pattern

```julia
# # 1-Element Validation: Quad4 Elasticity
#
# This tutorial computes the stiffness matrix for a single Quad4 element
# under plane stress conditions. The solution can be verified by hand.
#
# **Use case:** Validating FEM implementations (see Issue #265)
#
# ## Problem Setup
#
# Geometry: Unit square [0,1] × [0,1]
# Material: E = 100.0, ν = 0.3 (simple values)
# Element: Single Quad4 with nodes at corners
#
# ## Hand Calculation
#
# For Quad4 under plane stress, the stiffness matrix has structure:
# ... detailed derivation ...
#
# Expected K[1,1] = ... (show calculation)

using JuliaFEM, Test

# Define nodes (unit square)
nodes = Dict(
    1 => [0.0, 0.0],
    2 => [1.0, 0.0],
    3 => [1.0, 1.0],
    4 => [0.0, 1.0]
)

# Create element
element = Element(Quad4, (1, 2, 3, 4))
update!(element, "geometry", nodes)
update!(element, "youngs modulus", 100.0)
update!(element, "poissons ratio", 0.3)

# Assemble stiffness matrix
K = assemble_stiffness_matrix(element)

# Validate against hand calculation
@testset "1-Element Validation: Quad4 Stiffness" begin
    # Check specific entries that we calculated by hand
    @test K[1,1] ≈ 42.5  # Hand calculated value
    @test K[1,2] ≈ 10.0  # Hand calculated value
    # ... more validations
    
    # Symmetry check
    @test issymmetric(K)
    
    # Positive definite check (all eigenvalues > 0, after BC)
    # ... (need to apply constraints first)
end
```

### Benefits

1. **Validation tool** - Others can use JuliaFEM as reference
2. **Debugging aid** - If basic case fails, know where to look
3. **Educational** - Shows the math behind FEM
4. **Confidence** - Proves implementation is correct
5. **Regression test** - Any refactoring must pass these

### Implementation Priority

**High priority** - These tests are foundational. Should be in Tutorial 4 (before basis functions).

**Coverage:** Create 1-element tests for each element type:

- Seg2 (1D bar)
- Tri3 (2D triangle)
- Quad4 (2D quadrilateral)
- Tet4 (3D tetrahedron)
- Hex8 (3D hexahedron)

---

## Questions to Resolve

### 1. Literate.jl Execution Strategy

**Option A:** Tutorials are in `test/tutorials/`, executed by `Pkg.test()`

- Pro: Ensures tutorials always work
- Con: Slows down test suite

**Option B:** Tutorials are in `docs/tutorials/`, executed by docs build

- Pro: Fast test suite
- Con: Tutorials might break without noticing

**Recommendation:** Option A (tutorials in test/), with `RUN_TUTORIALS` env var

### 2. Visualization in Tutorials

Should tutorials include plots/visualizations?

**Option A:** Yes, using Makie.jl or similar

- Pro: Very educational, shows results
- Con: Heavy dependency, slow to precompile

**Option B:** No plots in tests, but show code in docs

- Pro: Fast tests
- Con: Less visual learning

**Recommendation:** Option B initially, add plots later as optional

### 3. Mesh Files ✅ DECIDED

**Decision:** Use Gmsh.jl for mesh generation (Option B)

**Rationale:**

- No license required (ABAQUS needs license, not accessible)
- No heavy dependencies (HDF5/MED format requires HDF5.jl)
- Programmatic generation (reproducible, can show recipe)
- Modern and well-maintained
- Julia native integration

**Implementation Pattern:**

1. Create `*_recipe.jl` script showing how mesh was generated
2. Run recipe to generate `*.msh` file
3. Commit both recipe and .msh file to repository
4. Test reads the .msh file (fast, reliable)
5. Users can see recipe to understand mesh structure

**Example:**

```text
test/tutorials/01_fundamentals/
├── reading_gmsh_meshes.jl          # The actual test
├── reading_gmsh_meshes.msh         # Pre-generated mesh (committed)
└── reading_gmsh_meshes_recipe.jl   # Shows how mesh was created
```

**Benefits:**

- Self-contained tests (mesh next to test file)
- Reproducible (recipe shows exactly how to recreate)
- Fast (don't generate mesh in CI, just read it)
- Educational (recipe teaches mesh generation)
- Accessible (no external software needed to run tests)

---

## Next Steps (This Week)

### Immediate (November 9, 2025)

1. ✅ Create this document (done!)
2. ✅ Create tutorial directory structure
3. ✅ Write first tutorial (creating elements - 5 tests passing)
4. ✅ Write second tutorial (reading Gmsh meshes - 72 tests passing)
5. ✅ Create new test runner (`test/runtests_new.jl`)
6. 🔄 **Now: Tutorial 4 - 1-element validation** (priority for Issue #265)
7. ⏳ Tutorial 3 - Basis functions
8. ⏳ Add Literate.jl to docs dependencies
9. ⏳ Update `docs/make.jl` for HTML generation

### This Week (Week 1 - November 9-15, 2025)

**Completed:**

1. ✅ Tutorial 1: Creating elements and fields (5 tests)
2. ✅ Tutorial 2: Reading Gmsh meshes (72 tests)

**In Progress:**
3. 🔄 Tutorial 4: 1-element validation (priority - helps validate other FEM software)
4. 🔄 Tutorial 3: Basis functions and integration

**Planned:**
5. Tutorial 5: Simple 2D elasticity (10 elements, use Tutorial 2 mesh)
6. Set up coverage workflow locally
7. Test Literate.jl → HTML generation
8. Update CI to run new test structure

**Target:** 5 tutorials by end of week (currently 2/5)

### Next Week

Continue writing tutorials, aim for 10 total by end of week.

---

## Conclusion

**This is a complete overhaul of testing strategy** - from "fix 49 failing tests" to "build educational test suite that reaches 99% coverage."

**Timeline:** ~8 weeks for full implementation  
**Effort:** Significant, but creates lasting value  
**Benefit:** Tests become documentation, documentation is always tested

**Philosophy:** Tests are not a chore - they're the best way to teach users how to use JuliaFEM.

---

**Status:** 📋 Planning complete, ready to implement  
**Next:** Get your feedback, then start Phase 1
