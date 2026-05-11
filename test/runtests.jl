# SPDX-FileCopyrightText: 2015-2026 Jukka Aho
# SPDX-License-Identifier: MIT

using JuliaFEM, Test

# Shared zero-allocation / LLVM IR helpers for topic tests (loaded once).
include(joinpath(@__DIR__, "zero_alloc_helpers.jl"))
include(joinpath(@__DIR__, "test_api_contract.jl"))
include(joinpath(@__DIR__, "test_pkg_hygiene.jl"))
include(joinpath(@__DIR__, "test_inference_smoke.jl"))

# The active test tree mirrors `src/`. Every subdirectory below has its
# own `runtests.jl` that bundles the topic into a single `@testset`.
# Adding a new topic = adding the directory to `topics` below.
const TOPICS = [
    "basis",
    "fields",
    "physics",
    "elements",
    "materials",
    "topology",
    "geometry",
    "mesh",
    "io",
    "quadrature",
    "domains/continuum",
    "domains/heat",
    "domains/darcy",
    "domains/thermo_elastic",
    "domains/poroelastic",
    "domains/thermo_poroelastic",
    "dofs",
    "interface",
    "assemblers",
    "sparse",
    "validation",
    "reference",
    "docs",
]

# If ARGS has no positional topic names, run every entry in `TOPICS`.
# Otherwise run only listed topics (in `TOPICS` order). Unknown names warn;
# if none match, run the full suite with a warning.
#
# `Pkg.test(; test_args=["assemblers", ...])` forwards those strings to ARGS.
# Only names from `TOPICS` below are accepted (e.g. `"assemblers"`), not file paths.
function _topics_from_args()
    requested = String[s for s in ARGS if !startswith(s, '-')]
    isempty(requested) && return TOPICS
    known = Set(TOPICS)
    unknown = filter(!in(known), requested)
    if !isempty(unknown)
        @warn "Unknown test topics (skipped): $(collect(unknown))"
    end
    want = Set(requested)
    chosen = filter(in(want), TOPICS)
    if isempty(chosen)
        @warn "No valid topics in ARGS $(requested); running full suite"
        return TOPICS
    end
    return chosen
end

@testset "JuliaFEM.jl" begin
    @testset "Package loads" begin
        @test isdefined(JuliaFEM, :AbstractBasis)
        @test isdefined(JuliaFEM, :AbstractMesh)
        @test isdefined(JuliaFEM, :AbstractInterfaceMesh)
        @test isdefined(JuliaFEM, :InterfaceMesh)
        @test isdefined(JuliaFEM, :AbstractTopology)
        @test isdefined(JuliaFEM, :AbstractMaterial)
        @test isdefined(JuliaFEM, :AbstractPhysics)
        @test isdefined(JuliaFEM, :AbstractField)
        @test isdefined(JuliaFEM, :AbstractFormulation)
    end

    @testset "Basic types instantiate" begin
        @test AbstractBasis isa Type
        @test AbstractMesh isa Type
        @test AbstractInterfaceMesh isa Type
        @test AbstractTopology isa Type
        @test AbstractMaterial isa Type
        @test AbstractPhysics isa Type
        @test AbstractField isa Type
        @test AbstractFormulation isa Type
    end

    @testset "Concrete types exist" begin
        @test Mesh isa Type
        @test LinearElastic isa Type
        @test Hex8 isa Type
        @test Tet4 isa Type
        @test LocalField isa Type
    end

    run_api_contract_tests(TOPICS, @__DIR__)

    run_pkg_hygiene_tests()
    run_inference_smoke_tests()

    for topic in _topics_from_args()
        include(joinpath(topic, "runtests.jl"))
    end
end

# Tests that target the older / Legacy API or experimental specs live under
# `llm/design/legacy-tests/` and are not picked up by this runner.
