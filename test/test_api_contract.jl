# SPDX-FileCopyrightText: 2015-2026 Jukka Aho
# SPDX-License-Identifier: MIT

"""
Cross-cutting API contract tests for LLM-heavy workflows.

1. Every `export` name listed in `src/exports.jl` resolves in `JuliaFEM`
   after `using JuliaFEM` (catches dropped bindings / typos in exports).
2. Every directory listed in `TOPICS` in `test/runtests.jl` contains a
   `runtests.jl` entry point (catches forgotten wiring when adding topics).

Call `run_api_contract_tests(TOPICS, test_root_dir)` from `test/runtests.jl`
inside the main `@testset` so results nest correctly (`include` alone would
evaluate child testsets at module scope).
"""

using Test

"""
Parse `src/exports.jl` export statements, including comma-continued blocks.

Each comma-separated token is validated with `Base.Meta.parse`, so Unicode
identifiers (for example `owned_norm²_packed`) and macro exports (`@DOFSet`)
are accepted. Returns a `Vector{Symbol}` of unique exported names (order
preserved).
"""
function _parse_exported_symbols(exports_path::String)
    lines = readlines(exports_path)
    names = Symbol[]
    seen = Set{Symbol}()
    i = 1
    while i ≤ length(lines)
        raw = lines[i]
        stripped = lstrip(raw)
        if !startswith(stripped, "export ")
            i += 1
            continue
        end
        rest = stripped[begin+length("export "):end]
        hash = findfirst('#', rest)
        if hash !== nothing
            rest = rest[1:prevind(rest, hash)]
        end
        buf = IOBuffer()
        print(buf, rest)
        i += 1
        while i ≤ length(lines)
            cont = lines[i]
            t = lstrip(cont)
            isempty(t) && break
            startswith(t, "export ") && break
            startswith(t, '#') && break
            print(buf, ' ')
            h2 = findfirst('#', cont)
            chunk = h2 === nothing ? cont : cont[1:prevind(cont, h2)]
            print(buf, strip(chunk))
            i += 1
        end
        blob = String(take!(buf))
        for part in split(blob, ',')
            tok = strip(part)
            isempty(tok) && continue
            ex = Base.Meta.parse(tok; raise = false)
            sym = if ex isa Symbol
                ex::Symbol
            elseif ex isa Expr && ex.head === :macrocall && length(ex.args) ≥ 1 && ex.args[1] isa Symbol
                ex.args[1]::Symbol
            else
                error("Unexpected export token in exports.jl: $(repr(tok))")
            end
            if !(sym in seen)
                push!(names, sym)
                push!(seen, sym)
            end
        end
    end
    return names
end

"""
Run nested `@testset`s under the caller's current testset context.
"""
function run_api_contract_tests(topics, test_root_dir::String)
    @testset "TOPICS directories have runtests.jl" begin
        for topic in topics
            p = joinpath(test_root_dir, topic, "runtests.jl")
            @test isfile(p)
        end
        @test length(topics) == length(unique(topics))
    end

    @testset "exports.jl names are defined on JuliaFEM" begin
        exports_path = joinpath(test_root_dir, "..", "src", "exports.jl")
        @test isfile(exports_path)
        syms = _parse_exported_symbols(exports_path)
        @test length(syms) ≥ 200
        missing = [s for s in syms if !isdefined(JuliaFEM, s)]
        if !isempty(missing)
            @error "exports.jl lists symbols not defined on JuliaFEM" missing = missing
        end
        @test isempty(missing)
    end
    return nothing
end
