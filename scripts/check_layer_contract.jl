#!/usr/bin/env julia
#
# Static audit for docs/src/developer/architecture_layers.md (layers A–C).
# Run from anywhere:
#   julia scripts/check_layer_contract.jl [REPO_ROOT]
#
# Uses only Base — no --project required.

const REPO_ROOT = isempty(ARGS) ? dirname(dirname(@__FILE__)) : abspath(ARGS[1])

const INCLUDE_IO_REGEX = r"include\s*\(\s*\"[^\"]*\bio/"

function jl_files(dir::AbstractString)
    list = String[]
    isdir(dir) || return list
    for (root, _, files) in walkdir(dir)
        for f in files
            endswith(f, ".jl") || continue
            push!(list, joinpath(root, f))
        end
    end
    sort!(list)
end

function code_fragment(line::AbstractString)::SubString{String}
    # Ignore trailing `# …` so notes do not false-positive the auditors.
    parts = split(line, '#'; limit = 2)
    return rstrip(parts[1])
end

function skip_line(line::AbstractString)::Bool
    s = strip(line)
    isempty(s) && return true
    startswith(s, '#') && return true
    return false
end

const DOMAIN_FORBIDDEN_SUBSTRINGS = [
    "GmshReader",
    "read_gmsh_mesh",
    "GmshMesh",
    "JuliaFEM.Legacy",
    "KernelAbstractions",
    "CUDABackend",
    "MetalBackend",
    "AMDGPUBackend",
    "oneAPIBackend",
]

const FOUNDATION_EXTRA_SUBSTRINGS = [
    "DOFHandler",
    "create_structured_box_mesh",
]

function violations_for_line(line::AbstractString, extra_substrings)::Vector{String}
    hits = String[]
    if occursin(INCLUDE_IO_REGEX, line)
        push!(hits, "include(\"…io/…\")")
    end
    for tok in DOMAIN_FORBIDDEN_SUBSTRINGS
        if occursin(tok, line)
            push!(hits, tok)
        end
    end
    for tok in extra_substrings
        if occursin(tok, line)
            push!(hits, tok)
        end
    end
    return hits
end

function audit_tree(label::AbstractString, dir::AbstractString, extra_substrings)::Bool
    ok = true
    rel = relpath(dir, REPO_ROOT)
    for path in jl_files(dir)
        lineno = 0
        for line in eachline(path)
            lineno += 1
            skip_line(line) && continue
            frag = code_fragment(line)
            isempty(strip(frag)) && continue
            hits = violations_for_line(frag, extra_substrings)
            isempty(hits) && continue
            ok = false
            relfile = relpath(path, REPO_ROOT)
            println(stderr, "$relfile:$lineno: [layer contract] $(join(sort!(unique(hits)), ", "))")
        end
    end
    ok || println(stderr, "($label) violations under $rel")
    return ok
end

function main()::Int
    src = joinpath(REPO_ROOT, "src")
    isdir(src) || error("expected src/ under $REPO_ROOT")

    println("Layer contract audit (see docs/src/developer/architecture_layers.md)")
    println("Repository: ", REPO_ROOT)

    all_ok = true

    domains = joinpath(src, "domains")
    all_ok &= audit_tree("Layer C (domains)", domains, String[])

    foundation_dirs = [
        joinpath(src, "topology"),
        joinpath(src, "quadrature"),
        joinpath(src, "geometry"),
        joinpath(src, "basis"),
        joinpath(src, "sparse"),
    ]
    for d in foundation_dirs
        all_ok &= audit_tree("Layer A (foundation)", d, FOUNDATION_EXTRA_SUBSTRINGS)
    end

    if all_ok
        println("OK — no forbidden coupling patterns found.")
        return 0
    else
        println(stderr, "")
        println(stderr, "FAILED — fix imports/includes or move code to an outer layer.")
        return 1
    end
end

exit(main())
