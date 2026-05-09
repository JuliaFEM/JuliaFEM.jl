#!/usr/bin/env julia
# SPDX-FileCopyrightText: 2015-2026 Jukka Aho
# SPDX-License-Identifier: MIT
#
# Run the JuliaFEM test suite under `--code-coverage=user`, summarise the
# resulting `.cov` files, and emit `lcov.info` for editor / CI tooling.
#
# Usage:
#   julia scripts/coverage.jl                  # full pass: tests + summary
#   julia scripts/coverage.jl --summary-only   # skip tests, summarise existing .cov
#   julia scripts/coverage.jl --threshold 95        # all src/ (incl. legacy files)
#   julia scripts/coverage.jl --threshold-live 95   # shipped runtime only (see summarise_live)
#   julia scripts/coverage.jl --legacy         # tests with JULIAFEM_ENABLE_LEGACY=1
#   julia scripts/coverage.jl --top 50         # show top-N least-covered files
#
# Side effects:
#   - Writes `.cov` files next to every source file under `src/`.
#   - Writes `coverage/lcov.info` for editor coverage gutters / Codecov.
#   - Prints a per-file table sorted by uncovered LOC.
#
# Coverage tooling (Coverage.jl) lives in the dedicated `scripts/coverage/`
# environment so the package's runtime and test deps stay clean.

using Pkg
using Printf
using Logging

const ROOT = normpath(joinpath(@__DIR__, ".."))
const COV_ENV = joinpath(@__DIR__, "coverage")
const SRC_DIR = joinpath(ROOT, "src")
const LCOV_DIR = joinpath(ROOT, "coverage")
const LCOV_PATH = joinpath(LCOV_DIR, "lcov.info")
const COVERAGE_UUID = Base.UUID("a2441757-f6aa-5fb2-8edb-039e3f45d037")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

mutable struct Args
    summary_only::Bool
    legacy::Bool
    threshold::Union{Nothing,Float64}
    threshold_live::Union{Nothing,Float64}
    top_n::Int
end

function parse_args(argv::Vector{String})
    a = Args(false, false, nothing, nothing, 25)
    i = 1
    while i ≤ length(argv)
        arg = argv[i]
        if arg == "--summary-only"
            a.summary_only = true
        elseif arg == "--legacy"
            a.legacy = true
        elseif arg == "--threshold"
            i += 1
            i ≤ length(argv) || error("--threshold expects a value")
            a.threshold = parse(Float64, argv[i])
        elseif startswith(arg, "--threshold=")
            a.threshold = parse(Float64, split(arg, "=", limit=2)[2])
        elseif arg == "--threshold-live"
            i += 1
            i ≤ length(argv) || error("--threshold-live expects a value")
            a.threshold_live = parse(Float64, argv[i])
        elseif startswith(arg, "--threshold-live=")
            a.threshold_live = parse(Float64, split(arg, "=", limit=2)[2])
        elseif arg == "--top"
            i += 1
            i ≤ length(argv) || error("--top expects a value")
            a.top_n = parse(Int, argv[i])
        elseif arg in ("-h", "--help")
            print(stderr, help_text())
            exit(0)
        else
            error("Unknown argument: $arg")
        end
        i += 1
    end
    return a
end

function help_text()
    out = IOBuffer()
    for line in eachline(@__FILE__)
        startswith(line, "#") || break
        println(out, line)
    end
    return String(take!(out))
end

# ---------------------------------------------------------------------------
# Test invocation
# ---------------------------------------------------------------------------

function clean_existing_coverage!(src_dir::AbstractString)
    n = 0
    for (root, _, files) in walkdir(src_dir)
        for f in files
            if endswith(f, ".cov")
                rm(joinpath(root, f); force=true)
                n += 1
            end
        end
    end
    return n
end

function run_tests_with_coverage(; legacy::Bool)
    @info "Running test suite with coverage instrumentation" legacy
    julia = Base.julia_cmd()
    env = copy(ENV)
    if legacy
        env["JULIAFEM_ENABLE_LEGACY"] = "1"
    else
        delete!(env, "JULIAFEM_ENABLE_LEGACY")
    end
    cmd = setenv(
        `$(julia) --project=$(ROOT) -e "using Pkg; Pkg.test(\"JuliaFEM\"; coverage=true)"`,
        env,
    )
    return success(run(cmd; wait=true))
end

# ---------------------------------------------------------------------------
# Coverage gathering (loaded from the side env)
# ---------------------------------------------------------------------------

function load_coverage_pkg()
    Pkg.activate(COV_ENV)
    Pkg.instantiate()
    return Base.require(Base.PkgId(COVERAGE_UUID, "Coverage"))
end

function summarise(Coverage; src_dir::AbstractString)
    coverage = with_logger(Logging.NullLogger()) do
        Base.invokelatest(Coverage.process_folder, src_dir)
    end
    coverage = Base.invokelatest(Coverage.merge_coverage_counts, coverage)

    rows = Vector{NamedTuple{(:file, :covered, :lines, :pct, :uncovered),
                             Tuple{String,Int,Int,Float64,Int}}}()
    total_covered = 0
    total_lines = 0

    for fc in coverage
        rel = relpath(fc.filename, ROOT)
        c, l = Base.invokelatest(Coverage.get_summary, fc)
        total_covered += c
        total_lines += l
        pct = l == 0 ? 100.0 : 100 * c / l
        push!(rows, (file=rel, covered=c, lines=l, pct=pct, uncovered=(l - c)))
    end

    return rows, total_covered, total_lines
end

function print_table(rows; top_n::Int)
    sort!(rows; by=r -> (r.uncovered, -r.lines), rev=true)

    n_show = min(top_n, length(rows))
    println()
    println("Files with the most uncovered lines (top $n_show of $(length(rows))):")
    println()
    @printf "  %-60s %8s %8s %8s\n" "file" "covered" "total" "pct"
    @printf "  %-60s %8s %8s %8s\n" repeat("-", 60) repeat("-", 8) repeat("-", 8) repeat("-", 8)
    for r in Iterators.take(rows, n_show)
        @printf "  %-60s %8d %8d %7.2f%%\n" trunc_left(r.file, 60) r.covered r.lines r.pct
    end
    println()
end

# Group rows by top-level src/ subdirectory and summarise.
function print_subdir_breakdown(rows)
    groups = Dict{String,Tuple{Int,Int,Int}}()  # subdir => (covered, lines, files)
    for r in rows
        # `r.file` looks like "src/<subdir>/...". Bucket by <subdir>.
        parts = split(r.file, '/')
        subdir = length(parts) ≥ 2 ? parts[2] : "<root>"
        c, l, n = get(groups, subdir, (0, 0, 0))
        groups[subdir] = (c + r.covered, l + r.lines, n + 1)
    end
    ordered = sort(collect(groups); by=g -> g.second[2], rev=true)

    println("Coverage by src/ subdirectory:")
    println()
    @printf "  %-22s %6s %8s %8s %8s\n" "subdir" "files" "covered" "total" "pct"
    @printf "  %-22s %6s %8s %8s %8s\n" repeat("-", 22) repeat("-", 6) repeat("-", 8) repeat("-", 8) repeat("-", 8)
    for (subdir, (c, l, n)) in ordered
        pct = l == 0 ? 100.0 : 100 * c / l
        @printf "  %-22s %6d %8d %8d %7.2f%%\n" subdir n c l pct
    end
    println()
end

# Live coverage excludes:
# - `src/legacy/` (only loaded with `JULIAFEM_ENABLE_LEGACY=1`)
# - Generator-only sources under `src/basis/` that are never `include`d from
#   `JuliaFEM.jl` (they would otherwise show as 0% and distort the metric).
const LIVE_EXCLUDE_FILES = (
    "src/basis/basis_generator.jl",
    "src/basis/subs.jl",
    "src/basis/vandermonde.jl",
)

function summarise_live(rows)
    live = filter(rows) do r
        startswith(r.file, "src/legacy/") && return false
        r.file in LIVE_EXCLUDE_FILES && return false
        return true
    end
    covered = sum(r -> r.covered, live; init=0)
    lines = sum(r -> r.lines, live; init=0)
    return covered, lines, length(live)
end

function trunc_left(s::AbstractString, n::Int)
    length(s) ≤ n ? s : "…" * s[end - n + 2:end]
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

function main(argv::Vector{String})
    args = parse_args(argv)

    if !args.summary_only
        nclean = clean_existing_coverage!(SRC_DIR)
        nclean > 0 && @info "Removed $nclean stale .cov files"
        ok = run_tests_with_coverage(; legacy=args.legacy)
        ok || error("Test suite failed; see output above.")
    else
        @info "--summary-only: reusing existing .cov files under $(SRC_DIR)"
    end

    @info "Loading Coverage.jl from $(COV_ENV)"
    Coverage = load_coverage_pkg()

    @info "Gathering coverage from $(SRC_DIR)"
    rows, total_covered, total_lines = summarise(Coverage; src_dir=SRC_DIR)

    isdir(LCOV_DIR) || mkpath(LCOV_DIR)
    with_logger(Logging.NullLogger()) do
        cov = Base.invokelatest(Coverage.process_folder, SRC_DIR)
        Base.invokelatest(Coverage.LCOV.writefile, LCOV_PATH, cov)
    end
    @info "Wrote LCOV report" LCOV_PATH

    print_table(rows; top_n=args.top_n)
    print_subdir_breakdown(rows)

    pct = total_lines == 0 ? 100.0 : 100 * total_covered / total_lines
    n_files = length(rows)
    n_uncovered_files = count(r -> r.covered == 0 && r.lines > 0, rows)
    live_c, live_l, live_n = summarise_live(rows)
    live_pct = live_l == 0 ? 100.0 : 100 * live_c / live_l

    @printf "Files:                       %d total, %d with zero coverage\n" n_files n_uncovered_files
    @printf "Lines:                       %d / %d covered\n" total_covered total_lines
    @printf "Total coverage (all src/):   %.2f%%\n" pct
    @printf "Live coverage (excl legacy): %.2f%%  (%d / %d lines, %d files)\n" live_pct live_c live_l live_n
    println()

    if args.threshold !== nothing
        if pct < args.threshold
            @error "Coverage $(round(pct, digits=2))% is below threshold $(args.threshold)%"
            exit(2)
        else
            @info "Coverage $(round(pct, digits=2))% meets threshold $(args.threshold)%"
        end
    end

    if args.threshold_live !== nothing
        if live_pct < args.threshold_live
            @error "Live coverage $(round(live_pct, digits=2))% is below threshold $(args.threshold_live)%"
            exit(2)
        else
            @info "Live coverage $(round(live_pct, digits=2))% meets threshold $(args.threshold_live)%"
        end
    end
end

main(ARGS)
