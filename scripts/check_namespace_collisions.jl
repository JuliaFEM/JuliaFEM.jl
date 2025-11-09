#!/usr/bin/env julia
"""
Analyze namespace collisions before consolidation
Finds function definitions that might conflict when merging packages
"""

using Printf

function find_function_definitions(dir::String)
    """Find all function definitions in .jl files"""
    functions = Dict{String,Vector{String}}()

    for (root, dirs, files) in walkdir(dir)
        for file in files
            if !endswith(file, ".jl")
                continue
            end

            filepath = joinpath(root, file)
            relpath_str = relpath(filepath, dir)

            try
                content = read(filepath, String)
                lines = split(content, '\n')

                for (lineno, line) in enumerate(lines)
                    # Match function definitions
                    m = match(r"^\s*function\s+([a-zA-Z_][a-zA-Z0-9_!]*)", line)
                    if m !== nothing
                        fname = m.captures[1]
                        location = "$relpath_str:$lineno"

                        if !haskey(functions, fname)
                            functions[fname] = String[]
                        end
                        push!(functions[fname], location)
                    end

                    # Match short-form function definitions
                    m = match(r"^\s*([a-zA-Z_][a-zA-Z0-9_!]*)\([^)]*\)\s*=", line)
                    if m !== nothing && !occursin("function", line)
                        fname = m.captures[1]
                        location = "$relpath_str:$lineno"

                        if !haskey(functions, fname)
                            functions[fname] = String[]
                        end
                        push!(functions[fname], location)
                    end
                end
            catch e
                @warn "Error reading $filepath: $e"
            end
        end
    end

    return functions
end

function main()
    println("="^80)
    println("NAMESPACE COLLISION ANALYSIS - JuliaFEM Consolidation")
    println("="^80)
    println()

    # Analyze vendor packages
    vendor_dir = "/home/juajukka/dev/JuliaFEM.jl/vendor"
    src_dir = "/home/juajukka/dev/JuliaFEM.jl/src"

    println("Analyzing vendor packages...")
    vendor_functions = find_function_definitions(vendor_dir)

    println("Analyzing main src/...")
    src_functions = find_function_definitions(src_dir)

    # Find collisions (functions defined in multiple places)
    println()
    println("="^80)
    println("POTENTIAL COLLISIONS (functions defined multiple times)")
    println("="^80)
    println()

    collision_count = 0
    high_risk = String[]

    # Check vendor collisions
    for (fname, locations) in sort(collect(vendor_functions), by=x -> length(x[2]), rev=true)
        if length(locations) > 1
            collision_count += 1

            # Check if it's a common override pattern (likely safe)
            is_override = any(occursin(r"FEMBase\.|get_|assemble", fname) for loc in locations)

            risk = is_override ? "🟢 LOW" : "🔴 HIGH"

            if !is_override
                push!(high_risk, fname)
            end

            println("$risk: $fname ($(length(locations)) definitions)")
            for loc in locations[1:min(5, length(locations))]
                println("    $loc")
            end
            if length(locations) > 5
                println("    ... and $(length(locations) - 5) more")
            end
            println()
        end
    end

    println("="^80)
    println("SUMMARY")
    println("="^80)
    println()
    @printf "Total unique functions in vendor: %d\n" length(vendor_functions)
    @printf "Total unique functions in src:    %d\n" length(src_functions)
    @printf "Functions defined multiple times:  %d\n" collision_count
    @printf "High-risk collisions:              %d\n" length(high_risk)
    println()

    if length(high_risk) > 0
        println("High-risk collisions to review:")
        for fname in high_risk[1:min(10, length(high_risk))]
            println("  - $fname")
        end
        if length(high_risk) > 10
            println("  ... and $(length(high_risk) - 10) more")
        end
    else
        println("✅ No high-risk collisions detected!")
        println("   Most collisions are likely dispatch specializations (safe)")
    end

    println()
    println("="^80)
    println("RECOMMENDATION")
    println("="^80)
    println("""
    Most function "collisions" in FEM packages are actually safe:
    - Different Problem{T} types dispatch correctly
    - get_* and assemble_* are intentional overrides

    Review high-risk collisions manually before consolidation.
    """)
end

main()
