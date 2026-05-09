#!/usr/bin/env julia
# Fail if obsolete API strings appear in curated website Markdown.
# Run from repository root:
#   julia --project=. scripts/check_curated_doc_vocabulary.jl

const ROOT = dirname(@__DIR__)
const NEEDLES = ["register_fields!", "count_field_dofs"]
# User-facing guides only: developer pages (for example `dof-manager-guide.md`)
# may mention removed APIs when contrasting with `DOFHandler`.
const SCAN_DIRS = [joinpath(ROOT, "juliafem.github.io", "docs", "user-guide")]

for dir in SCAN_DIRS
    isdir(dir) || error("Missing directory: ", dir)
    for (root, _, files) in walkdir(dir)
        for fn in files
            endswith(fn, ".md") || continue
            path = joinpath(root, fn)
            text = read(path, String)
            for needle in NEEDLES
                if occursin(needle, text)
                    rel = relpath(path, ROOT)
                    error("Curated doc vocabulary: found $(repr(needle)) in ", rel)
                end
            end
        end
    end
end
println("check_curated_doc_vocabulary.jl: OK")
