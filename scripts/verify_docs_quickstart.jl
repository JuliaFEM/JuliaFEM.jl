#!/usr/bin/env julia
# Verify the Documenter minimal elasticity snippet (repository root).
#   julia --project=. scripts/verify_docs_quickstart.jl

const ROOT = dirname(@__DIR__)

using Pkg
Pkg.activate(ROOT)
Pkg.instantiate()

include(joinpath(ROOT, "docs", "src", "snippets", "minimal_elasticity_quickstart.jl"))
r = minimal_elasticity_quickstart()
if r.ndofs != 375 || r.nnz_stiffness != 19773
    error("quickstart regression: expected (375, 19773), got ($(r.ndofs), $(r.nnz_stiffness))")
end
println("verify_docs_quickstart.jl: OK")
