# SPDX-FileCopyrightText: 2015-2026 Jukka Aho
# SPDX-License-Identifier: MIT

"""
Test-only package hygiene via Aqua.jl.

Ambiguities, unbound-args scanning, piracy detection, and persistent-task
audits are disabled here: they are either too noisy or too slow for a large
numerical codebase on every `Pkg.test`.

`stale_deps` ignores `CSV`: it is a direct dependency for optional I/O paths
but is not loaded by `using JuliaFEM` alone, which is what Aqua probes.

`deps_compat` skips compat requirements for standard-library packages listed
in `[deps]` and turns off the extras table: adding compat entries for every
stdlib and test-only extra is policy noise for this repository.
"""

using Aqua
using Test

function run_pkg_hygiene_tests(m::Module = JuliaFEM)
    Aqua.test_all(
        m;
        ambiguities = false,
        unbound_args = false,
        undefined_exports = true,
        project_extras = true,
        stale_deps = (; ignore = [:CSV]),
        deps_compat = (;
            ignore = [:LinearAlgebra, :Logging, :SparseArrays],
            check_extras = false,
        ),
        piracies = false,
        persistent_tasks = false,
        undocumented_names = false,
    )
    return nothing
end
