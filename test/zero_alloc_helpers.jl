# SPDX-FileCopyrightText: 2015-2026 Jukka Aho
# SPDX-License-Identifier: MIT

"""
Shared helpers for zero-allocation regression tests.

`@allocated` can miss some paths or mask one-shot caches; scanning optimized
LLVM IR for Julia GC entrypoints catches heap traffic inside compiled code.
Names differ across Julia versions (`jl_*` vs `ijl_*`, `julia.gc_alloc`, …);
`count_llvm_gc_alloc_sites` tracks the union used in practice.

Note: `Pkg.test` runs Julia with `--check-bounds=yes` (`JLOptions().check_bounds == 1`).
Some small indexed kernels then show GC-related `call` lines in optimized IR even when
`@allocated` is zero. Prefer `@allocated` plus LLVM scans for large stable kernels
(`assemble!`, `apply_K!`, …); for tight index loops, gate IR scans or accept that CI
skips the IR assertion.
"""

using InteractiveUtils: code_llvm

# Every needle must appear only inside a `call` line in IR we care about.
const _LLVM_GC_ALLOC_MARKERS = (
    "julia.gc_alloc",
    "jl_gc_pool_alloc",
    "jl_gc_big_alloc",
    "jl_gc_alloc_typed",
    "jl_gc_small_alloc",
    "ijl_gc_pool_alloc",
    "ijl_gc_big_alloc",
    "ijl_gc_alloc_typed",
    "ijl_gc_small_alloc",
)

"""
    count_llvm_gc_alloc_sites(ir::AbstractString) -> Int

Count LLVM `call` lines that invoke a Julia GC allocation runtime function.
Used on the output of `llvm_ir` / `InteractiveUtils.code_llvm`.
"""
function count_llvm_gc_alloc_sites(ir::AbstractString)
    n = 0
    for line in eachsplit(ir, '\n')
        occursin("call ", line) || continue
        for needle in _LLVM_GC_ALLOC_MARKERS
            if occursin(needle, line)
                n += 1
                break
            end
        end
    end
    return n
end

"""
    llvm_ir(f, ::Type{<:Tuple}; optimize::Bool=true) -> String

Optimized LLVM IR for `f` at argument tuple type `T`, as a single string.
"""
function llvm_ir(@nospecialize(f), @nospecialize(T::Type{<:Tuple}); optimize::Bool = true)
    io = IOBuffer()
    code_llvm(io, f, T; optimize = optimize)
    return String(take!(io))
end

"""
    llvm_gc_alloc_site_count(f, ::Type{<:Tuple}; optimize::Bool=true) -> Int

Convenience: `count_llvm_gc_alloc_sites` ∘ `llvm_ir`.
"""
function llvm_gc_alloc_site_count(
    @nospecialize(f),
    @nospecialize(T::Type{<:Tuple});
    optimize::Bool = true,
)
    return count_llvm_gc_alloc_sites(llvm_ir(f, T; optimize = optimize))
end
