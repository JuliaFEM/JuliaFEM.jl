# src/assemblers/element_based/

Classic element-by-element assembler. The driver loops over elements,
calls the kernel for each one, and scatters the resulting block into a
global COO triplet store.

This is the more general path: it works for arbitrary element / kernel
combinations and is the entry point used by `assemble!(::COOAssembler, ...)`.

## Files

- `element_based_coo.jl` — `assemble!(::COOAssembler, ...)`, plus `create_cache(::COOAssembler, ...)`. Pulls in the scatter routines listed below.
- `scatter_blocks_to_triplets_symmetric_direct.jl` — symmetric block scatter into the COO triplet arrays.
- `scatter_blocks_to_force.jl` — block-structured force scatter.

The scatter routines are split per shape so that the compiler can pick
the right specialisation without runtime branches. Earlier alternative
scatter implementations (dense triplets, non-symmetric blocks, manually
unrolled symmetric blocks) were removed when none of the live drivers
referenced them.
