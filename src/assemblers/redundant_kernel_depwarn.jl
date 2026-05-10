# SPDX-FileCopyrightText: 2015-2026 Jukka Aho
# SPDX-License-Identifier: MIT

# One `Base.depwarn` per session for overloads that take a redundant volume `kernel`
# (ignored in favour of `cache.kernel_column` on the DOF-based path, or unused in
# symmetric Neumann `apply_load!` signatures).
const _REDUNDANT_KERNEL_ARG_DEPWARN = Ref(false)

@noinline function _depwarn_redundant_kernel_arg!(caller::Symbol)
    if !_REDUNDANT_KERNEL_ARG_DEPWARN[]
        _REDUNDANT_KERNEL_ARG_DEPWARN[] = true
        Base.depwarn(
            "overload with a redundant `kernel` argument is deprecated (it is not read; " *
            "DOF-based volume assembly reads kernels from `cache.kernel_column`). " *
            "Prefer `(cache, asm, mesh; …)`, `(y, cache, asm, mesh, x; …)`, and the same " *
            "pattern for matrix-free operators, preconditioners, eigensolve helpers, " *
            "partitioned matvec / MPI drivers, and `compute_block_diagonal!` / " *
            "`BlockJacobiPreconditioner` entry points without a separate volume kernel.",
            caller,
        )
    end
    return nothing
end
