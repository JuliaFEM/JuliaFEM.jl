# SPDX-FileCopyrightText: 2015-2026 Jukka Aho
# SPDX-License-Identifier: MIT

using Tensors

"""Fourth-order symmetric identity 𝕀ˢʸᵐ for Lamé-style stiffness `λ I⊗I + 2μ 𝕀ˢʸᵐ`."""
@inline function symmetric_identity_tensor()
    return SymmetricTensor{4,3}((i, j, k, l) ->
        (i == k && j == l ? 0.5 : 0.0) + (i == l && j == k ? 0.5 : 0.0))
end
