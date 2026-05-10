# SPDX-FileCopyrightText: 2015-2026 Jukka Aho
# SPDX-License-Identifier: MIT

#=
Backend-agnostic GPU port of DOF-based matrix-free stiffness and mass
products using KernelAbstractions.jl.

The CPU `apply_K!` / `apply_M!` paths (in `dof_based_coo.jl`) iterate global
DOF rows and call `evaluate_entry` or `evaluate_mass_entry` on each incident
element. The same loops map to one device thread per DOF row without atomics.

This file provides:

* `DOFBasedCOOCacheKA{...}`  — mirror of the CPU cache carrying the batched
  geometry, `qp_buffers`, flat DOF maps, and flattened element-template layout.

* `apply_K_kernel!` / `apply_M_kernel!` — `@kernel` entry points.

* `apply_K!(y, cache_ka, kernel, x)` / `apply_M!(y, cache_ka, kernel, x)` —
  launchers (same precision contract as the CPU cache / vectors).

Pass 1 still runs on the CPU; after `sync_from_cpu!`, the matvec runs on the
chosen backend. Local validation uses the `CPU()` KA backend against the direct
CPU loops.
=#

using KernelAbstractions
using Adapt
using Tensors

using ..JuliaFEM: AbstractKernel
using ..JuliaFEM: DOFLayoutEntry, local_dof_layout, entity_local, component
using ..JuliaFEM: GeometryCache, evaluate_entry, evaluate_mass_entry
using ..JuliaFEM: DOFBasedCOOCache, _prepare_caches!
using ..JuliaFEM: PerElementKernelColumn, ka_per_element_kernel_column_supported

function _assert_ka_cpu_cache_kernel_column_supported!(cpu_cache::DOFBasedCOOCache)
    if cpu_cache.kernel_column isa PerElementKernelColumn &&
            !ka_per_element_kernel_column_supported(cpu_cache.kernel_column)
        throw(ArgumentError(
            "DOFBasedCOOCacheKA: this per-element kernel column is not supported on the KA/GPU " *
            "path (the launcher passes a single prototype kernel to `evaluate_entry` / `evaluate_mass_entry`). " *
            "Use CPU `apply_K!` / `apply_M!` with the CPU cache, or a column of " *
            "`ContinuumKernel` / `HeatKernel` / `ThermoElasticKernel` / `BiotPoroelasticKernel` / " *
            "`ThermoPoroelasticKernel` with pairwise-compatible kernels (see " *
            "`ka_per_element_kernel_column_supported`).",
        ))
    end
    return nothing
end


"""
    DOFBasedCOOCacheKA{T_∇N, T_W, T_QP, T_DOFS, T_CONN, T_LAY, T_CNT}

Backend-agnostic mirror of `DOFBasedCOOCache` carrying just the state
`apply_K!` / `apply_M!` read:

| field              | shape                                  | what  |
| --- | --- | --- |
| `∇N_batch`         | `(n_ips, max_nnodes, n_elems)`         | physical gradients |
| `detJ_w_batch`     | `(n_ips, n_elems)`                     | `detJ * weight`    |
| `qp_buffers`       | `(n_ips, n_elems)`                     | per-IP kernel buffer (e.g. elasticity tensor) |
| `elem_dofs`        | `(n_local_dofs, n_elems)`              | global DOF index per local DOF per element |
| `dof_elem_ids`     | `(max_connections, n_dofs)`            | element id of every DOF→element link |
| `dof_local_idx`    | `(max_connections, n_dofs)`            | local DOF index of every DOF→element link |
| `dof_counts`       | `(n_dofs,)`                            | how many element entries each DOF has |
| `layout_field`     | `(n_local_dofs,)`                      | `field_idx(layout[k])` for `k` in `1:N`        |
| `layout_entity`    | `(n_local_dofs,)`                      | `entity_local(layout[k])` for `k` in `1:N`     |
| `layout_component` | `(n_local_dofs,)`                      | `component(layout[k])` for `k` in `1:N`        |

The element-template DOF layout (compile-time `NTuple{N, DOFLayoutEntry}`)
is flattened into two small `Int16`/`Int8` arrays so the kernel can
look up `(entity_local, component)` by indexing instead of by tuple
destructuring (cheaper for both CPU vector code and SIMT divergence).

All array fields are `AbstractArray`-typed so `Adapt.adapt(backend, ka)`
swaps the storage in place — same struct, different backing.
"""
struct DOFBasedCOOCacheKA{T_X<:AbstractMatrix{<:Vec{3,<:AbstractFloat}},
                          T_N<:AbstractArray{<:AbstractFloat, 3},
                          T_∇N<:AbstractArray{<:Vec{3,<:AbstractFloat}, 3},
                          T_W<:AbstractMatrix{<:AbstractFloat},
                          T_QP<:AbstractMatrix,
                          T_DOFS<:AbstractMatrix{Int32},
                          T_EID<:AbstractMatrix{Int32},
                          T_LIDX<:AbstractMatrix{Int16},
                          T_CNT<:AbstractVector{Int32},
                          T_LF<:AbstractVector{Int8},
                          T_LE<:AbstractVector{Int16},
                          T_LC<:AbstractVector{Int8}}
    X_batch::T_X
    N_batch::T_N
    ∇N_batch::T_∇N
    detJ_w_batch::T_W
    qp_buffers::T_QP
    elem_dofs::T_DOFS
    dof_elem_ids::T_EID
    dof_local_idx::T_LIDX
    dof_counts::T_CNT
    layout_field::T_LF
    layout_entity::T_LE
    layout_component::T_LC
    n_local_dofs::Int32
    n_ips::Int32
end

# Adapt.jl: when adapted to a backend, swap each array (and re-build
# the struct with the new types).
Adapt.@adapt_structure DOFBasedCOOCacheKA


"""
    DOFBasedCOOCacheKA(cpu_cache::DOFBasedCOOCache; max_connections=20)

Build the backend-agnostic mirror on the CPU, drawing all flat
arrays from the existing CPU cache.

The concrete element type — needed to look up the compile-time
`local_dof_layout` — is read from the cache itself (`eltype(cpu.elements)`),
so callers don't have to pass it in.

Move the result to a different backend afterwards via
`Adapt.adapt(backend, ka)`. For `CPU()` backend, no copy is needed and
`adapt` is a no-op.

Pass 1 must have run on the CPU cache before `sync_from_cpu!` so that
`qp_buffers` is filled.
"""
function DOFBasedCOOCacheKA(cpu_cache::DOFBasedCOOCache;
                            max_connections::Int = 20)
    E = eltype(cpu_cache.elements)
    return _build_dof_based_coo_cache_ka(cpu_cache, E; max_connections)
end

function _build_dof_based_coo_cache_ka(cpu_cache::DOFBasedCOOCache, ::Type{E};
                                       max_connections::Int = 20) where {E}
    n_elems = length(cpu_cache.elements)
    n_dofs  = cpu_cache.ndofs

    # ---- Element-template DOF layout (compile-time NTuple) ----
    layout = local_dof_layout(E)
    n_local_dofs = length(layout)
    layout_field     = Vector{Int8}(undef,  n_local_dofs)
    layout_entity    = Vector{Int16}(undef, n_local_dofs)
    layout_component = Vector{Int8}(undef,  n_local_dofs)
    @inbounds for k in 1:n_local_dofs
        layout_field[k]     = Int8(layout[k].field_idx)
        layout_entity[k]    = Int16(entity_local(layout[k]))
        layout_component[k] = Int8(component(layout[k]))
    end

    # ---- Per-element DOF index table (flat (N, n_elems) matrix) ----
    elem_dofs = Matrix{Int32}(undef, n_local_dofs, n_elems)
    @inbounds for eid in 1:n_elems
        dofs = cpu_cache.element_caches[eid].dofs
        for k in 1:n_local_dofs
            elem_dofs[k, eid] = Int32(dofs[k])
        end
    end

    # ---- Flat DOF→element map (GPU-friendly fixed-size matrices) ----
    dof_to_elements = cpu_cache.dof_connectivity.dof_to_elements
    dof_elem_ids  = zeros(Int32, max_connections, n_dofs)
    dof_local_idx = zeros(Int16, max_connections, n_dofs)
    dof_counts    = zeros(Int32, n_dofs)
    @inbounds for dof_i in 1:n_dofs
        conns = dof_to_elements[dof_i]
        cnt   = length(conns)
        if cnt > max_connections
            error("DOFBasedCOOCacheKA: dof $dof_i has $cnt touching elements " *
                  "but max_connections=$max_connections. Increase the keyword.")
        end
        dof_counts[dof_i] = cnt
        for k in 1:cnt
            conn = conns[k]
            dof_elem_ids[k, dof_i]  = Int32(conn.elem_id)
            dof_local_idx[k, dof_i] = Int16(conn.local_dof_idx)
        end
    end

    n_ips = size(cpu_cache.detJ_w_batch, 1)
    _assert_ka_cpu_cache_kernel_column_supported!(cpu_cache)

    return DOFBasedCOOCacheKA(
        cpu_cache.X_batch,
        cpu_cache.N_batch,
        cpu_cache.∇N_batch,
        cpu_cache.detJ_w_batch,
        cpu_cache.qp_buffers,
        elem_dofs,
        dof_elem_ids,
        dof_local_idx,
        dof_counts,
        layout_field,
        layout_entity,
        layout_component,
        Int32(n_local_dofs),
        Int32(n_ips),
    )
end


"""
    to_float32(cpu_cache::DOFBasedCOOCache; max_connections = 20)
        -> DOFBasedCOOCacheKA  (with Float32 storage)

Convert a CPU `DOFBasedCOOCache` (stored in `Float64`) into a
Float32-storage KernelAbstractions mirror suitable for backends that
cannot hold double precision (Apple Metal, mobile / embedded GPUs).

The Float32 cache shares the same flat-array layout as the default
`DOFBasedCOOCacheKA(cpu_cache)` mirror — the only difference is element
type. Specifically:

| Field          | F64 cache type                          | F32 cache type                          |
| -------------- | --------------------------------------- | --------------------------------------- |
| `X_batch`      | `Matrix{Vec{3,Float64}}`                | `Matrix{Vec{3,Float32}}`                |
| `N_batch`      | `Array{Float64, 3}`                     | `Array{Float32, 3}`                     |
| `∇N_batch`     | `Array{Vec{3,Float64}, 3}`              | `Array{Vec{3,Float32}, 3}`              |
| `detJ_w_batch` | `Matrix{Float64}`                       | `Matrix{Float32}`                       |
| `qp_buffers`   | `Matrix{SymmetricTensor{4,3,Float64}}`  | `Matrix{SymmetricTensor{4,3,Float32}}`  |

Pass 1 is still run on the `Float64` CPU cache (so material constitutive
math stays in double precision); only the *storage* and the *device-side
arithmetic* in `apply_K_kernel!` drop to single precision. After
calling `to_float32`, the workflow is:

```julia
cache_ka_f32 = to_float32(cpu_cache)              # build F32 mirror
sync_from_cpu!(cache_ka_f32, cpu_cache)           # downcast Pass 1 outputs
cache_metal  = adapt(MetalBackend(), cache_ka_f32)
y32 = Metal.zeros(Float32, n)
x32 = Metal.rand(Float32, n)
apply_K!(y32, cache_metal, kernel, x32)           # runs on Metal in F32!
```

The resulting `apply_K!` output agrees with the Float64 reference to
within Float32 precision (`~1e-6` relative on well-conditioned
problems). This is the precision-parametric piece that closes the
"works on Apple Silicon" gap left by `F-next` — the test suite locks
the F32 / F64 agreement on the CPU backend, so the only remaining
device-specific code is the `adapt(MetalBackend(), ...)` call.
"""
function to_float32(cpu_cache::DOFBasedCOOCache; max_connections::Int = 20)
    E = eltype(cpu_cache.elements)
    return _build_dof_based_coo_cache_ka_f(cpu_cache, E, Float32; max_connections)
end

# Generic precision-aware builder. Replicates `_build_dof_based_coo_cache_ka`
# (the Float64 path) but allocates new geometry / qp arrays in the
# requested precision `F` and downcasts the CPU contents into them.
function _build_dof_based_coo_cache_ka_f(cpu_cache::DOFBasedCOOCache, ::Type{E},
                                         ::Type{F};
                                         max_connections::Int = 20) where {E, F<:AbstractFloat}
    n_elems = length(cpu_cache.elements)
    n_dofs  = cpu_cache.ndofs

    layout = local_dof_layout(E)
    n_local_dofs = length(layout)
    layout_field     = Vector{Int8}(undef,  n_local_dofs)
    layout_entity    = Vector{Int16}(undef, n_local_dofs)
    layout_component = Vector{Int8}(undef,  n_local_dofs)
    @inbounds for k in 1:n_local_dofs
        layout_field[k]     = Int8(layout[k].field_idx)
        layout_entity[k]    = Int16(entity_local(layout[k]))
        layout_component[k] = Int8(component(layout[k]))
    end

    elem_dofs = Matrix{Int32}(undef, n_local_dofs, n_elems)
    @inbounds for eid in 1:n_elems
        dofs = cpu_cache.element_caches[eid].dofs
        for k in 1:n_local_dofs
            elem_dofs[k, eid] = Int32(dofs[k])
        end
    end

    dof_to_elements = cpu_cache.dof_connectivity.dof_to_elements
    dof_elem_ids  = zeros(Int32, max_connections, n_dofs)
    dof_local_idx = zeros(Int16, max_connections, n_dofs)
    dof_counts    = zeros(Int32, n_dofs)
    @inbounds for dof_i in 1:n_dofs
        conns = dof_to_elements[dof_i]
        cnt   = length(conns)
        if cnt > max_connections
            error("DOFBasedCOOCacheKA: dof $dof_i has $cnt touching elements " *
                  "but max_connections=$max_connections. Increase the keyword.")
        end
        dof_counts[dof_i] = cnt
        for k in 1:cnt
            conn = conns[k]
            dof_elem_ids[k, dof_i]  = Int32(conn.elem_id)
            dof_local_idx[k, dof_i] = Int16(conn.local_dof_idx)
        end
    end

    n_ips = size(cpu_cache.detJ_w_batch, 1)
    _assert_ka_cpu_cache_kernel_column_supported!(cpu_cache)

    # Allocate fresh F-typed geometry batches. We don't share storage
    # with the Float64 CPU cache when F != Float64 — that would defeat
    # the purpose. The downcast happens in `sync_from_cpu!`.
    if F === eltype(cpu_cache.detJ_w_batch)
        # No-op fast-path for F == Float64: alias the CPU storage so
        # the CPU backend continues to share buffers (this is the
        # original `DOFBasedCOOCacheKA(cpu_cache)` behaviour).
        X_batch       = cpu_cache.X_batch
        N_batch       = cpu_cache.N_batch
        ∇N_batch      = cpu_cache.∇N_batch
        detJ_w_batch  = cpu_cache.detJ_w_batch
        qp_buffers    = cpu_cache.qp_buffers
    else
        max_nnodes = size(cpu_cache.X_batch, 1)
        X_batch      = Matrix{Vec{3,F}}(undef, max_nnodes, n_elems)
        N_batch      = Array{F, 3}(undef, n_ips, max_nnodes, n_elems)
        ∇N_batch     = Array{Vec{3,F}, 3}(undef, n_ips, max_nnodes, n_elems)
        detJ_w_batch = Matrix{F}(undef, n_ips, n_elems)
        # qp_buffer's element type also follows F: convert each tensor
        # entry's element type. Works for SymmetricTensor{2,3,_,6},
        # SymmetricTensor{4,3,_,36}, and any future kernel-defined buffer
        # type that supports `convert(NewType, x)`.
        Buf_F64 = eltype(cpu_cache.qp_buffers)
        Buf_F   = _convert_buffer_eltype(Buf_F64, F)
        qp_buffers = Matrix{Buf_F}(undef, n_ips, n_elems)
    end

    return DOFBasedCOOCacheKA(
        X_batch,
        N_batch,
        ∇N_batch,
        detJ_w_batch,
        qp_buffers,
        elem_dofs,
        dof_elem_ids,
        dof_local_idx,
        dof_counts,
        layout_field,
        layout_entity,
        layout_component,
        Int32(n_local_dofs),
        Int32(n_ips),
    )
end

# Map a Float64-parametric tensor type (`SymmetricTensor{2,3,Float64,6}`,
# `SymmetricTensor{4,3,Float64,36}`, …) to its F-parametric counterpart.
# Falls back to `F` itself for plain scalar buffer eltypes.
@inline _convert_buffer_eltype(::Type{SymmetricTensor{O,D,Float64,L}}, ::Type{F}) where {O,D,L,F<:AbstractFloat} =
    SymmetricTensor{O,D,F,L}
@inline _convert_buffer_eltype(::Type{Tensor{O,D,Float64,L}}, ::Type{F}) where {O,D,L,F<:AbstractFloat} =
    Tensor{O,D,F,L}
@inline _convert_buffer_eltype(::Type{Vec{D,Float64}}, ::Type{F}) where {D,F<:AbstractFloat} =
    Vec{D,F}
@inline _convert_buffer_eltype(::Type{Float64}, ::Type{F}) where {F<:AbstractFloat} = F
@inline _convert_buffer_eltype(::Type{T}, ::Type{F}) where {T,F<:AbstractFloat} = T  # leave anything else alone


"""
    sync_from_cpu!(ka::DOFBasedCOOCacheKA, cpu::DOFBasedCOOCache)

After Pass 1 has populated `cpu.∇N_batch / detJ_w_batch / qp_buffers`,
copy the values into `ka` so the next `apply_K!(ka, ...)` sees them.

For the CPU backend this is a no-op when `ka.∇N_batch === cpu.∇N_batch`
(the constructor aliases by default). For a GPU backend, override with a
backend-specific method that does `copyto!(ka.∇N_batch, cpu.∇N_batch)`
across the host/device boundary.
"""
function sync_from_cpu!(ka::DOFBasedCOOCacheKA, cpu::DOFBasedCOOCache)
    # Aliased fast-path: when the KA cache shares storage with the CPU
    # cache (the F == Float64 default), Pass 1's writes are already
    # visible — nothing to do.
    if ka.∇N_batch === cpu.∇N_batch
        return ka
    end

    # Independent storage: copy each batch with element-wise precision
    # conversion. `_copyto_with_convert!` handles both same-precision
    # (cheap memcpy) and Float64→Float32 (downcast) without allocation
    # in the hot path.
    _copyto_with_convert!(ka.X_batch,      cpu.X_batch)
    _copyto_with_convert!(ka.N_batch,      cpu.N_batch)
    _copyto_with_convert!(ka.∇N_batch,     cpu.∇N_batch)
    _copyto_with_convert!(ka.detJ_w_batch, cpu.detJ_w_batch)
    _copyto_with_convert!(ka.qp_buffers,   cpu.qp_buffers)
    return ka
end

# Element-wise typed copy: `dest .= convert(Tdest, src)`. Generic over
# scalars, `Vec`, `Tensor`, and `SymmetricTensor` element types because
# Tensors.jl ships `convert` methods for all of them.
function _copyto_with_convert!(dest::AbstractArray{Td}, src::AbstractArray) where {Td}
    @assert size(dest) == size(src) (
        "_copyto_with_convert!: shape mismatch $(size(dest)) vs $(size(src))")
    if eltype(src) === Td
        copyto!(dest, src)
    else
        @inbounds for i in eachindex(src)
            dest[i] = convert(Td, src[i])
        end
    end
    return dest
end


# ============================================================================
# KA kernel — one thread per DOF row.
#
# Each thread:
#   1. reads the count of touching (elem, local_i) pairs for its row,
#   2. for each pair, reads (eid, local_i),
#   3. constructs a per-element `GeometryCache` over views into the SoA
#      batches and a column view into `qp_buffers`,
#   4. loops `local_j in 1:n_local_dofs` and calls the kernel-agnostic
#      `evaluate_entry(kernel, geom, qp_buffer, layout_i, layout_j, elem_id)` —
#      the *same* function the CPU `assemble!` / `apply_K!` paths call, so
#      adding a new kernel only requires its `evaluate_entry` impl, no
#      GPU-specific duplication.
#   5. accumulates `K_ij * x[dof_j_global]` into a register `yi` and
#      writes `y[dof_i] = yi` exactly once.
#
# No atomics are needed — every `y[i]` has a single writer.
# ============================================================================

@kernel function apply_K_kernel!(
    y,                    # AbstractVector{F} where F is Float32 or Float64
    @Const(x),            # AbstractVector{F}
    @Const(X_batch),
    @Const(N_batch),
    @Const(∇N_batch),
    @Const(detJ_w_batch),
    @Const(qp_buffers),
    @Const(elem_dofs),
    @Const(dof_elem_ids),
    @Const(dof_local_idx),
    @Const(dof_counts),
    @Const(layout_field),
    @Const(layout_entity),
    @Const(layout_component),
    n_local_dofs::Int32,
    n_ips::Int32,
    kernel,
)
    dof_i = @index(Global)

    # The accumulator picks up the precision of the geometry / vector
    # storage automatically — `Float64` on the default CPU+CUDA path,
    # `Float32` on Apple Metal where `Float64` is unsupported.
    F   = eltype(y)
    cnt = dof_counts[dof_i]
    yi  = zero(F)

    @inbounds for k in 1:cnt
        eid     = dof_elem_ids[k, dof_i]
        local_i = dof_local_idx[k, dof_i]

        # View-backed per-element GeometryCache, constructed inline.
        # Same struct shape as the CPU cache so `evaluate_entry`
        # dispatches without device-specific code paths.
        geom = GeometryCache(
            view(X_batch, :, eid),
            view(N_batch, :, :, eid),
            view(∇N_batch, :, :, eid),
            view(detJ_w_batch, :, eid),
        )
        qp_buffer = view(qp_buffers, :, eid)

        ent_i = DOFLayoutEntry(layout_field[local_i], layout_entity[local_i], layout_component[local_i])

        for local_j in 1:n_local_dofs
            ent_j = DOFLayoutEntry(layout_field[local_j], layout_entity[local_j], layout_component[local_j])

            K_ij = evaluate_entry(kernel, geom, qp_buffer, ent_i, ent_j, Int(eid))

            dof_j_global = elem_dofs[local_j, eid]
            yi += K_ij * x[dof_j_global]
        end
    end

    @inbounds y[dof_i] = yi
end


# ============================================================================
# KA kernel — mass matvec `y = M * x` (same row layout as `apply_K_kernel!`).
# ============================================================================

@kernel function apply_M_kernel!(
    y,
    @Const(x),
    @Const(X_batch),
    @Const(N_batch),
    @Const(∇N_batch),
    @Const(detJ_w_batch),
    @Const(qp_buffers),
    @Const(elem_dofs),
    @Const(dof_elem_ids),
    @Const(dof_local_idx),
    @Const(dof_counts),
    @Const(layout_field),
    @Const(layout_entity),
    @Const(layout_component),
    n_local_dofs::Int32,
    n_ips::Int32,
    kernel,
)
    dof_i = @index(Global)

    F   = eltype(y)
    cnt = dof_counts[dof_i]
    yi  = zero(F)

    @inbounds for k in 1:cnt
        eid     = dof_elem_ids[k, dof_i]
        local_i = dof_local_idx[k, dof_i]

        geom = GeometryCache(
            view(X_batch, :, eid),
            view(N_batch, :, :, eid),
            view(∇N_batch, :, :, eid),
            view(detJ_w_batch, :, eid),
        )
        qp_buffer = view(qp_buffers, :, eid)

        ent_i = DOFLayoutEntry(layout_field[local_i], layout_entity[local_i], layout_component[local_i])

        for local_j in 1:n_local_dofs
            ent_j = DOFLayoutEntry(layout_field[local_j], layout_entity[local_j], layout_component[local_j])

            M_ij = evaluate_mass_entry(kernel, geom, qp_buffer, ent_i, ent_j)

            dof_j_global = elem_dofs[local_j, eid]
            yi += M_ij * x[dof_j_global]
        end
    end

    @inbounds y[dof_i] = yi
end


"""
    apply_K!(y::AbstractVector{Float64},
             cache_ka::DOFBasedCOOCacheKA,
             kernel::AbstractKernel,
             x::AbstractVector{Float64};
             workgroupsize::Int = 256)

Backend-agnostic matrix-free `y = K * x` on top of the KA cache.

Backend is inferred from `y` (so a `CuArray` `y` runs on `CUDABackend`,
a `MtlArray` `y` runs on `MetalBackend`, an ordinary `Array` runs on
`CPU()`).

Caller is responsible for having Pass 1 already populated the geometry
batches and qp buffers (run `_prepare_caches!` on the CPU cache, then
`sync_from_cpu!(cache_ka, cpu_cache)`).

# Example — CPU

```julia
using JuliaFEM
cache    = DOFBasedCOOCache(elements, dof_handler, mesh, kernel)
asm      = DOFBasedCOOAssembler()
assemble!(cache, asm, kernel, mesh)               # Pass 1 + COO

cache_ka = DOFBasedCOOCacheKA(cache)              # CPU mirror
sync_from_cpu!(cache_ka, cache)                   # publish Pass 1 outputs

y = zeros(cache.ndofs); x = randn(cache.ndofs)
apply_K!(y, cache_ka, kernel, x)                   # runs on CPU() backend
```

# Example — CUDA

```julia
using JuliaFEM, CUDA, Adapt
# build cache + run Pass 1 on CPU as above
cache_gpu = adapt(CUDABackend(), DOFBasedCOOCacheKA(cache))
sync_from_cpu!(cache_gpu, cache)                  # H2D copies

y = CUDA.zeros(Float64, cache.ndofs)
x = CUDA.rand(Float64, cache.ndofs)
apply_K!(y, cache_gpu, kernel, x)                  # runs on CUDABackend()
```

The same pattern applies to `Metal.jl` (`MetalBackend()`),
`AMDGPU.jl` (`ROCBackend()`), and `oneAPI.jl` (`oneAPIBackend()`). No
extension code is needed in JuliaFEM itself — `Adapt.@adapt_structure
DOFBasedCOOCacheKA` plus the GPU package's own `Adapt` rules for its
device array type take care of the storage move; KA dispatches the
kernel to the right backend.
"""
function apply_K!(
    y::AbstractVector{F},
    cache_ka::DOFBasedCOOCacheKA,
    kernel::AbstractKernel,
    x::AbstractVector{F};
    workgroupsize::Int = 256,
) where {F<:AbstractFloat}
    @assert length(y) == length(cache_ka.dof_counts)
    @assert length(x) == length(cache_ka.dof_counts)
    @assert eltype(cache_ka.detJ_w_batch) === F (
        "apply_K!: y/x precision $F mismatches cache precision " *
        "$(eltype(cache_ka.detJ_w_batch)) — use `to_float32(cache)` to build a F32 cache.")

    backend = get_backend(y)
    krn = apply_K_kernel!(backend, workgroupsize)
    krn(
        y, x,
        cache_ka.X_batch,
        cache_ka.N_batch,
        cache_ka.∇N_batch,
        cache_ka.detJ_w_batch,
        cache_ka.qp_buffers,
        cache_ka.elem_dofs,
        cache_ka.dof_elem_ids,
        cache_ka.dof_local_idx,
        cache_ka.dof_counts,
        cache_ka.layout_field,
        cache_ka.layout_entity,
        cache_ka.layout_component,
        cache_ka.n_local_dofs,
        cache_ka.n_ips,
        kernel;
        ndrange = length(y),
    )
    KernelAbstractions.synchronize(backend)
    return y
end


"""
    apply_M!(y, cache_ka::DOFBasedCOOCacheKA, kernel::AbstractKernel, x;
              workgroupsize::Int = 256) -> y

Matrix-free mass product `y = M * x` on the KA cache, calling
`evaluate_mass_entry` with the same geometry / `qp_buffers` views as the CPU
`apply_M!(y, cache, assembler, mesh, x)` in `dof_based_coo.jl`.

Requires the same Pass~1 + `sync_from_cpu!` preparation as `apply_K!`.
Per-element kernel columns must satisfy `ka_per_element_kernel_column_supported`
(the same gate as stiffness: prototype kernel plus per-element buffers).
"""
function apply_M!(
    y::AbstractVector{F},
    cache_ka::DOFBasedCOOCacheKA,
    kernel::AbstractKernel,
    x::AbstractVector{F};
    workgroupsize::Int = 256,
) where {F<:AbstractFloat}
    @assert length(y) == length(cache_ka.dof_counts)
    @assert length(x) == length(cache_ka.dof_counts)
    @assert eltype(cache_ka.detJ_w_batch) === F (
        "apply_M!: y/x precision $F mismatches cache precision " *
        "$(eltype(cache_ka.detJ_w_batch)) — use `to_float32(cache)` to build a F32 cache.")

    backend = get_backend(y)
    krn = apply_M_kernel!(backend, workgroupsize)
    krn(
        y, x,
        cache_ka.X_batch,
        cache_ka.N_batch,
        cache_ka.∇N_batch,
        cache_ka.detJ_w_batch,
        cache_ka.qp_buffers,
        cache_ka.elem_dofs,
        cache_ka.dof_elem_ids,
        cache_ka.dof_local_idx,
        cache_ka.dof_counts,
        cache_ka.layout_field,
        cache_ka.layout_entity,
        cache_ka.layout_component,
        cache_ka.n_local_dofs,
        cache_ka.n_ips,
        kernel;
        ndrange = length(y),
    )
    KernelAbstractions.synchronize(backend)
    return y
end
