# JuliaFEM Technology Demonstrations

This directory contains demonstrations of key technologies and architectural decisions for JuliaFEM v1.0.

## Overview

These demos validate that type-stable field storage enables modern high-performance computing patterns: GPU execution, MPI communication, and Krylov iterative solvers.

## Demonstrations

### 1. GPU and MPI Communication (`gpu_mpi_demo.jl`)

**Purpose:** Prove that type-stable data structures flow efficiently to GPU and MPI.

**What it demonstrates:**

- Real CUDA GPU kernel execution
- MPI data transfer between processes
- Combined GPU+MPI workflow

**Run:**

```bash
mpiexec -np 2 julia --project=. demos/gpu_mpi_demo.jl
```

**Documentation:** [README_GPU_MPI.md](README_GPU_MPI.md)

### 2. Multi-GPU MPI Krylov Solver (`krylov_mpi_gpu_demo.jl`)

**Purpose:** Complete distributed FEM solver workflow with nodal assembly.

**What it demonstrates:**

- Nodal assembly pattern (row-by-row matrix construction)
- Distributed matrix-vector products
- Conjugate Gradient solver with MPI
- Multi-GPU execution
- Solution verification (10×10 SPD system)

**Run:**

```bash
mpiexec -np 2 julia --project=. demos/krylov_mpi_gpu_demo.jl
```

**Documentation:** [README_KRYLOV_DEMO.md](README_KRYLOV_DEMO.md)

**Results:**

- Converges in 9 iterations
- Relative error: 7.73 × 10⁻¹⁴
- Validates complete distributed solving workflow

## Requirements

### Required

- Julia 1.9+
- MPI installation (e.g., OpenMPI, MPICH)
- MPI.jl package

### Optional (for GPU demos)

- CUDA-capable GPU
- CUDA.jl package

If CUDA is not available, demos will fall back to CPU execution while still demonstrating the distributed computing patterns.

## Key Insights

### Type Stability is Not Optional

These demos prove that type-stable field storage is **required** (not just "nice to have") for:

| Feature | Why Type Stability Required |
|---------|----------------------------|
| GPU execution | CUDA kernels cannot compile with abstract types |
| Fast MPI | Typed buffers avoid serialization overhead |
| Krylov solvers | Matrix-free operators need concrete types |
| CPU performance | 9-92× speedup measured (see CPU benchmarks) |

### Nodal Assembly Pattern

The Krylov demo shows row-by-row matrix construction (`get_row()` abstraction), which:

- Aligns naturally with contact mechanics (nodal constraints)
- Enables domain decomposition (each rank owns nodes)
- Supports matrix-free solving (never form global matrix)
- Scales to large problems (O(N) memory vs O(N²))

### Architectural Validation

These demos validate the design decisions for JuliaFEM v1.0:

**v0.5.1 (2019):**

- Dict-based fields → Type instability
- Element assembly → Global matrix
- Direct solvers → O(N³) time, O(N²) memory
- Single-threaded CPU

**v1.0 (target, validated here):**

- Type-stable fields → GPU/MPI capable
- Nodal assembly → Distributed construction
- Krylov solvers → O(N·iter) time, O(N) memory
- Multi-GPU + MPI

## References

- **CPU benchmarks:** See `benchmarks/field_storage_comparison.jl` for 9-92× speedup measurements
- **Design documentation:** See `docs/book/zero_allocation_fields_v2.md` for architectural rationale
- **Session notes:** See `llm/sessions/2025-11-09_gpu_mpi_validation.md` for development history

## Contributing

These demos are educational and meant to be:

- **Clear:** Understand what's being demonstrated
- **Minimal:** No unnecessary complexity
- **Runnable:** Work on typical hardware (fall back to CPU if needed)
- **Validated:** Compare against exact solutions

When adding new demos, follow this pattern and document thoroughly.
