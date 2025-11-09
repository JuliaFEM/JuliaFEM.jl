# Running GPU and MPI Demonstrations

## Prerequisites

These demonstrations require MPI and CUDA to be installed globally:

```bash
# MPI is already installed on your system
# CUDA is already installed on your system
```

The packages are loaded dynamically, so they don't need to be in `Project.toml`.

## Running the Demonstrations

### MPI Communication Test

Test data transfer between 2 MPI processes:

```bash
cd /home/juajukka/dev/JuliaFEM.jl
mpirun -np 2 julia benchmarks/gpu_mpi_demo.jl
```

Expected output:

- Rank 0 sends typed arrays to Rank 1
- Shows bytes transferred
- Validates data integrity

### GPU + MPI Combined Test

If CUDA GPU is available, runs full workflow:

```bash
mpirun -np 2 julia benchmarks/gpu_mpi_demo.jl
```

Expected output:

- Detects GPU (if available)
- Compiles type-stable kernel for GPU
- Executes on real hardware
- Transfers results via MPI

### Single-Process GPU Test

To test GPU without MPI:

```bash
julia benchmarks/gpu_only_demo.jl
```

## What Gets Demonstrated

1. **Type Stability Requirement**
   - `Matrix{Float64}` transfers to GPU ✅
   - `Dict{String,Any}` would FAIL GPU compilation ❌

2. **MPI Fast Transfer**
   - Typed arrays: Fast buffer transfer (memcpy)
   - Mixed types: Slow serialization (~100× slower)

3. **Real Hardware Execution**
   - Actual CUDA kernel compilation and execution
   - Actual MPI inter-process communication
   - No mocks, no simulation

## Interpreting Results

Success indicators:

- ✅ "MPI communication successful" - Type-stable data transferred
- ✅ "GPU execution successful" - Kernel compiled and ran on GPU  
- ✅ "Combined GPU+MPI workflow successful" - End-to-end validated

The key insight: The same type-stable patterns that give 9-92× CPU speedup also enable GPU execution and efficient MPI communication.
