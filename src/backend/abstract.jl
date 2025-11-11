# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
Backend abstraction for JuliaFEM solvers

This file defines the backend selection and dispatch mechanism.
Users never see backend types - everything is automatic!
"""

"""
    AbstractBackend

Abstract type for computation backend selection.

Subtypes:
- `Auto`: Automatic selection (GPU if available, else CPU)
- `GPU`: Force GPU backend
- `CPU`: Force CPU backend with specified threads
"""
abstract type AbstractBackend end

"""
    Auto <: AbstractBackend

Automatic backend selection. Chooses GPU if CUDA is available and functional,
otherwise falls back to CPU with all available threads.

# Example

```julia
sol = solve!(physics; backend=Auto())  # Default
```
"""
struct Auto <: AbstractBackend end

"""
    GPU <: AbstractBackend

Force GPU backend. Errors if CUDA is not available.

# Example

```julia
sol = solve!(physics; backend=GPU())
```
"""
struct GPU <: AbstractBackend end

"""
    CPU <: AbstractBackend

Force CPU backend with specified number of threads.

# Example

```julia
sol = solve!(physics; backend=CPU(8))  # Use 8 threads
```
"""
struct CPU <: AbstractBackend
    nthreads::Int
end

CPU() = CPU(Threads.nthreads())

"""
    select_backend(backend::AbstractBackend)

Select concrete backend based on user request and available hardware.

Returns GPU() or CPU(nthreads).
"""
function select_backend(::Auto)
    # Try GPU first if CUDA is available
    if @isdefined(CUDA)
        try
            if CUDA.functional()
                @info "Backend: GPU (automatic selection)"
                return GPU()
            end
        catch
            # CUDA exists but not functional
        end
    end

    # Fall back to CPU
    nthreads = Threads.nthreads()
    @info "Backend: CPU with $nthreads threads (automatic selection)"
    return CPU(nthreads)
end

function select_backend(::GPU)
    # Check if CUDA module exists in Main (user did 'using CUDA')
    if !isdefined(Main, :CUDA)
        error("GPU backend requested but CUDA package not loaded. Add 'using CUDA' before 'using JuliaFEM'.")
    end

    CUDA_mod = getfield(Main, :CUDA)

    if !CUDA_mod.functional()
        error("GPU backend requested but CUDA is not functional on this system.")
    end

    @info "Backend: GPU (user requested)"
    return GPU()
end

function select_backend(cpu::CPU)
    @info "Backend: CPU with $(cpu.nthreads) threads (user requested)"
    return cpu
end

"""
    AbstractElasticityData

Abstract type for backend-specific elasticity data.

Subtypes:
- `ElasticityDataGPU`: GPU-resident data (CuArrays)
- `ElasticityDataCPU`: CPU data with threading support
"""
abstract type AbstractElasticityData end

"""
    ElasticitySolution

Solution from elasticity solve.

Fields:
- `u`: Displacement vector (always CPU array, 3*n_nodes DOFs)
- `newton_iterations`: Number of Newton iterations (1 for linear problems)
- `cg_iterations`: Total CG iterations (across all Newton steps)
- `residual`: Final residual norm
- `solve_time`: Wall-clock time for solve (seconds)
- `history`: Newton-Krylov history [(cg_iters, R_norm, η), ...] (empty for linear)
"""
struct ElasticitySolution
    u::Vector{Float64}
    newton_iterations::Int
    cg_iterations::Int
    residual::Float64
    solve_time::Float64
    history::Vector{Tuple{Int,Float64,Float64}}  # (cg_iters, R_norm, η)
end

# Convenience constructor for linear problems (backward compatibility)
function ElasticitySolution(u::Vector{Float64}, iterations::Int, residual::Float64, solve_time::Float64)
    # Linear problem: 1 Newton iteration, no history
    return ElasticitySolution(u, 1, iterations, residual, solve_time, Tuple{Int,Float64,Float64}[])
end

"""
    solve!(physics::Physics{Elasticity}; backend=Auto(), kwargs...)

Solve elasticity problem with automatic or specified backend.

# Arguments
- `physics`: Problem definition with elements and BCs
- `backend`: Backend selection (Auto(), GPU(), or CPU(nthreads))
- `time`: Current time for time-dependent BCs (default: 0.0)
- `tol`: CG convergence tolerance (default: 1e-6)
- `max_iter`: Maximum CG iterations (default: 1000)

# Returns
- `ElasticitySolution`: Solution with displacement field u

# Examples

```julia
# Automatic backend (GPU if available, else CPU)
sol = solve!(physics)

# Force GPU
sol = solve!(physics; backend=GPU())

# Force CPU with 8 threads
sol = solve!(physics; backend=CPU(8))
```
"""
function solve!(physics::Physics{ElasticityPhysicsType};
    backend::AbstractBackend=Auto(),
    time::Float64=0.0,
    tol::Float64=1e-6,
    max_iter::Int=1000,
    newton_tol::Float64=1e-6,
    max_newton::Int=20,
    max_cg_per_newton::Int=50)

    # Select concrete backend
    backend_impl = select_backend(backend)

    # Initialize backend-specific data
    t0 = time_ns()
    data = initialize_backend(backend_impl, physics, time)

    # Solve using backend
    result = solve_backend!(data, physics;
        tol, max_iter,
        newton_tol, max_newton, max_cg_per_newton)
    t1 = time_ns()

    solve_time = (t1 - t0) / 1e9

    # Handle different return formats (backward compatibility)
    if length(result) == 3
        # Old format: (u, iterations, residual)
        u, iterations, residual = result
        return ElasticitySolution(Array(u), iterations, residual, solve_time)
    elseif length(result) == 5
        # New format: (u, newton_iters, cg_iters, residual, history)
        u, newton_iters, cg_iters, residual, history = result
        return ElasticitySolution(Array(u), newton_iters, cg_iters, residual, solve_time, history)
    else
        error("Unexpected return format from solve_backend!")
    end
end

"""
    initialize_backend(backend, physics, time)

Initialize backend-specific data from physics problem.

This function is implemented by backends:
- GPU backend in src/backend/gpu.jl (later ext/JuliaFEMCUDAExt/)
- CPU backend in src/backend/cpu.jl
"""
function initialize_backend end

"""
    solve_backend!(data, physics; tol, max_iter)

Solve elasticity problem using backend-specific data and algorithm.

Returns: (u, iterations, residual)

This function is implemented by backends:
- GPU backend in src/backend/gpu.jl (later ext/JuliaFEMCUDAExt/)
- CPU backend in src/backend/cpu.jl
"""
function solve_backend! end
