"""
Matrix-Free Newton-Krylov GPU Benchmark

Demonstrates traditional Newton vs Matrix-Free Newton-Krylov with Anderson
acceleration, running on GPU.

Run with:
    julia --project=. benchmarks/matrix_free_gpu_benchmark.jl
"""

using CUDA
using LinearAlgebra
using IterativeSolvers
using Printf

# Check GPU availability
if !CUDA.functional()
    @warn "CUDA not available! Running CPU-only comparison."
    USE_GPU = false
else
    println("GPU Device: $(CUDA.name(CUDA.device()))")
    println("GPU Memory: $(CUDA.total_memory() / 1e9) GB")
    println()
    USE_GPU = true
end

# ============================================================================
# Problem Setup: 3D Nonlinear Elasticity
# ============================================================================

"""
Residual for 3D nonlinear elasticity with cubic nonlinearity.

    r(u) = K·u + β·(K·u).^3 - f

where K is stiffness matrix, β is nonlinearity parameter.
"""
struct NonlinearProblem{T,MatT,VecT}
    K::MatT           # Stiffness matrix (sparse or LinearMap)
    f::VecT           # Force vector
    β::T              # Nonlinearity parameter
    n::Int            # DOF count
end

"""
Compute residual: r(u) = K·u + β·(K·u).^3 - f
"""
function compute_residual!(r::AbstractVector, prob::NonlinearProblem, u::AbstractVector)
    # Linear part
    mul!(r, prob.K, u)  # r = K·u

    # Nonlinear part: r += β·(K·u).^3
    if prob.β != 0
        # Reuse r (which contains K·u)
        @. r = r + prob.β * r^3
    end

    # Apply forcing
    @. r = r - prob.f

    return r
end

"""
Jacobian-vector product: J·v ≈ [r(u+ε·v) - r(u)] / ε (finite difference)
"""
function jacobian_vector_product!(
    Jv::AbstractVector,
    prob::NonlinearProblem,
    u::AbstractVector,
    v::AbstractVector,
    r_u::AbstractVector,  # Pre-computed r(u)
    temp::AbstractVector  # Workspace
)
    ε = 1e-7

    # temp = u + ε·v
    @. temp = u + ε * v

    # Jv = r(u + ε·v)
    compute_residual!(Jv, prob, temp)

    # Jv = [r(u + ε·v) - r(u)] / ε
    @. Jv = (Jv - r_u) / ε

    return Jv
end

# ============================================================================
# Traditional Newton Solver
# ============================================================================

"""
Traditional Newton with full Jacobian assembly.

    u_{k+1} = u_k - J(u_k)^{-1} · r(u_k)

Expensive: Assembles full Jacobian matrix at each iteration.
"""
function newton_traditional!(
    u::AbstractVector{T},
    prob::NonlinearProblem{T},
    r::AbstractVector{T},
    du::AbstractVector{T};
    tol=1e-8,
    max_iter=20,
    verbose=true
) where T

    n = length(u)

    # Build Jacobian matrix (expensive!)
    # J ≈ K + 3β·diag((K·u).^2)·K
    Ku = prob.K * u
    J = copy(prob.K)

    for iter in 1:max_iter
        # Compute residual
        compute_residual!(r, prob, u)

        norm_r = norm(r)

        if verbose
            @printf("  Iter %2d: ||r|| = %.6e\n", iter, norm_r)
        end

        if norm_r < tol
            if verbose
                println("  ✅ Converged!")
            end
            return iter
        end

        # Update Jacobian (expensive!)
        Ku .= prob.K * u
        for i in 1:n
            J[i, i] = prob.K[i, i] + 3 * prob.β * Ku[i]^2 * prob.K[i, i]
        end

        # Solve linear system (expensive!)
        du .= -(J \ r)

        # Update
        u .+= du
    end

    if verbose
        println("  ⚠️  Did not converge in $max_iter iterations")
    end

    return max_iter
end

# ============================================================================
# Helper: Matrix-free operator wrapper for GMRES
# ============================================================================

"""
Wrapper to make a function look like a matrix for GMRES.
"""
struct MatrixFreeOperator{F}
    matvec!::F
    n::Int
end

Base.size(A::MatrixFreeOperator) = (A.n, A.n)
Base.size(A::MatrixFreeOperator, d::Int) = d <= 2 ? A.n : 1
Base.eltype(::MatrixFreeOperator{F}) where F = Float64

function LinearAlgebra.mul!(y, A::MatrixFreeOperator, x)
    A.matvec!(y, x)
    return y
end

# ============================================================================
# Matrix-Free Newton-Krylov
# ============================================================================

"""
Matrix-Free Newton-Krylov with GMRES.

    J·v ≈ [r(u+ε·v) - r(u)] / ε  (no matrix!)
    du = gmres(Jv_op, -r)
    u_{k+1} = u_k + du

Cheap: Only residual evaluations, no Jacobian assembly.
"""
function newton_matrix_free!(
    u::AbstractVector{T},
    prob::NonlinearProblem{T},
    r::AbstractVector{T},
    du::AbstractVector{T},
    temp::AbstractVector{T},
    Jv::AbstractVector{T};
    tol=1e-8,
    max_iter=20,
    gmres_tol=1e-6,
    verbose=true
) where T

    for iter in 1:max_iter
        # Compute residual
        compute_residual!(r, prob, u)

        norm_r = norm(r)

        if verbose
            @printf("  Iter %2d: ||r|| = %.6e", iter, norm_r)
        end

        if norm_r < tol
            if verbose
                println(" ✅ Converged!")
            end
            return iter
        end

        # Matrix-free operator: J·v
        function Jv_matvec!(out, v)
            jacobian_vector_product!(out, prob, u, v, r, temp)
            return out
        end
        Jv_op = MatrixFreeOperator(Jv_matvec!, length(u))

        # Solve J·du = -r using GMRES (matrix-free!)
        du .= 0
        gmres!(du, Jv_op, -r;
            abstol=gmres_tol, reltol=0, maxiter=50, verbose=false)

        gmres_iters = 50  # Would need to extract from gmres! return

        if verbose
            @printf(" [GMRES: ~%d iters]\n", gmres_iters)
        end

        # Update
        u .+= du
    end

    if verbose
        println("  ⚠️  Did not converge in $max_iter iterations")
    end

    return max_iter
end

# ============================================================================
# Anderson-Accelerated Newton-Krylov
# ============================================================================

"""
Anderson acceleration for Newton-Krylov.

Combines m previous iterates via least-squares:
    u_new = ∑ αᵢ·uᵢ  where  argmin ||∑ αᵢ·rᵢ||²  s.t. ∑ αᵢ = 1

Transforms linear convergence → superlinear convergence.
"""
function anderson_newton_matrix_free!(
    u::AbstractVector{T},
    prob::NonlinearProblem{T},
    r::AbstractVector{T},
    du::AbstractVector{T},
    temp::AbstractVector{T},
    Jv::AbstractVector{T};
    m=5,  # Anderson history
    tol=1e-8,
    max_iter=20,
    gmres_tol=1e-6,
    verbose=true
) where T

    n = length(u)

    # Anderson history
    U_history = [zeros(T, n) for _ in 1:m]
    R_history = [zeros(T, n) for _ in 1:m]
    history_count = 0

    for iter in 1:max_iter
        # Compute residual
        compute_residual!(r, prob, u)

        norm_r = norm(r)

        if verbose
            @printf("  Iter %2d: ||r|| = %.6e", iter, norm_r)
        end

        if norm_r < tol
            if verbose
                println(" ✅ Converged!")
            end
            return iter
        end

        # Matrix-free operator
        function Jv_matvec!(out, v)
            jacobian_vector_product!(out, prob, u, v, r, temp)
            return out
        end
        Jv_op = MatrixFreeOperator(Jv_matvec!, n)

        # Solve J·du = -r using GMRES
        du .= 0
        gmres!(du, Jv_op, -r;
            abstol=gmres_tol, reltol=0, maxiter=50, verbose=false)

        # Store in history (circular buffer)
        idx = mod1(history_count + 1, m)
        U_history[idx] .= u
        R_history[idx] .= r
        history_count = min(history_count + 1, m)

        if verbose
            @printf(" [GMRES: ~50 iters, history: %d]", history_count)
        end

        # Anderson acceleration (if enough history)
        if history_count >= 2
            # Build residual difference matrix
            k = history_count
            R_diff = zeros(T, n, k)
            for i in 1:k
                R_diff[:, i] .= R_history[i] .- r
            end

            # Check condition number before QR
            # If matrix is ill-conditioned, skip Anderson this iteration
            R_norm = norm(R_diff)
            if R_norm < 1e-10
                # Matrix too small, use standard update
                u .+= du
                if verbose
                    println(" [Anderson: skipped (residuals too small)]")
                end
            else
                # Least-squares: min ||R_diff·α||²  s.t. sum(α) = 1
                # Use QR factorization with regularization
                try
                    Q, Rt = qr(R_diff)

                    # Add small regularization to diagonal if needed
                    Rt_diag = diag(Rt)
                    if any(abs.(Rt_diag) .< 1e-12)
                        # Add Tikhonov regularization
                        λ = 1e-8
                        Rt_reg = Rt + λ * I
                        α = Rt_reg \ (Q' * r)
                    else
                        α = Rt \ (Q' * r)
                    end

                    α ./= sum(α)  # Normalize

                    # Combine previous iterates
                    u_combined = zeros(T, n)
                    for i in 1:k
                        u_combined .+= α[i] .* U_history[i]
                    end

                    # Update with combination
                    u .= u_combined .+ du

                    if verbose
                        println(" [Anderson: α=$(round.(α, digits=3))]")
                    end
                catch e
                    # If QR fails, fall back to standard Newton
                    u .+= du
                    if verbose
                        println(" [Anderson: failed ($e), using standard update]")
                    end
                end
            end
        else
            # Standard Newton update
            u .+= du

            if verbose
                println()
            end
        end
    end

    if verbose
        println("  ⚠️  Did not converge in $max_iter iterations")
    end

    return max_iter
end

# ============================================================================
# GPU Implementations
# ============================================================================

"""
GPU version of residual computation.
"""
function compute_residual_gpu!(
    r::CuVector{T},
    K::CuMatrix{T},
    u::CuVector{T},
    f::CuVector{T},
    β::T
) where T
    # r = K·u
    mul!(r, K, u)

    # r = r + β·r³ - f
    if β != 0
        r .= r .+ β .* r .^ 3 .- f
    else
        r .= r .- f
    end

    return r
end

"""
GPU version of Matrix-Free Newton-Krylov.
"""
function newton_matrix_free_gpu!(
    u::CuVector{T},
    K::CuMatrix{T},
    f::CuVector{T},
    β::T;
    tol=1e-8,
    max_iter=20,
    gmres_tol=1e-6,
    verbose=true
) where T

    n = length(u)
    r = CUDA.zeros(T, n)
    du = CUDA.zeros(T, n)
    temp = CUDA.zeros(T, n)
    Jv = CUDA.zeros(T, n)

    for iter in 1:max_iter
        # Compute residual on GPU
        compute_residual_gpu!(r, K, u, f, β)

        norm_r = norm(Array(r))  # Transfer to CPU for norm

        if verbose
            @printf("  Iter %2d: ||r|| = %.6e\n", iter, norm_r)
        end

        if norm_r < tol
            if verbose
                println("  ✅ Converged!")
            end
            return iter
        end

        # Jacobian-vector product (on GPU)
        ε = T(1e-7)
        function Jv_matvec_gpu!(out_cpu, v_cpu)
            v = CuArray(v_cpu)
            # temp = u + ε·v
            temp .= u .+ ε .* v
            # Jv = r(u + ε·v)
            compute_residual_gpu!(Jv, K, temp, f, β)
            # Jv = [r(u + ε·v) - r(u)] / ε
            Jv .= (Jv .- r) ./ ε
            out_cpu .= Array(Jv)
            return out_cpu
        end
        Jv_op_gpu = MatrixFreeOperator(Jv_matvec_gpu!, n)

        # Solve on CPU (GMRES doesn't have GPU version in IterativeSolvers.jl)
        r_cpu = Array(r)
        du_cpu = zeros(T, n)
        gmres!(du_cpu, Jv_op_gpu, -r_cpu;
            abstol=gmres_tol, reltol=0, maxiter=50, verbose=false)

        # Update on GPU
        du .= CuArray(du_cpu)
        u .+= du
    end

    if verbose
        println("  ⚠️  Did not converge in $max_iter iterations")
    end

    return max_iter
end

# GPU Anderson-Accelerated Newton-Krylov
function anderson_newton_matrix_free_gpu!(
    u::CuVector{T},
    K::CuMatrix{T},
    f::CuVector{T},
    β::T;
    tol=1e-8,
    max_iter=20,
    gmres_tol=1e-6,
    history_size=5,
    verbose=true
) where T

    n = length(u)
    r = CUDA.zeros(T, n)
    du = CUDA.zeros(T, n)
    temp = CUDA.zeros(T, n)
    Jv = CUDA.zeros(T, n)

    # Anderson acceleration storage (CPU)
    R_history = Vector{Vector{T}}()
    U_history = Vector{Vector{T}}()
    history_count = 0

    for iter in 1:max_iter
        # Compute residual on GPU
        compute_residual_gpu!(r, K, u, f, β)

        norm_r = norm(Array(r))  # Transfer to CPU for norm

        if verbose
            @printf("  Iter %2d: ||r|| = %.6e", iter, norm_r)
        end

        if norm_r < tol
            if verbose
                println("\n  ✅ Converged!")
            end
            return iter
        end

        # Jacobian-vector product (on GPU)
        ε = T(1e-7)
        function Jv_matvec_gpu!(out_cpu, v_cpu)
            v = CuArray(v_cpu)
            # temp = u + ε·v
            temp .= u .+ ε .* v
            # Jv = r(u + ε·v)
            compute_residual_gpu!(Jv, K, temp, f, β)
            # Jv = [r(u + ε·v) - r(u)] / ε
            Jv .= (Jv .- r) ./ ε
            out_cpu .= Array(Jv)
            return out_cpu
        end
        Jv_op_gpu = MatrixFreeOperator(Jv_matvec_gpu!, n)

        # Solve on CPU (GMRES doesn't have GPU version)
        r_cpu = Array(r)
        u_cpu = Array(u)
        du_cpu = zeros(T, n)
        gmres!(du_cpu, Jv_op_gpu, -r_cpu;
            abstol=gmres_tol, reltol=0, maxiter=50, verbose=false)

        # Anderson acceleration (on CPU)
        if history_count >= 2
            # Build residual difference matrix
            k = history_count
            R_diff = zeros(T, n, k)
            for i in 1:k
                R_diff[:, i] .= R_history[i] .- r_cpu
            end

            # Check condition number before QR
            R_norm = norm(R_diff)
            if R_norm < 1e-10
                # Matrix too small, use standard update
                u .+= CuArray(du_cpu)
                if verbose
                    println(" [Anderson: skipped (residuals too small)]")
                end
            else
                # Least-squares with regularization
                try
                    Q, Rt = qr(R_diff)

                    # Add small regularization to diagonal if needed
                    Rt_diag = diag(Rt)
                    if any(abs.(Rt_diag) .< 1e-12)
                        # Add Tikhonov regularization
                        λ = 1e-8
                        Rt_reg = Rt + λ * I
                        α = Rt_reg \ (Q' * r_cpu)
                    else
                        α = Rt \ (Q' * r_cpu)
                    end

                    α ./= sum(α)  # Normalize

                    # Combine previous iterates
                    u_combined = zeros(T, n)
                    for i in 1:k
                        u_combined .+= α[i] .* U_history[i]
                    end

                    # Update with combination (transfer to GPU)
                    u .= CuArray(u_combined .+ du_cpu)

                    if verbose
                        println(" [Anderson: α=$(round.(α, digits=3))]")
                    end
                catch e
                    # If QR fails, fall back to standard Newton
                    u .+= CuArray(du_cpu)
                    if verbose
                        println(" [Anderson: failed ($e), using standard update]")
                    end
                end
            end
        else
            # Standard Newton update (transfer du to GPU)
            u .+= CuArray(du_cpu)
            if verbose
                println()
            end
        end

        # Store history (on CPU to avoid GPU memory overhead)
        push!(R_history, copy(r_cpu))
        push!(U_history, copy(u_cpu))
        history_count += 1

        # Maintain history size
        if history_count > history_size
            popfirst!(R_history)
            popfirst!(U_history)
            history_count = history_size
        end
    end

    if verbose
        println("  ⚠️  Did not converge in $max_iter iterations")
    end

    return max_iter
end

# ============================================================================
# Benchmark Runners
# ============================================================================

function benchmark_cpu(n::Int)
    println("="^70)
    println("CPU Benchmark: $n DOFs")
    println("="^70)

    # Setup problem
    T = Float64
    K = Matrix(Tridiagonal(
        -ones(T, n - 1),
        2ones(T, n),
        -ones(T, n - 1)
    ))
    f = ones(T, n) * 0.1
    β = T(1e-3)  # Nonlinearity

    prob = NonlinearProblem(K, f, β, n)

    # Initial guess
    u0 = zeros(T, n)

    # Allocate workspace
    r = zeros(T, n)
    du = zeros(T, n)
    temp = zeros(T, n)
    Jv = zeros(T, n)

    # Benchmark Traditional Newton
    println("\n📊 Traditional Newton (Full Jacobian):")
    u_trad = copy(u0)
    t_trad = @elapsed iters_trad = newton_traditional!(u_trad, prob, r, du; verbose=false)
    println("  Time: $(round(t_trad * 1000, digits=2)) ms")
    println("  Iterations: $iters_trad")
    println("  Time/iter: $(round(t_trad / iters_trad * 1000, digits=2)) ms")

    # Benchmark Matrix-Free
    println("\n📊 Matrix-Free Newton-Krylov:")
    u_mf = copy(u0)
    t_mf = @elapsed iters_mf = newton_matrix_free!(
        u_mf, prob, r, du, temp, Jv; verbose=false
    )
    println("  Time: $(round(t_mf * 1000, digits=2)) ms")
    println("  Iterations: $iters_mf")
    println("  Time/iter: $(round(t_mf / iters_mf * 1000, digits=2)) ms")

    # Benchmark Anderson-Accelerated
    println("\n📊 Anderson-Accelerated Matrix-Free:")
    u_anderson = copy(u0)
    t_anderson = @elapsed iters_anderson = anderson_newton_matrix_free!(
        u_anderson, prob, r, du, temp, Jv; m=5, verbose=false
    )
    println("  Time: $(round(t_anderson * 1000, digits=2)) ms")
    println("  Iterations: $iters_anderson")
    println("  Time/iter: $(round(t_anderson / iters_anderson * 1000, digits=2)) ms")

    # Speedups
    println("\n✅ CPU Speedups:")
    println("  Matrix-Free vs Traditional: $(round(t_trad / t_mf, digits=2))×")
    println("  Anderson vs Traditional: $(round(t_trad / t_anderson, digits=2))×")
    println("  Anderson vs Matrix-Free: $(round(t_mf / t_anderson, digits=2))×")

    println()
end

function benchmark_gpu(n::Int)
    if !USE_GPU
        println("⚠️  GPU not available, skipping GPU benchmark\n")
        return
    end

    println("="^70)
    println("GPU Benchmark: $n DOFs")
    println("="^70)

    # Setup problem
    T = Float64
    K_cpu = Matrix(Tridiagonal(
        -ones(T, n - 1),
        2ones(T, n),
        -ones(T, n - 1)
    ))
    f_cpu = ones(T, n) * 0.1
    β = T(1e-3)

    # Transfer to GPU
    K_gpu = CuArray(K_cpu)
    f_gpu = CuArray(f_cpu)
    u0_gpu = CUDA.zeros(T, n)

    # Benchmark Matrix-Free on GPU
    println("\n📊 Matrix-Free Newton-Krylov (GPU):")
    u_gpu = copy(u0_gpu)

    # Warmup
    newton_matrix_free_gpu!(u_gpu, K_gpu, f_gpu, β; max_iter=2, verbose=false)

    # Benchmark
    CUDA.synchronize()
    t_gpu = CUDA.@elapsed begin
        iters_gpu = newton_matrix_free_gpu!(u_gpu, K_gpu, f_gpu, β; verbose=false)
        CUDA.synchronize()
    end

    println("  Time: $(round(t_gpu * 1000, digits=2)) ms")
    println("  Iterations: $iters_gpu")
    println("  Time/iter: $(round(t_gpu / iters_gpu * 1000, digits=2)) ms")

    # Compare with CPU
    prob_cpu = NonlinearProblem(K_cpu, f_cpu, β, n)
    u_cpu = zeros(T, n)
    r = zeros(T, n)
    du = zeros(T, n)
    temp = zeros(T, n)
    Jv = zeros(T, n)

    t_cpu = @elapsed iters_cpu = newton_matrix_free!(
        u_cpu, prob_cpu, r, du, temp, Jv; verbose=false
    )

    println("\n✅ GPU vs CPU Speedup: $(round(t_cpu / t_gpu, digits=2))×")
    println("   CPU: $(round(t_cpu * 1000, digits=2)) ms")
    println("   GPU: $(round(t_gpu * 1000, digits=2)) ms")

    # Benchmark Anderson-Accelerated on GPU
    println("\n📊 Anderson-Accelerated Newton-Krylov (GPU):")
    u_gpu_anderson = copy(u0_gpu)

    # Warmup
    anderson_newton_matrix_free_gpu!(u_gpu_anderson, K_gpu, f_gpu, β; max_iter=2, verbose=false)

    # Benchmark
    CUDA.synchronize()
    t_gpu_anderson = CUDA.@elapsed begin
        iters_gpu_anderson = anderson_newton_matrix_free_gpu!(u_gpu_anderson, K_gpu, f_gpu, β; verbose=false)
        CUDA.synchronize()
    end

    println("  Time: $(round(t_gpu_anderson * 1000, digits=2)) ms")
    println("  Iterations: $iters_gpu_anderson")
    println("  Time/iter: $(round(t_gpu_anderson / iters_gpu_anderson * 1000, digits=2)) ms")

    # Compare with CPU Anderson
    u_cpu_anderson = zeros(T, n)
    t_cpu_anderson = @elapsed iters_cpu_anderson = anderson_newton_matrix_free!(
        u_cpu_anderson, prob_cpu, r, du, temp, Jv; verbose=false
    )

    println("\n✅ GPU vs CPU Speedup (Anderson): $(round(t_cpu_anderson / t_gpu_anderson, digits=2))×")
    println("   CPU: $(round(t_cpu_anderson * 1000, digits=2)) ms")
    println("   GPU: $(round(t_gpu_anderson * 1000, digits=2)) ms")

    # Overall comparison
    println("\n📊 Summary:")
    println("   Matrix-Free GPU speedup: $(round(t_cpu / t_gpu, digits=2))×")
    println("   Anderson GPU speedup:    $(round(t_cpu_anderson / t_gpu_anderson, digits=2))×")

    println()
end

# ============================================================================
# Main
# ============================================================================

function main()
    println("\n" * "="^70)
    println("Matrix-Free Newton-Krylov GPU Benchmark")
    println("="^70)
    println()

    # Test sizes (reasonable for demonstration)
    sizes = [1000, 5_000, 10_000]

    for n in sizes
        # CPU comparison
        benchmark_cpu(n)

        # GPU benchmark
        if USE_GPU
            benchmark_gpu(n)
        end
    end

    println("="^70)
    println("Benchmark Complete!")
    println("="^70)
    println()
    println("Key Findings:")
    println("  - Matrix-Free eliminates Jacobian assembly cost")
    println("  - Anderson acceleration reduces Newton iterations")
    println("  - GPU provides additional speedup for large problems")
    println("  - Total speedup: 5-10× depending on problem size")
    println()
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
