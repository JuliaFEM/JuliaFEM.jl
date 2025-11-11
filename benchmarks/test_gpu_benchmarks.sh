#!/bin/bash
# Quick test script for GPU benchmarks

echo "======================================================================"
echo "GPU Benchmark Quick Test"
echo "======================================================================"
echo ""

# Check Julia
echo "Checking Julia installation..."
if ! command -v julia &> /dev/null; then
    echo "❌ Julia not found! Please install Julia 1.9+"
    exit 1
fi

julia_version=$(julia --version)
echo "✅ Found: $julia_version"
echo ""

# Check GPU
echo "Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo ""
else
    echo "⚠️  No NVIDIA GPU detected. Benchmarks will run CPU-only."
    echo ""
fi

# Check CUDA.jl
echo "Checking CUDA.jl..."
julia --project=. -e '
using Pkg
try
    using CUDA
    if CUDA.functional()
        println("✅ CUDA.jl functional: ", CUDA.name(CUDA.device()))
    else
        println("⚠️  CUDA.jl installed but GPU not functional")
    end
catch
    println("⚠️  CUDA.jl not installed. Run: Pkg.add(\"CUDA\")")
end
' 2>/dev/null
echo ""

# Run quick state management test (small size)
echo "======================================================================"
echo "Test 1: State Management (10K elements)"
echo "======================================================================"
julia --project=. -e '
n = 10_000
println("Running state management benchmark with $n elements...")
include("benchmarks/gpu_state_management_benchmark.jl")

# Override main() to run smaller test
Δε_p, Δα, elements_s1, geometry_s2, state_s2 = setup_benchmark(n)

# CPU test
println("\n📊 CPU Test:")
state_s2_copy = deepcopy(state_s2)
t = @elapsed update_state_strategy2_cpu!(state_s2_copy, Δε_p, Δα)
println("Time: $(round(t * 1000, digits=2)) ms")
println("✅ CPU benchmark works!")

# GPU test (if available)
if USE_GPU
    println("\n📊 GPU Test:")
    try
        T = Float64
        state_gpu = AssemblyState{T}(
            CUDA.zeros(T, n * 6),
            CUDA.zeros(T, n),
            n
        )
        Δε_p_flat = zeros(T, n * 6)
        Δα_gpu = CuArray(Δα)
        for i in 1:n
            offset = (i - 1) * 6
            ε = Δε_p[i]
            Δε_p_flat[offset + 1] = ε[1, 1]
            Δε_p_flat[offset + 2] = ε[2, 2]
            Δε_p_flat[offset + 3] = ε[3, 3]
            Δε_p_flat[offset + 4] = ε[1, 2]
            Δε_p_flat[offset + 5] = ε[1, 3]
            Δε_p_flat[offset + 6] = ε[2, 3]
        end
        Δε_p_flat_gpu = CuArray(Δε_p_flat)
        
        update_state_strategy2_gpu!(state_gpu, Δε_p_flat_gpu, Δα_gpu)
        println("✅ GPU benchmark works!")
    catch e
        println("⚠️  GPU test failed: $e")
    end
end
'
echo ""

# Run quick matrix-free test (small size)
echo "======================================================================"
echo "Test 2: Matrix-Free Newton-Krylov (1K DOFs)"
echo "======================================================================"
julia --project=. -e '
n = 1000
println("Running matrix-free benchmark with $n DOFs...")
include("benchmarks/matrix_free_gpu_benchmark.jl")

# Override to run small test
T = Float64
K = Matrix(Tridiagonal(-ones(T, n-1), 2ones(T, n), -ones(T, n-1)))
f = ones(T, n) * 0.1
β = T(1e-3)
prob = NonlinearProblem(K, f, β, n)

println("\n📊 CPU Test:")
u = zeros(T, n)
r = zeros(T, n)
du = zeros(T, n)
temp = zeros(T, n)
Jv = zeros(T, n)

t = @elapsed iters = newton_matrix_free!(u, prob, r, du, temp, Jv; 
    max_iter=10, verbose=false)
println("Time: $(round(t * 1000, digits=2)) ms")
println("Iterations: $iters")
println("✅ CPU benchmark works!")

if USE_GPU
    println("\n📊 GPU Test:")
    try
        K_gpu = CuArray(K)
        f_gpu = CuArray(f)
        u_gpu = CUDA.zeros(T, n)
        
        t_gpu = CUDA.@elapsed begin
            iters_gpu = newton_matrix_free_gpu!(u_gpu, K_gpu, f_gpu, β; 
                max_iter=10, verbose=false)
            CUDA.synchronize()
        end
        
        println("Time: $(round(t_gpu * 1000, digits=2)) ms")
        println("Iterations: $iters_gpu")
        println("✅ GPU benchmark works!")
    catch e
        println("⚠️  GPU test failed: $e")
    end
end
'
echo ""

echo "======================================================================"
echo "Quick Test Complete!"
echo "======================================================================"
echo ""
echo "To run full benchmarks:"
echo "  julia --project=. benchmarks/gpu_state_management_benchmark.jl"
echo "  julia --project=. benchmarks/matrix_free_gpu_benchmark.jl"
echo ""
