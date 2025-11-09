#!/usr/bin/env julia
#
# Nodal Assembly Scalability Benchmark
#
# Tests:
# 1. Single-threaded baseline
# 2. Multi-threaded scaling (2, 4, 8 threads)
# 3. Actual speedup measurements
# 4. Interface communication overhead
#
# Run with: julia --project=. -t 8 benchmarks/nodal_assembly_scalability.jl

using LinearAlgebra
using Printf
using Base.Threads

println("="^70)
println("Nodal Assembly Scalability Benchmark")
println("="^70)
println()
println("Julia threads available: ", nthreads())
println()

# ============================================================================
# Data Structures
# ============================================================================

struct Node
    id::Int
    x::Float64
    y::Float64
    z::Float64
end

struct Element
    id::Int
    connectivity::NTuple{8,Int}  # Hex8
end

struct Partition
    rank::Int
    owned_nodes::UnitRange{Int}
    ghost_nodes::Vector{Int}
    local_elements::Vector{Int}
    node_to_elements::Vector{Vector{Int}}
    interface_nodes::Dict{Int,Vector{Int}}
end

# ============================================================================
# Mesh Generation
# ============================================================================

function create_hex_mesh(nx, ny, nz)
    """Create structured hexahedral mesh"""
    n_nodes = nx * ny * nz
    n_elements = (nx - 1) * (ny - 1) * (nz - 1)

    # Create nodes
    nodes = Node[]
    for k in 1:nz, j in 1:ny, i in 1:nx
        node_id = (k - 1) * nx * ny + (j - 1) * nx + i
        push!(nodes, Node(node_id, Float64(i), Float64(j), Float64(k)))
    end

    # Create elements (Hex8)
    elements = Element[]
    for k in 1:(nz-1), j in 1:(ny-1), i in 1:(nx-1)
        n1 = (k - 1) * nx * ny + (j - 1) * nx + i
        n2 = n1 + 1
        n3 = n2 + nx
        n4 = n1 + nx
        n5 = n1 + nx * ny
        n6 = n2 + nx * ny
        n7 = n3 + nx * ny
        n8 = n4 + nx * ny

        elem_id = length(elements) + 1
        push!(elements, Element(elem_id, (n1, n2, n3, n4, n5, n6, n7, n8)))
    end

    return nodes, elements
end

function build_node_to_elements(nodes, elements)
    """Build inverse connectivity"""
    node_to_elems = [Int[] for _ in 1:length(nodes)]

    for (elem_id, element) in enumerate(elements)
        for node_id in element.connectivity
            push!(node_to_elems[node_id], elem_id)
        end
    end

    return node_to_elems
end

# ============================================================================
# Partitioning
# ============================================================================

function partition_mesh(nodes, elements, n_partitions)
    """Partition mesh by nodes"""
    n_nodes = length(nodes)
    nodes_per_partition = ceil(Int, n_nodes / n_partitions)

    node_to_elems = build_node_to_elements(nodes, elements)

    partitions = Partition[]

    for rank in 0:(n_partitions-1)
        # Owned nodes
        start_node = rank * nodes_per_partition + 1
        end_node = min((rank + 1) * nodes_per_partition, n_nodes)
        owned_nodes = start_node:end_node

        # Find elements touching owned nodes
        local_elements = Int[]
        ghost_nodes = Set{Int}()

        for (elem_id, element) in enumerate(elements)
            if any(nid in owned_nodes for nid in element.connectivity)
                push!(local_elements, elem_id)

                # Mark ghost nodes
                for nid in element.connectivity
                    if !(nid in owned_nodes)
                        push!(ghost_nodes, nid)
                    end
                end
            end
        end

        # Build local node_to_elements (owned nodes only)
        local_node_to_elems = [node_to_elems[nid] for nid in owned_nodes]

        # Find interface nodes (owned nodes that couple to other partitions)
        interface = Dict{Int,Vector{Int}}()
        for neighbor_rank in 0:(n_partitions-1)
            if neighbor_rank == rank
                continue
            end

            neighbor_start = neighbor_rank * nodes_per_partition + 1
            neighbor_end = min((neighbor_rank + 1) * nodes_per_partition, n_nodes)
            neighbor_owned = neighbor_start:neighbor_end

            interface_with_neighbor = Int[]
            for elem_id in local_elements
                element = elements[elem_id]
                has_owned = any(nid in owned_nodes for nid in element.connectivity)
                has_neighbor = any(nid in neighbor_owned for nid in element.connectivity)

                if has_owned && has_neighbor
                    for nid in element.connectivity
                        if nid in owned_nodes && !(nid in interface_with_neighbor)
                            push!(interface_with_neighbor, nid)
                        end
                    end
                end
            end

            if !isempty(interface_with_neighbor)
                interface[neighbor_rank] = interface_with_neighbor
            end
        end

        partition = Partition(
            rank,
            owned_nodes,
            collect(ghost_nodes),
            local_elements,
            local_node_to_elems,
            interface
        )

        push!(partitions, partition)
    end

    return partitions
end

# ============================================================================
# Nodal Assembly (Matrix-Vector Product)
# ============================================================================

function matvec_single_threaded!(y, x, nodes, elements, node_to_elements)
    """Single-threaded nodal assembly"""
    fill!(y, 0.0)

    for node_id in 1:length(nodes)
        node = nodes[node_id]

        # Get node DOFs
        dof_start = (node_id - 1) * 3 + 1
        y_nodal = zeros(3)

        # Gather from connected elements
        for elem_id in node_to_elements[node_id]
            element = elements[elem_id]

            # Mock element contribution (just for timing)
            for nid in element.connectivity
                x_dof_start = (nid - 1) * 3 + 1
                for d in 1:3
                    y_nodal[d] += 0.1 * x[x_dof_start+d-1]
                end
            end
        end

        # Write to global
        for d in 1:3
            y[dof_start+d-1] = y_nodal[d]
        end
    end
end

function matvec_multi_threaded!(y, x, nodes, elements, node_to_elements)
    """Multi-threaded nodal assembly (direct, no partitioning)"""
    fill!(y, 0.0)

    @threads for node_id in 1:length(nodes)
        node = nodes[node_id]

        dof_start = (node_id - 1) * 3 + 1
        y_nodal = zeros(3)

        for elem_id in node_to_elements[node_id]
            element = elements[elem_id]

            for nid in element.connectivity
                x_dof_start = (nid - 1) * 3 + 1
                for d in 1:3
                    y_nodal[d] += 0.1 * x[x_dof_start+d-1]
                end
            end
        end

        for d in 1:3
            y[dof_start+d-1] = y_nodal[d]
        end
    end
end

function matvec_partitioned!(y, x, nodes, elements, partitions)
    """Multi-threaded with explicit partitioning (simulates multi-GPU)"""
    fill!(y, 0.0)

    # Each partition processed by one thread
    @threads for partition in partitions
        # Process owned nodes
        for (local_idx, node_id) in enumerate(partition.owned_nodes)
            node = nodes[node_id]

            dof_start = (node_id - 1) * 3 + 1
            y_nodal = zeros(3)

            for elem_id in partition.node_to_elements[local_idx]
                element = elements[elem_id]

                for nid in element.connectivity
                    x_dof_start = (nid - 1) * 3 + 1
                    for d in 1:3
                        y_nodal[d] += 0.1 * x[x_dof_start+d-1]
                    end
                end
            end

            for d in 1:3
                y[dof_start+d-1] = y_nodal[d]
            end
        end
    end
end

# ============================================================================
# Benchmarks
# ============================================================================

function run_benchmark(name, nx, ny, nz, n_warmup=2, n_runs=10)
    println("\n" * "="^70)
    println("Benchmark: $name")
    println("  Mesh: $nx × $ny × $nz = $(nx*ny*nz) nodes, $((nx-1)*(ny-1)*(nz-1)) elements")
    println("="^70)

    # Create mesh
    print("Creating mesh... ")
    nodes, elements = create_hex_mesh(nx, ny, nz)
    node_to_elements = build_node_to_elements(nodes, elements)
    n_dofs = 3 * length(nodes)
    println("✓")

    println("  Nodes: ", length(nodes))
    println("  Elements: ", length(elements))
    println("  DOFs: ", n_dofs)
    println("  Avg elements/node: ", sum(length.(node_to_elements)) / length(nodes))

    # Test vectors
    x = randn(n_dofs)
    y_ref = zeros(n_dofs)
    y_test = zeros(n_dofs)

    # ========================================================================
    # 1. Single-threaded baseline
    # ========================================================================
    println("\n1. Single-threaded baseline:")

    # Warmup
    for _ in 1:n_warmup
        matvec_single_threaded!(y_ref, x, nodes, elements, node_to_elements)
    end

    # Benchmark
    times = Float64[]
    for _ in 1:n_runs
        t_start = time_ns()
        matvec_single_threaded!(y_ref, x, nodes, elements, node_to_elements)
        t_end = time_ns()
        push!(times, (t_end - t_start) / 1e9)
    end

    t_single = minimum(times)
    println("  Time: ", @sprintf("%.3f ms", t_single * 1000))
    println("  Throughput: ", @sprintf("%.2f Mnodes/s", length(nodes) / t_single / 1e6))

    # ========================================================================
    # 2. Multi-threaded (all available threads)
    # ========================================================================
    if nthreads() > 1
        println("\n2. Multi-threaded ($(nthreads()) threads):")

        # Warmup
        for _ in 1:n_warmup
            matvec_multi_threaded!(y_test, x, nodes, elements, node_to_elements)
        end

        # Verify correctness
        error = norm(y_test - y_ref) / norm(y_ref)
        println("  Verification: ", error < 1e-10 ? "✓ PASS" : "✗ FAIL (error=$error)")

        # Benchmark
        times = Float64[]
        for _ in 1:n_runs
            t_start = time_ns()
            matvec_multi_threaded!(y_test, x, nodes, elements, node_to_elements)
            t_end = time_ns()
            push!(times, (t_end - t_start) / 1e9)
        end

        t_multi = minimum(times)
        speedup = t_single / t_multi
        efficiency = speedup / nthreads() * 100

        println("  Time: ", @sprintf("%.3f ms", t_multi * 1000))
        println("  Speedup: ", @sprintf("%.2fx", speedup))
        println("  Efficiency: ", @sprintf("%.1f%%", efficiency))
        println("  Throughput: ", @sprintf("%.2f Mnodes/s", length(nodes) / t_multi / 1e6))
    end

    # ========================================================================
    # 3. Partitioned (simulates multi-GPU)
    # ========================================================================
    if nthreads() >= 4
        n_partitions = 4
        println("\n3. Partitioned ($n_partitions partitions, 4 threads):")

        # Create partitions
        print("  Creating partitions... ")
        partitions = partition_mesh(nodes, elements, n_partitions)
        println("✓")

        # Print partition info
        for partition in partitions
            n_owned = length(partition.owned_nodes)
            n_ghost = length(partition.ghost_nodes)
            n_interface = sum(length.(values(partition.interface_nodes)))
            interface_pct = n_interface / n_owned * 100

            println("    Partition $(partition.rank): $n_owned owned, $n_ghost ghost, " *
                    "$n_interface interface ($(round(interface_pct, digits=1))%)")
        end

        # Warmup
        for _ in 1:n_warmup
            matvec_partitioned!(y_test, x, nodes, elements, partitions)
        end

        # Verify correctness
        error = norm(y_test - y_ref) / norm(y_ref)
        println("  Verification: ", error < 1e-10 ? "✓ PASS" : "✗ FAIL (error=$error)")

        # Benchmark
        times = Float64[]
        for _ in 1:n_runs
            t_start = time_ns()
            matvec_partitioned!(y_test, x, nodes, elements, partitions)
            t_end = time_ns()
            push!(times, (t_end - t_start) / 1e9)
        end

        t_partitioned = minimum(times)
        speedup = t_single / t_partitioned
        efficiency = speedup / n_partitions * 100

        println("  Time: ", @sprintf("%.3f ms", t_partitioned * 1000))
        println("  Speedup: ", @sprintf("%.2fx", speedup))
        println("  Efficiency: ", @sprintf("%.1f%%", efficiency))
        println("  Throughput: ", @sprintf("%.2f Mnodes/s", length(nodes) / t_partitioned / 1e6))
    end

    println()
end

# ============================================================================
# Run Benchmarks
# ============================================================================

# Small mesh
run_benchmark("Small Mesh", 20, 20, 20)

# Medium mesh
run_benchmark("Medium Mesh", 40, 40, 40)

# Large mesh (if enough threads)
if nthreads() >= 4
    run_benchmark("Large Mesh", 60, 60, 60)
end

println("="^70)
println("✓ Benchmark Complete")
println("="^70)
println()
println("Notes:")
println("  - Single-threaded: Baseline performance")
println("  - Multi-threaded: All available threads, direct parallelization")
println("  - Partitioned: Simulates multi-GPU with explicit partitions")
println("  - Efficiency = Speedup / N_threads × 100%")
println("  - Near 100% efficiency = perfect scaling")
println()
