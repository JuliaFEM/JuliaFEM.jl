#!/usr/bin/env julia
#
# Multi-GPU Nodal Assembly Benchmark with MPI + CUDA
#
# Usage:
#   mpirun -np 2 julia --project=. benchmarks/multigpu_mpi_benchmark.jl
#   mpirun -np 4 julia --project=. benchmarks/multigpu_mpi_benchmark.jl
#
# Each MPI rank gets one GPU

using MPI
using CUDA
using LinearAlgebra
using Printf

MPI.Init()

const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)
const nranks = MPI.Comm_size(comm)

# Set GPU device based on rank
if CUDA.functional()
    CUDA.device!(rank % CUDA.ndevices())
    if rank == 0
        println("="^70)
        println("Multi-GPU Nodal Assembly Benchmark (MPI + CUDA)")
        println("="^70)
        println("MPI ranks: $nranks")
        println("CUDA devices: $(CUDA.ndevices())")
        println("CUDA functional: $(CUDA.functional())")
        println("="^70)
        println()
    end
else
    if rank == 0
        println("ERROR: CUDA not functional!")
        println("Install CUDA.jl: using Pkg; Pkg.add(\"CUDA\")")
    end
    MPI.Finalize()
    exit(1)
end

# ============================================================================
# Data Structures
# ============================================================================

struct Node
    id::Int32
    x::Float32
    y::Float32
    z::Float32
end

struct Element
    id::Int32
    connectivity::NTuple{8,Int32}  # Hex8
end

struct Partition
    rank::Int
    owned_nodes::UnitRange{Int}
    ghost_nodes::Vector{Int}
    local_elements::Vector{Int}
    node_to_elements::Vector{Vector{Int}}
    interface_neighbors::Vector{Int}  # Neighbor ranks
    interface_send::Dict{Int,Vector{Int}}  # rank → local DOF indices to send
    interface_recv::Dict{Int,Vector{Int}}  # rank → local DOF indices to receive
end

# ============================================================================
# Mesh Generation
# ============================================================================

function create_hex_mesh(nx, ny, nz)
    """Create structured hexahedral mesh"""
    n_nodes = nx * ny * nz
    n_elements = (nx - 1) * (ny - 1) * (nz - 1)

    nodes = Node[]
    for k in 1:nz, j in 1:ny, i in 1:nx
        node_id = Int32((k - 1) * nx * ny + (j - 1) * nx + i)
        push!(nodes, Node(node_id, Float32(i), Float32(j), Float32(k)))
    end

    elements = Element[]
    for k in 1:(nz-1), j in 1:(ny-1), i in 1:(nx-1)
        n1 = Int32((k - 1) * nx * ny + (j - 1) * nx + i)
        n2 = n1 + 1
        n3 = n2 + nx
        n4 = n1 + nx
        n5 = n1 + nx * ny
        n6 = n2 + nx * ny
        n7 = n3 + nx * ny
        n8 = n4 + nx * ny

        elem_id = Int32(length(elements) + 1)
        push!(elements, Element(elem_id, (n1, n2, n3, n4, n5, n6, n7, n8)))
    end

    return nodes, elements
end

function build_node_to_elements(nodes, elements)
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

function partition_mesh_for_rank(nodes, elements, my_rank, n_ranks)
    """Create partition for this MPI rank"""
    n_nodes = length(nodes)
    nodes_per_rank = ceil(Int, n_nodes / n_ranks)

    # Owned nodes
    start_node = my_rank * nodes_per_rank + 1
    end_node = min((my_rank + 1) * nodes_per_rank, n_nodes)
    owned_nodes = start_node:end_node

    node_to_elems = build_node_to_elements(nodes, elements)

    # Find local elements (touching owned nodes)
    local_elements = Int[]
    ghost_nodes = Set{Int}()

    for (elem_id, element) in enumerate(elements)
        if any(Int(nid) in owned_nodes for nid in element.connectivity)
            push!(local_elements, elem_id)

            for nid in element.connectivity
                if !(Int(nid) in owned_nodes)
                    push!(ghost_nodes, Int(nid))
                end
            end
        end
    end

    # Build local node_to_elements
    local_node_to_elems = [
        filter(eid -> eid in local_elements, node_to_elems[nid])
        for nid in owned_nodes
    ]

    # Find interface nodes with each neighbor
    interface_send = Dict{Int,Vector{Int}}()
    interface_recv = Dict{Int,Vector{Int}}()

    for neighbor_rank in 0:(n_ranks-1)
        if neighbor_rank == my_rank
            continue
        end

        neighbor_start = neighbor_rank * nodes_per_rank + 1
        neighbor_end = min((neighbor_rank + 1) * nodes_per_rank, n_nodes)
        neighbor_owned = neighbor_start:neighbor_end

        # Nodes I own that neighbor needs (I send)
        send_nodes = Int[]
        for elem_id in local_elements
            element = elements[elem_id]
            has_neighbor = any(Int(nid) in neighbor_owned for nid in element.connectivity)
            if has_neighbor
                for nid in element.connectivity
                    if Int(nid) in owned_nodes && !(Int(nid) in send_nodes)
                        push!(send_nodes, Int(nid))
                    end
                end
            end
        end

        # Nodes neighbor owns that I need (I receive)
        recv_nodes = Int[]
        for nid in ghost_nodes
            if Int(nid) in neighbor_owned
                push!(recv_nodes, Int(nid))
            end
        end

        if !isempty(send_nodes) || !isempty(recv_nodes)
            # Convert to local DOF indices
            send_dofs = Int[]
            for nid in send_nodes
                local_idx = nid - start_node + 1
                for d in 0:2
                    push!(send_dofs, (local_idx - 1) * 3 + d + 1)
                end
            end

            recv_dofs = Int[]
            for nid in recv_nodes
                ghost_idx = findfirst(==(nid), sort(collect(ghost_nodes)))
                for d in 0:2
                    # Ghost DOFs come after owned DOFs
                    push!(recv_dofs, length(owned_nodes) * 3 + (ghost_idx - 1) * 3 + d + 1)
                end
            end

            if !isempty(send_dofs)
                interface_send[neighbor_rank] = send_dofs
            end
            if !isempty(recv_dofs)
                interface_recv[neighbor_rank] = recv_dofs
            end
        end
    end

    interface_neighbors = sort(collect(keys(interface_send) ∪ keys(interface_recv)))

    return Partition(
        my_rank,
        owned_nodes,
        sort(collect(ghost_nodes)),
        local_elements,
        local_node_to_elems,
        interface_neighbors,
        interface_send,
        interface_recv
    )
end

# ============================================================================
# GPU Kernel: Nodal Assembly
# ============================================================================

function gpu_matvec_kernel!(
    y::CuDeviceArray{Float32,1},
    x::CuDeviceArray{Float32,1},
    nodes::CuDeviceArray{Node,1},
    elements::CuDeviceArray{Element,1},
    node_to_elems_offsets::CuDeviceArray{Int32,1},
    node_to_elems_data::CuDeviceArray{Int32,1},
    n_owned_nodes::Int32,
)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    if idx > n_owned_nodes
        return
    end

    # This thread processes owned node idx
    node = nodes[idx]

    dof_start = (idx - 1) * 3 + 1

    # Initialize nodal contribution
    y1 = Float32(0.0)
    y2 = Float32(0.0)
    y3 = Float32(0.0)

    # Get connected elements using CSR-like format (1-based indexing)
    if idx + Int32(1) > length(node_to_elems_offsets)
        return
    end

    elem_start = node_to_elems_offsets[idx] + Int32(1)
    elem_end = node_to_elems_offsets[idx+Int32(1)]

    for i in elem_start:elem_end
        if i > length(node_to_elems_data)
            return
        end
        elem_id = node_to_elems_data[i]
        if elem_id > length(elements)
            return
        end
        element = elements[elem_id]

        # Add contribution from all nodes in this element
        for j in 1:8
            nid = element.connectivity[j]
            x_dof_start = (nid - 1) * 3 + 1

            # Mock stiffness contribution
            y1 += Float32(0.1) * x[x_dof_start]
            y2 += Float32(0.1) * x[x_dof_start+1]
            y3 += Float32(0.1) * x[x_dof_start+2]
        end
    end

    # Write to output
    y[dof_start] = y1
    y[dof_start+1] = y2
    y[dof_start+2] = y3

    return nothing
end

# ============================================================================
# Multi-GPU Communication
# ============================================================================

function exchange_ghost_values!(
    x_local::CuArray{Float32,1},
    partition::Partition,
    comm::MPI.Comm
)
    """Exchange interface DOF values between MPI ranks"""

    # Prepare send/recv buffers on CPU
    send_bufs = Dict{Int,Vector{Float32}}()
    recv_bufs = Dict{Int,Vector{Float32}}()

    # Copy data from GPU to CPU for sending
    x_cpu = Array(x_local)

    for neighbor in partition.interface_neighbors
        if haskey(partition.interface_send, neighbor)
            send_dofs = partition.interface_send[neighbor]
            send_bufs[neighbor] = x_cpu[send_dofs]
        end

        if haskey(partition.interface_recv, neighbor)
            recv_dofs = partition.interface_recv[neighbor]
            recv_bufs[neighbor] = zeros(Float32, length(recv_dofs))
        end
    end

    # MPI communication
    requests = MPI.Request[]

    # Post receives
    for neighbor in partition.interface_neighbors
        if haskey(recv_bufs, neighbor)
            req = MPI.Irecv!(recv_bufs[neighbor], comm; source=neighbor, tag=neighbor)
            push!(requests, req)
        end
    end

    # Post sends
    for neighbor in partition.interface_neighbors
        if haskey(send_bufs, neighbor)
            req = MPI.Isend(send_bufs[neighbor], comm; dest=neighbor, tag=partition.rank)
            push!(requests, req)
        end
    end

    # Wait for all communications
    MPI.Waitall(requests)

    # Copy received data back to GPU
    for neighbor in partition.interface_neighbors
        if haskey(partition.interface_recv, neighbor)
            recv_dofs = partition.interface_recv[neighbor]
            x_cpu[recv_dofs] .= recv_bufs[neighbor]
        end
    end

    # Update GPU array
    copyto!(x_local, x_cpu)
end

# ============================================================================
# Benchmark
# ============================================================================

function run_multigpu_benchmark(nx, ny, nz, n_warmup=5, n_runs=10)
    if rank == 0
        println("\n" * "="^70)
        println("Multi-GPU Benchmark: $nx × $ny × $nz mesh")
        println("="^70)
    end

    # Create full mesh on all ranks
    nodes, elements = create_hex_mesh(nx, ny, nz)

    if rank == 0
        println("  Total nodes: ", length(nodes))
        println("  Total elements: ", length(elements))
        println("  Total DOFs: ", 3 * length(nodes))
    end

    # Partition for this rank
    partition = partition_mesh_for_rank(nodes, elements, rank, nranks)

    n_owned = length(partition.owned_nodes)
    n_ghost = length(partition.ghost_nodes)
    n_local_dofs = 3 * (n_owned + n_ghost)

    println("Rank $rank: $n_owned owned nodes, $n_ghost ghost nodes, " *
            "$(length(partition.local_elements)) elements")

    # Prepare GPU data
    local_nodes = [nodes[i] for i in vcat(collect(partition.owned_nodes), partition.ghost_nodes)]

    # Create mapping from global node ID to local index
    global_to_local_node = Dict{Int,Int32}()
    for (local_idx, global_nid) in enumerate(vcat(collect(partition.owned_nodes), partition.ghost_nodes))
        global_to_local_node[global_nid] = Int32(local_idx)
    end

    # Remap element connectivity to local node indices
    local_elements = Element[]
    for global_eid in partition.local_elements
        element = elements[global_eid]
        # Convert global node IDs to local indices
        local_conn = ntuple(8) do i
            global_nid = Int(element.connectivity[i])
            global_to_local_node[global_nid]
        end
        push!(local_elements, Element(element.id, local_conn))
    end

    # Create mapping from global element ID to local index (for CSR data)
    global_to_local_elem = Dict{Int,Int}()
    for (local_idx, global_id) in enumerate(partition.local_elements)
        global_to_local_elem[global_id] = local_idx
    end

    # Convert node_to_elements to GPU-friendly flat format
    # Format: offsets array + flat data array (CSR-like)
    # IMPORTANT: Convert global element IDs to local indices
    node_to_elems_offsets = Int32[0]
    node_to_elems_data = Int32[]
    for arr in partition.node_to_elements
        # Map global element IDs to local indices
        local_indices = [global_to_local_elem[global_id] for global_id in arr]
        append!(node_to_elems_data, Int32.(local_indices))
        push!(node_to_elems_offsets, length(node_to_elems_data))
    end

    # Debug: check element ID range
    if rank == 0 && length(node_to_elems_data) > 0
        min_elem_id = minimum(node_to_elems_data)
        max_elem_id = maximum(node_to_elems_data)
        println("\nCSR data element ID range: $min_elem_id to $max_elem_id")
        println("Local elements array size: $(length(local_elements))")
        if max_elem_id > length(local_elements)
            println("❌ WARNING: Element ID $max_elem_id > array size $(length(local_elements))")
        end
    end

    # Transfer to GPU
    nodes_gpu = CuArray(local_nodes)
    elements_gpu = CuArray(local_elements)
    node_to_elems_offsets_gpu = CuArray(node_to_elems_offsets)
    node_to_elems_data_gpu = CuArray(node_to_elems_data)

    # Debug: print array sizes
    if rank == 0
        println("\nArray sizes on GPU:")
        println("  nodes: $(length(nodes_gpu))")
        println("  elements: $(length(elements_gpu))")
        println("  node_to_elems_offsets: $(length(node_to_elems_offsets_gpu))")
        println("  node_to_elems_data: $(length(node_to_elems_data_gpu))")
        println("  Expected offsets length: $(n_owned + 1)")
    end

    # Test vectors
    x_local = CUDA.rand(Float32, n_local_dofs)
    y_local = CUDA.zeros(Float32, n_local_dofs)

    # Kernel launch parameters
    threads_per_block = 256
    n_blocks = cld(n_owned, threads_per_block)

    if rank == 0
        println("\nGPU configuration:")
        println("  Threads per block: $threads_per_block")
        println("  Blocks per rank: $n_blocks")
    end

    # Warmup
    for _ in 1:n_warmup
        exchange_ghost_values!(x_local, partition, comm)
        CUDA.@sync @cuda threads = threads_per_block blocks = n_blocks gpu_matvec_kernel!(
            y_local, x_local, nodes_gpu, elements_gpu,
            node_to_elems_offsets_gpu, node_to_elems_data_gpu, Int32(n_owned)
        )
    end

    MPI.Barrier(comm)

    # Benchmark
    times = Float64[]
    comm_times = Float64[]
    compute_times = Float64[]

    for _ in 1:n_runs
        t_start = time_ns()

        # Communication
        t_comm_start = time_ns()
        exchange_ghost_values!(x_local, partition, comm)
        MPI.Barrier(comm)
        t_comm_end = time_ns()

        # Computation
        t_compute_start = time_ns()
        CUDA.@sync @cuda threads = threads_per_block blocks = n_blocks gpu_matvec_kernel!(
            y_local, x_local, nodes_gpu, elements_gpu,
            node_to_elems_offsets_gpu, node_to_elems_data_gpu, Int32(n_owned)
        )
        MPI.Barrier(comm)
        t_compute_end = time_ns()

        t_end = time_ns()

        push!(times, (t_end - t_start) / 1e9)
        push!(comm_times, (t_comm_end - t_comm_start) / 1e9)
        push!(compute_times, (t_compute_end - t_compute_start) / 1e9)
    end

    # Gather results
    local_time = minimum(times)
    local_comm = minimum(comm_times)
    local_compute = minimum(compute_times)

    all_times = MPI.Gather(local_time, 0, comm)
    all_comm = MPI.Gather(local_comm, 0, comm)
    all_compute = MPI.Gather(local_compute, 0, comm)

    if rank == 0
        println("\nResults:")
        println("  Rank | Owned Nodes | Total Time | Comm Time | Compute Time | Comm %")
        println("  " * "-"^70)
        for r in 0:(nranks-1)
            nodes_str = lpad(string(length(partition.owned_nodes)), 11)
            total_str = @sprintf("%.3f ms", all_times[r+1] * 1000)
            comm_str = @sprintf("%.3f ms", all_comm[r+1] * 1000)
            compute_str = @sprintf("%.3f ms", all_compute[r+1] * 1000)
            comm_pct = @sprintf("%.1f%%", all_comm[r+1] / all_times[r+1] * 100)

            println("  $r    | $nodes_str  | $(lpad(total_str, 10)) | " *
                    "$(lpad(comm_str, 9)) | $(lpad(compute_str, 12)) | $(lpad(comm_pct, 6))")
        end

        max_time = maximum(all_times)
        avg_compute = sum(all_compute) / length(all_compute)
        avg_comm = sum(all_comm) / length(all_comm)

        println("\n  Maximum time: ", @sprintf("%.3f ms", max_time * 1000))
        println("  Average compute: ", @sprintf("%.3f ms", avg_compute * 1000))
        println("  Average communication: ", @sprintf("%.3f ms", avg_comm * 1000))
        println("  Communication overhead: ", @sprintf("%.1f%%", avg_comm / max_time * 100))

        throughput = length(nodes) / max_time / 1e6
        println("  Throughput: ", @sprintf("%.2f Mnodes/s", throughput))
    end
end

# ============================================================================
# Main
# ============================================================================

if rank == 0
    println("Starting benchmarks...")
    println()
end

# Run benchmarks with increasing mesh sizes
run_multigpu_benchmark(30, 30, 30)
run_multigpu_benchmark(50, 50, 50)
run_multigpu_benchmark(70, 70, 70)

if rank == 0
    println("\n" * "="^70)
    println("✓ Multi-GPU Benchmark Complete")
    println("="^70)
end

MPI.Finalize()
