# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/GraphOrdering.jl/blob/master/LICENSE
#
# Graph ordering algorithms (RCM bandwidth minimization)
# Consolidated from GraphOrdering.jl

struct GraphOrderingResult
    perm :: Vector{Int}
    invperm :: Vector{Int}
    degrees :: Vector{Int}
    edge :: Vector{Int}
    dist :: Vector{Int}
end

"""
    bandwidth(G)

Calculate the bandwidth of the graph G.
"""

function bandwidth(G)
    bw = -1
    for (v, adj) in G
        for w in adj
            bw = max(bw, abs(v-w))
        end
    end
    return 2*bw + 1
end

"""
    symrcm(G, v)

Sparse reverse Cuthill-McKee ordering. `G` is graph presented as adjacency list
and `v` is starting vertex. Returns `Result` type, which contains bandwidth
minimizing permutation, inverse of permutation and some other results.
"""
function symrcm(G, v)
    n = length(G)
    permutation = zeros(Int, n)
    permutation[1] = v
    visited = zeros(Bool, n)
    visited[v] = true
    degrees = [length(G[i]) for i in 1:n]
    order = Base.Order.By(j -> degrees[j])
    connected = zeros(Int, n)
    edge = zeros(Int, n)
    dist = zeros(Int, n)
    nwrk = 0
    wrk = zeros(Int, nwrk)
    idx = 2

    for i=1:n

        v = permutation[i]
        adj = G[v]
        nadj = length(adj)

        aux = adj
        # aux is adj or wrk, depending is adj in order. If adj is not sorted,
        # copy degrees from adj to aux and make in-place sort
        if (!issorted(aux, order))
            aux = wrk
            resize!(aux, max(nadj, length(aux)))
            copyto!(aux, adj)
            sort!(aux, 1, nadj, InsertionSort, order)
        end

        for wi = 1:nadj
            w = aux[wi]
            visited[w] && continue
            visited[w] = true
            permutation[idx] = w
            edge[w] = v
            dist[w] = dist[v] + 1
            idx += 1
        end

    end

    perm = reverse(permutation)
    return GraphOrderingResult(perm, invperm(perm), degrees, edge, dist)
end

"""
    reorder(G, res)

Given result of `symrcm`, reorder nodes of graph `G`. Returns
new graph with nodes ordered corresponding the new permuation.
"""
function reorder(G, res::GraphOrderingResult)
    H = empty(G)
    for (v, adj) in G
        H[res.invperm[v]] = [res.invperm[w] for w in adj]
    end
    return H
end

