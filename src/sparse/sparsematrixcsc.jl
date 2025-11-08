struct AssemblerSparsityPattern{Tv,Ti}
    K::SparseMatrixCSC{Tv,Ti}
    f::Vector{Tv}
    permutation::Vector{Int}
    sorteddofs::Vector{Int}
end

"""
    start_assemble(K::SparseMatrixCSC, [f::Vector])

Create an `AssemblerSparsityPattern` from the sparsity pattern in `K`
and optionally a "force" vector.
"""
function start_assemble(K::SparseMatrixCSC, f = eltype(K)[])
    if !isempty(f)
        @assert size(K, 1) == length(f)
    end
    AssemblerSparsityPattern(K, f, Int[], Int[])
end


"""
    assemble_local_vector!(f, dofs, fe)
Assembles the element residual `fe` into the global residual vector `f`.
"""
Base.@propagate_inbounds function assemble_local_vector!(f::AbstractVector, dofs::AbstractVector{Int}, fe::AbstractVector)
    @boundscheck checkbounds(f, dofs)
    @inbounds for i in 1:length(dofs)
        f[dofs[i]] += fe[i]
    end
end


function assemble_local!(A::AssemblerSparsityPattern, dofs::AbstractVector, Ke::AbstractMatrix, fe::AbstractVector)
    @assert !isempty(A.f)
    @boundscheck checkbounds(A.K, dofs, dofs)
    @boundscheck checkbounds(A.f, dofs)
    @inbounds assemble_local_vector!(A.f, dofs, fe)
    @inbounds assemble_local_matrix!(A, dofs, Ke)
end

"""
    assemble!(A::AssemblerSparsityPattern, dofs2, Ke)

Assemble a local dense element matrix `Ke` into the sparse matrix
wrapped by the assembler `A`.
location given by lists of indices `dofs1` and `dofs2`.

# Example

```julia
using SparseArrays
sparsity_pattern = sparse([1. 0 1; 1 0 1; 1 1 1])
fill!(sparsity_pattern, 0)
assembler = FEMSparse.start_assemble(sparsity_pattern)
dofs = [1, 3]
Ke = [1.0 2.0; 3.0 4.0]
FEMSparse.FEMSparse.assemble_local_matrix!(assembler, dofs, Ke)
Matrix(sparsity_pattern)

# output
julia> Matrix(sparsity_pattern)
3×3 Array{Float64,2}:
 1.0  0.0  2.0
 0.0  0.0  0.0
 3.0  0.0  4.0
```
"""
Base.@propagate_inbounds function assemble_local_matrix!(A::AssemblerSparsityPattern, dofs::AbstractVector{Int}, Ke::AbstractMatrix)
    permutation = A.permutation
    sorteddofs = A.sorteddofs
    K = A.K

    @boundscheck checkbounds(K, dofs, dofs)
    resize!(permutation, length(dofs))
    resize!(sorteddofs, length(dofs))
    copyto!(sorteddofs, dofs)
    sortperm2!(sorteddofs, permutation)

    current_col = 1
    @inbounds for Kcol in sorteddofs
        maxlookups = length(dofs)
        current_idx = 1
        for r in nzrange(K, Kcol)
            Kerow = permutation[current_idx]
            if K.rowval[r] == dofs[Kerow]
                Kecol = permutation[current_col]
                K.nzval[r] += Ke[Kerow, Kecol]
                current_idx += 1
            end
            current_idx > maxlookups && break
        end
        if current_idx <= maxlookups
            error("some row indices were not found")
        end
        current_col += 1
    end
end


##################
# Sort utilities #
##################
# We bundle a few sorting utilities here because the ones
# in have some unacceptable overhead.

# Sorts B and stores the permutation in `ii`
function sortperm2!(B, ii)
   @inbounds for i = 1:length(B)
      ii[i] = i
   end
   quicksort!(B, ii)
   return
end

function quicksort!(A, order, i=1,j=length(A))
    @inbounds if j > i
        if  j - i <= 12
           # Insertion sort for small groups is faster than Quicksort
           insertionsort!(A, order, i, j)
           return A
        end

        pivot = A[div(i+j,2)]
        left, right = i, j
        while left <= right
            while A[left] < pivot
                left += 1
            end
            while A[right] > pivot
                right -= 1
            end
            if left <= right
                A[left], A[right] = A[right], A[left]
                order[left], order[right] = order[right], order[left]

                left += 1
                right -= 1
            end
        end  # left <= right

        quicksort!(A,order, i,   right)
        quicksort!(A,order, left,j)
    end  # j > i

    return A
end

function insertionsort!(A, order, ii=1, jj=length(A))
    @inbounds for i = ii+1 : jj
        j = i - 1
        temp  = A[i]
        itemp = order[i]

        while true
            if j == ii-1
                break
            end
            if A[j] <= temp
                break
            end
            A[j+1] = A[j]
            order[j+1] = order[j]
            j -= 1
        end

        A[j+1] = temp
        order[j+1] = itemp
    end  # i
    return
end
