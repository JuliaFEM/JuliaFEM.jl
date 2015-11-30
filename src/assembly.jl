# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

# Functions to handle global assembly of problem

type CAssembly
    interior_dofs :: Vector{Int}
    boundary_dofs :: Vector{Int}
    F :: Factorization
    Kc :: SparseMatrixCSC
    fc :: SparseMatrixCSC
    Ki :: SparseMatrixCSC
    fi :: SparseMatrixCSC
end

function assemble!(assembly::Assembly, problem::AllProblems, time::Float64, empty_assembly::Bool=true)
    if empty_assembly
        empty!(assembly)
    end
    for element in get_elements(problem)
        assemble!(assembly, problem, element, time)
    end
end

function assemble(problem::AllProblems, time::Float64)
    assembly = Assembly()
    for element in get_elements(problem)
        assemble!(assembly, problem, element, time)
    end
    return assembly
end

""" Return condensed system. """
function assemble(problem::FieldProblem, time::Float64, boundary_dofs::Vector{Int})
    assembly = Assembly()
    for element in get_elements(problem)
        assemble!(assembly, problem, element, time)
    end
    return condensate(assembly, boundary_dofs)
end

function condensate(assembly::Assembly, boundary_dofs_::Vector{Int})
    K = sparse(assembly.stiffness_matrix)
    all_dofs = unique(assembly.stiffness_matrix.I)
    boundary_dofs = intersect(all_dofs, boundary_dofs_)
    interior_dofs = setdiff(all_dofs, boundary_dofs_)
    dim = size(K, 1)
    f = sparse(assembly.force_vector, dim, 1)

    # check that matrix is symmetric
    asdf = maximum(abs(1/2*(K + K') - K))
    if asdf > 1.0e-6
        info(full(K))
        error("asdf $asdf > 1.0e-6")
    end
    
    K = 1/2*(K + K')
    F::Factorization = cholfact(K[interior_dofs, interior_dofs])

#   info("condensation: all dofs: ", all_dofs)
#   info("condensation: interior dofs: ", interior_dofs)
#   info("condensation: boundary dofs: ", boundary_dofs)
#   info("manually condensated")
#   Kman = K[boundary_dofs, boundary_dofs] - K[boundary_dofs,interior_dofs] * inv(full(K[interior_dofs, interior_dofs])) * K[interior_dofs, boundary_dofs]
#   info("\n$(full(Kman))")
    #info("K = \n$(full(K))")
    #Ki = K[interior_dofs, boundary_dofs]
    Ki = K[interior_dofs, boundary_dofs]
    fi = f[interior_dofs]
#   info("condensated using factorization")
#   LL = K[boundary_dofs, boundary_dofs] - K[boundary_dofs, interior_dofs] * (K[interior_dofs, interior_dofs] \ K[interior_dofs, boundary_dofs])
#   info(LL)

    Ks = F \ Ki
    Fs = F \ fi

    dim = size(K, 1)
    Kc = spzeros(dim, dim)
    fc = spzeros(dim, 1)
    Kc[boundary_dofs, boundary_dofs] = K[boundary_dofs, boundary_dofs] - Ki' * Ks
    fc[boundary_dofs] = f[boundary_dofs] - Ki' * Fs


    return CAssembly(interior_dofs, boundary_dofs, F, Kc, fc, Ki, fi)
end

function reconstruct!(ca::CAssembly, x::SparseMatrixCSC)
#   info("size of la = ", size(la))
#   info("size of ca.Ki = ", size(ca.Ki))
#   info("size of ca.fi = ", size(ca.fi))
#   info("size of la[ca.interior_dofs] = ", size(la[ca.interior_dofs]))
#   info("interior dofs: $(ca.interior_dofs)")
#   info("boundary dofs: $(ca.boundary_dofs)")
#   info("ca.fi = $(ca.fi')")
#   info("sol1 = ", full(ca.F \ ca.fi)')
#   info("sol2 = ", full(ca.F \ (ca.Ki*x[ca.boundary_dofs]))')
    x[ca.interior_dofs] += ca.F \ (ca.fi - ca.Ki*x[ca.boundary_dofs])
end

function Base.(:+)(ass1::Assembly, ass2::Assembly)
    mass_matrix = ass1.mass_matrix + ass2.mass_matrix
    stiffness_matrix = ass1.stiffness_matrix + ass2.stiffness_matrix
    force_vector = ass1.force_vector + ass2.force_vector
    return Assembly(mass_matrix, stiffness_matrix, force_vector)
end
