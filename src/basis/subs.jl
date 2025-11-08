# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/FEMBasis.jl/blob/master/LICENSE

function subs(p::Number, ::Any)
    return p
end

function subs(p::Symbol, data::Pair{Symbol, T}) where T
    k, v = data
    if p == k
        return v
    end
    return p
end

function subs(p::Symbol, data::NTuple{N,Pair{Symbol, T}}) where {N, T}
    for (k, v) in data
        if p == k
            return v
        end
    end
    return p
end

function subs(p::Expr, d::Pair)
    v = copy(p)
    for j in 2:length(p.args)
        v.args[j] = subs(v.args[j], d)
    end
    return v
end

"""
    subs(expression, data)

Given expression and pair(s) of `symbol => value` data, substitute to expression.

# Examples

Let us have polynomial `1 + u + v + u*v^2`, and substitute u=1 and v=2:
```julia
expression = :(1 + u + v + u*v^2)
data = (:u => 1.0, :v => 2.0)
subs(expression, data)
8.0
```

"""
function subs(p::Expr, data::NTuple{N,Pair{Symbol, T}}) where {N, T}
    for di in data
        p = subs(p, di)
    end
    return Calculus.simplify(p)
end
