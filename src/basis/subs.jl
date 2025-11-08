# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/jl/blob/master/LICENSE

# Minimal simplify for subs (from SymDiff.jl by Jukka Aho)
simplify_local(f::Union{Number,Symbol}) = f

function simplify_local(ex::Expr)
    @assert ex.head == :call
    op = first(ex.args)

    if op == :*
        args = simplify_local.(ex.args[2:end])
        0 in args && return 0
        filter!(k -> !(isa(k, Number) && k == 1), args)
        length(args) == 0 && return 1
        length(args) == 1 && return first(args)
        # If all args are numbers, evaluate
        if all(isa(a, Number) for a in args)
            return prod(args)
        end
        return Expr(:call, :*, args...)
    elseif op == :+
        args = simplify_local.(ex.args[2:end])
        filter!(k -> !isa(k, Number) || k != 0, args)
        length(args) == 0 && return 0
        length(args) == 1 && return first(args)
        # If all args are numbers, evaluate
        if all(isa(a, Number) for a in args)
            return sum(args)
        end
        return Expr(:call, :+, args...)
    elseif op == :-
        args = simplify_local.(ex.args[2:end])
        filter!(k -> !isa(k, Number) || k != 0, args)
        length(args) == 0 && return 0
        length(args) == 1 && return first(args)
        # If all args are numbers, evaluate
        if all(isa(a, Number) for a in args)
            return length(args) == 2 ? args[1] - args[2] : -args[1]
        end
        return Expr(:call, :-, args...)
    elseif op == :^
        args = simplify_local.(ex.args[2:end])
        # If all args are numbers, evaluate
        if all(isa(a, Number) for a in args)
            return args[1]^args[2]
        end
        return Expr(:call, :^, args...)
    elseif op == :/
        args = simplify_local.(ex.args[2:end])
        # If all args are numbers, evaluate
        if all(isa(a, Number) for a in args)
            return args[1] / args[2]
        end
        return Expr(:call, :/, args...)
    else
        args = simplify_local.(ex.args[2:end])
        return Expr(:call, op, args...)
    end
end

function subs(p::Number, ::Any)
    return p
end

function subs(p::Symbol, data::Pair{Symbol,T}) where T
    k, v = data
    if p == k
        return v
    end
    return p
end

function subs(p::Symbol, data::NTuple{N,Pair{Symbol,T}}) where {N,T}
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
function subs(p::Expr, data::NTuple{N,Pair{Symbol,T}}) where {N,T}
    for di in data
        p = subs(p, di)
    end
    return simplify_local(p)  # Use local simplify
end
